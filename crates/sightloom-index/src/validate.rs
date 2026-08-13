//! Structural validation for `VisionIndex` documents.
//!
//! - [`VisionIndex::validate_fast`] — header + cheap integrity
//! - [`VisionIndex::validate_full`] — referential integrity, finite geometry, ordering
//! - [`VisionIndex::repair_plan`] — suggested repairs for full-report issues

use crate::{MemoryError, VisionIndex};
use sightloom_core::{EventId, SourceId, SubjectId, TrackId};

/// Severity of a validation finding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValidationSeverity {
    /// Document cannot be trusted for continue-ingest without repair.
    Error,
    /// Suspicious but loadable.
    Warning,
    /// Informational note.
    Info,
}

/// One validation finding with an object path and optional repair hint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationIssue {
    /// JSON-ish path to the offending object (e.g. `tracks[3].mask_ref`).
    pub path: String,
    /// Severity.
    pub severity: ValidationSeverity,
    /// Human-readable description.
    pub message: String,
    /// Optional suggested repair.
    pub repair: Option<String>,
}

/// Full validation report.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ValidationReport {
    /// Findings in discovery order.
    pub issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    /// True when any error-severity issue is present.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.issues
            .iter()
            .any(|issue| issue.severity == ValidationSeverity::Error)
    }

    /// Converts the report into a unit result.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Validation`] when errors are present.
    pub fn into_result(self) -> Result<(), MemoryError> {
        if self.has_errors() {
            let summary = self
                .issues
                .iter()
                .filter(|issue| issue.severity == ValidationSeverity::Error)
                .map(|issue| format!("{}: {}", issue.path, issue.message))
                .collect::<Vec<_>>()
                .join("; ");
            Err(MemoryError::Validation(summary))
        } else {
            Ok(())
        }
    }

    fn push(
        &mut self,
        path: impl Into<String>,
        severity: ValidationSeverity,
        message: impl Into<String>,
        repair: Option<&str>,
    ) {
        self.issues.push(ValidationIssue {
            path: path.into(),
            severity,
            message: message.into(),
            repair: repair.map(str::to_string),
        });
    }
}

impl VisionIndex {
    /// Fast validation: header schema and empty-path checks.
    ///
    /// # Errors
    ///
    /// Returns header validation errors.
    pub fn validate_fast(&self) -> Result<(), MemoryError> {
        self.header.validate()
    }

    /// Full structural validation with object paths and severities.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn validate_full(&self) -> ValidationReport {
        let mut report = ValidationReport::default();
        if let Err(error) = self.header.validate() {
            report.push(
                "header",
                ValidationSeverity::Error,
                format!("header invalid: {error:?}"),
                Some("fix schema version and required paths"),
            );
        }

        let source_ids: Vec<u32> = self.header.sources.iter().map(|s| s.source_id).collect();
        let mut seen_sources = Vec::new();
        for (i, source) in self.header.sources.iter().enumerate() {
            if seen_sources.contains(&source.source_id) {
                report.push(
                    format!("header.sources[{i}]"),
                    ValidationSeverity::Error,
                    format!("duplicate source_id {}", source.source_id),
                    Some("dedupe or reassign source ids"),
                );
            }
            seen_sources.push(source.source_id);
            if source.uri.is_empty() {
                report.push(
                    format!("header.sources[{i}].uri"),
                    ValidationSeverity::Warning,
                    "empty source uri",
                    Some("set a non-empty media uri"),
                );
            }
        }

        let subject_ids: Vec<u64> = self.subjects.iter().map(|s| s.subject_id.0).collect();
        let mut seen_subjects = Vec::new();
        for (i, subject) in self.subjects.iter().enumerate() {
            if seen_subjects.contains(&subject.subject_id.0) {
                report.push(
                    format!("subjects[{i}].subject_id"),
                    ValidationSeverity::Error,
                    format!("duplicate subject_id {}", subject.subject_id.0),
                    Some("merge or reassign subject ids"),
                );
            }
            seen_subjects.push(subject.subject_id.0);
        }

        let mask_handles: Vec<u64> = self.masks.entries().iter().map(|(h, _)| h.0).collect();
        let mut prev_pts: Option<(u32, u64)> = None;
        for (i, sample) in self.tracks.samples().iter().enumerate() {
            let path = format!("tracks[{i}]");
            if !source_ids.contains(&sample.source_id.0) {
                report.push(
                    format!("{path}.source_id"),
                    ValidationSeverity::Error,
                    format!("unknown source_id {}", sample.source_id.0),
                    Some("register source on header or drop sample"),
                );
            }
            if let Some(subject_id) = sample.subject_id
                && !subject_ids.is_empty()
                && !subject_ids.contains(&subject_id.0)
            {
                report.push(
                    format!("{path}.subject_id"),
                    ValidationSeverity::Warning,
                    format!("subject_id {} not in subjects table", subject_id.0),
                    Some("insert SubjectProfile or clear subject_id"),
                );
            }
            for (name, value) in [
                ("left", sample.left),
                ("top", sample.top),
                ("right", sample.right),
                ("bottom", sample.bottom),
                ("confidence", sample.confidence),
            ] {
                if !value.is_finite() {
                    report.push(
                        format!("{path}.{name}"),
                        ValidationSeverity::Error,
                        "non-finite coordinate or confidence",
                        Some("drop sample or clamp to finite values"),
                    );
                }
            }
            if sample.left > sample.right || sample.top > sample.bottom {
                report.push(
                    format!("{path}.bbox"),
                    ValidationSeverity::Error,
                    "degenerate bounding box",
                    Some("swap edges or drop sample"),
                );
            }
            if sample.mask_ref != 0 && !mask_handles.contains(&sample.mask_ref) {
                report.push(
                    format!("{path}.mask_ref"),
                    ValidationSeverity::Error,
                    format!("mask_ref {} missing from mask store", sample.mask_ref),
                    Some("clear mask_ref or restore mask blob"),
                );
            }
            let key = (sample.source_id.0, sample.frame_index);
            if let Some(prev) = prev_pts
                && prev.0 == key.0
                && key.1 < prev.1
            {
                report.push(
                    format!("{path}.frame_index"),
                    ValidationSeverity::Warning,
                    "frame_index not monotonic within source",
                    Some("sort track stream by (source_id, frame_index)"),
                );
            }
            prev_pts = Some(key);
        }

        let mut seen_event_ids = Vec::new();
        let mut prev_event_ns: Option<i64> = None;
        for (i, event) in self.events.iter().enumerate() {
            let path = format!("events[{i}]");
            if seen_event_ids.contains(&event.event_id.0) {
                report.push(
                    format!("{path}.event_id"),
                    ValidationSeverity::Error,
                    format!("duplicate event_id {}", event.event_id.0),
                    Some("renumber event ids monotonically"),
                );
            }
            seen_event_ids.push(event.event_id.0);
            if !source_ids.contains(&event.stamp.source_id.0) {
                report.push(
                    format!("{path}.stamp.source_id"),
                    ValidationSeverity::Error,
                    format!("unknown source_id {}", event.stamp.source_id.0),
                    Some("register source or drop event"),
                );
            }
            if let Some(subject_id) = event.subject_id
                && !subject_ids.is_empty()
                && !subject_ids.contains(&subject_id.0)
            {
                report.push(
                    format!("{path}.subject_id"),
                    ValidationSeverity::Warning,
                    format!("subject_id {} not in subjects table", subject_id.0),
                    Some("insert SubjectProfile or clear subject_id"),
                );
            }
            let ns = event.stamp.pts.as_nanos();
            if let Some(prev) = prev_event_ns
                && ns < prev
            {
                report.push(
                    format!("{path}.stamp.pts"),
                    ValidationSeverity::Info,
                    "event presentation time not globally monotonic",
                    None,
                );
            }
            prev_event_ns = Some(ns);
        }

        for (i, appearance) in self.appearances.iter().enumerate() {
            if let Some(subject_id) = appearance.subject_id
                && !subject_ids.is_empty()
                && !subject_ids.contains(&subject_id.0)
            {
                report.push(
                    format!("appearances[{i}].subject_id"),
                    ValidationSeverity::Warning,
                    format!("subject_id {} not in subjects table", subject_id.0),
                    Some("insert SubjectProfile or clear subject_id"),
                );
            }
            if !source_ids.contains(&appearance.source_id.0) {
                report.push(
                    format!("appearances[{i}].source_id"),
                    ValidationSeverity::Error,
                    format!("unknown source_id {}", appearance.source_id.0),
                    Some("register source or drop appearance"),
                );
            }
        }

        for (i, pattern) in self.patterns.iter().enumerate() {
            for (j, event_id) in pattern.evidence_events.iter().enumerate() {
                if !self.events.iter().any(|e| e.event_id == *event_id) {
                    report.push(
                        format!("patterns[{i}].evidence_events[{j}]"),
                        ValidationSeverity::Warning,
                        format!("evidence event_id {} missing", event_id.0),
                        Some("drop dangling evidence ref"),
                    );
                }
            }
        }

        for (i, anomaly) in self.anomalies.iter().enumerate() {
            if !anomaly.score.is_finite() {
                report.push(
                    format!("anomalies[{i}].score"),
                    ValidationSeverity::Error,
                    "non-finite anomaly score",
                    Some("clamp or drop anomaly"),
                );
            }
            for (j, event_id) in anomaly.evidence.iter().enumerate() {
                if !self.events.iter().any(|e| e.event_id == *event_id) {
                    report.push(
                        format!("anomalies[{i}].evidence[{j}]"),
                        ValidationSeverity::Warning,
                        format!("evidence event_id {} missing", event_id.0),
                        Some("drop dangling evidence ref"),
                    );
                }
            }
        }

        // Orphan observation note: track samples with subject but no subject profile already warned.
        let _ = (SourceId(0), TrackId(0), SubjectId(0), EventId(0));
        report
    }

    /// Produces a repair plan (issue + suggested action) from a full validation.
    #[must_use]
    pub fn repair_plan(&self) -> Vec<ValidationIssue> {
        self.validate_full()
            .issues
            .into_iter()
            .filter(|issue| issue.repair.is_some())
            .collect()
    }

    /// Validates the document (fast path, backward compatible).
    ///
    /// # Errors
    ///
    /// Propagates header validation errors.
    pub fn validate(&self) -> Result<(), MemoryError> {
        self.validate_fast()
    }
}
