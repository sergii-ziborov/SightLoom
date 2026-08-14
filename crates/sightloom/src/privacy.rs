//! Privacy and retention product surface for long-running sessions.
//!
//! Soft GC with **legal holds** (subjects/sources that must not be purged),
//! **per-source TTL**, global caps, and **forget subject** scrubbing.
//! This is not a legal-compliance suite — hosts still own policy UI and audit.

use sightloom_core::{SourceId, SubjectId};

/// Per-source max age for track samples (relative to that source's newest pts).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SourceTtl {
    /// Media source.
    pub source_id: u32,
    /// Max age in nanoseconds (`0` = unlimited for this entry).
    pub max_age_ns: i64,
}

/// Host retention / privacy policy.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RetentionPolicy {
    /// Max track samples kept (oldest first). `0` = unlimited.
    pub max_track_samples: u64,
    /// Drop track samples older than this many ns behind the **global** newest pts. `0` = off.
    pub max_track_age_ns: i64,
    /// Max identity audit events kept. `0` = unlimited.
    pub max_audit_events: u64,
    /// Max stored observations. `0` = unlimited.
    pub max_observations: u64,
    /// Max appearances kept. `0` = unlimited.
    pub max_appearances: u64,
    /// Max visits kept. `0` = unlimited.
    pub max_visits: u64,
    /// Max redaction intervals kept. `0` = unlimited.
    pub max_redaction_intervals: u64,
    /// Default per-source TTL when no entry in [`Self::source_ttls`]. `0` = off.
    pub default_source_ttl_ns: i64,
    /// Explicit per-source TTLs (override default).
    pub source_ttls: Vec<SourceTtl>,
    /// Subjects under legal hold (never purged / never forgotten).
    pub legal_hold_subjects: Vec<u64>,
    /// Sources under legal hold (samples never purged by TTL/caps).
    pub legal_hold_sources: Vec<u32>,
    /// When forgetting a subject, also drop its gallery reference samples.
    pub forget_clears_embeddings: bool,
    /// When trimming by count, drop unlabeled samples before labeled ones.
    pub drop_unlabeled_first: bool,
}

impl RetentionPolicy {
    /// Empty unlimited policy.
    #[must_use]
    pub fn unlimited() -> Self {
        Self::default()
    }

    /// True when `subject` is under legal hold.
    #[must_use]
    pub fn holds_subject(&self, subject: SubjectId) -> bool {
        self.legal_hold_subjects.contains(&subject.0)
    }

    /// True when `source` is under legal hold.
    #[must_use]
    pub fn holds_source(&self, source: SourceId) -> bool {
        self.legal_hold_sources.contains(&source.0)
    }

    /// TTL for a source (`0` = no TTL).
    #[must_use]
    pub fn ttl_for_source(&self, source: SourceId) -> i64 {
        self.source_ttls
            .iter()
            .find(|t| t.source_id == source.0)
            .map_or(self.default_source_ttl_ns, |t| t.max_age_ns)
    }

    /// Adds or replaces a per-source TTL.
    pub fn set_source_ttl(&mut self, source: SourceId, max_age_ns: i64) {
        if let Some(slot) = self
            .source_ttls
            .iter_mut()
            .find(|t| t.source_id == source.0)
        {
            slot.max_age_ns = max_age_ns;
        } else {
            self.source_ttls.push(SourceTtl {
                source_id: source.0,
                max_age_ns,
            });
        }
    }

    /// Places a subject under legal hold.
    pub fn hold_subject(&mut self, subject: SubjectId) {
        if !self.legal_hold_subjects.contains(&subject.0) {
            self.legal_hold_subjects.push(subject.0);
        }
    }

    /// Removes a subject from legal hold.
    pub fn release_subject(&mut self, subject: SubjectId) {
        self.legal_hold_subjects.retain(|id| *id != subject.0);
    }

    /// Places a source under legal hold.
    pub fn hold_source(&mut self, source: SourceId) {
        if !self.legal_hold_sources.contains(&source.0) {
            self.legal_hold_sources.push(source.0);
        }
    }

    /// Removes a source from legal hold.
    pub fn release_source(&mut self, source: SourceId) {
        self.legal_hold_sources.retain(|id| *id != source.0);
    }
}

/// Outcome of one retention / forget operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RetentionReport {
    /// Track samples dropped.
    pub dropped_tracks: usize,
    /// Observations dropped.
    pub dropped_observations: usize,
    /// Audit events dropped.
    pub dropped_audit: usize,
    /// Appearances dropped.
    pub dropped_appearances: usize,
    /// Visits dropped.
    pub dropped_visits: usize,
    /// Redaction intervals dropped.
    pub dropped_redactions: usize,
    /// Samples kept only because of legal hold.
    pub protected_by_hold: usize,
    /// Subjects successfully forgotten (0 or 1 for forget ops).
    pub forgotten_subjects: usize,
}

impl RetentionReport {
    /// Legacy triple for older callers.
    #[must_use]
    pub const fn as_legacy_triple(self) -> (usize, usize, usize) {
        (
            self.dropped_tracks,
            self.dropped_observations,
            self.dropped_audit,
        )
    }
}
