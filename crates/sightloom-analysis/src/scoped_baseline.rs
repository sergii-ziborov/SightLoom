//! Subject- and camera-scoped statistical baselines.
//!
//! Global baseline remains the fallback when a scope has too few samples.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

extern crate alloc;

use crate::input::AnalysisSeries;
use crate::stat_anomaly::{BaselineStats, StatAnomalyConfig, build_baseline, detect_statistical};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use sightloom_core::{SourceId, SubjectId};

/// Which scope a baseline row belongs to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BaselineScope {
    /// Pooled over all samples.
    Global,
    /// One subject.
    Subject(SubjectId),
    /// One source / camera.
    Source(SourceId),
}

/// Store of global + per-subject + per-source baselines.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScopedBaselineStore {
    /// Global fallback.
    pub global: BaselineStats,
    /// Subject id → baseline.
    pub by_subject: BTreeMap<u64, BaselineStats>,
    /// Source id → baseline.
    pub by_source: BTreeMap<u32, BaselineStats>,
}

impl ScopedBaselineStore {
    /// Builds scoped baselines from a history series.
    #[must_use]
    pub fn from_series(series: &AnalysisSeries, config: StatAnomalyConfig) -> Self {
        let global = build_baseline(series, config);
        let mut by_subject: BTreeMap<u64, BaselineStats> = BTreeMap::new();
        let mut by_source: BTreeMap<u32, BaselineStats> = BTreeMap::new();

        // Collect subject ids and source ids present.
        let mut subjects: Vec<SubjectId> = Vec::new();
        let mut sources: Vec<SourceId> = Vec::new();
        for e in &series.timed {
            if let Some(s) = e.subject_id
                && !subjects.contains(&s)
            {
                subjects.push(s);
            }
            if let Some(src) = e.source_id
                && !sources.contains(&src)
            {
                sources.push(src);
            }
        }
        for d in &series.durations {
            if let Some(s) = d.subject_id
                && !subjects.contains(&s)
            {
                subjects.push(s);
            }
            if let Some(src) = d.source_id
                && !sources.contains(&src)
            {
                sources.push(src);
            }
        }

        for s in subjects {
            let sub = filter_series_subject(series, s);
            let b = build_baseline(&sub, config);
            if b.dwell_n >= config.min_samples
                || b.gap_n >= config.min_samples
                || b.timed_n >= config.min_samples
            {
                by_subject.insert(s.0, b);
            }
        }
        for src in sources {
            let sub = filter_series_source(series, src);
            let b = build_baseline(&sub, config);
            if b.dwell_n >= config.min_samples
                || b.gap_n >= config.min_samples
                || b.timed_n >= config.min_samples
            {
                by_source.insert(src.0, b);
            }
        }

        Self {
            global,
            by_subject,
            by_source,
        }
    }

    /// Resolves the best baseline for a sample (subject > source > global).
    #[must_use]
    pub fn resolve(
        &self,
        subject_id: Option<SubjectId>,
        source_id: Option<SourceId>,
    ) -> &BaselineStats {
        if let Some(s) = subject_id
            && let Some(b) = self.by_subject.get(&s.0)
        {
            return b;
        }
        if let Some(src) = source_id
            && let Some(b) = self.by_source.get(&src.0)
        {
            return b;
        }
        &self.global
    }

    /// Number of subject-scoped baselines.
    #[must_use]
    pub fn subject_count(&self) -> usize {
        self.by_subject.len()
    }

    /// Number of source-scoped baselines.
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.by_source.len()
    }
}

/// Detects anomalies using the most specific baseline available per sample.
///
/// Implementation: run global detect, then re-run detect on each subject- and
/// source-filtered live slice and merge (dedupe by reason+time+subject is host-side).
#[must_use]
pub fn detect_statistical_scoped(
    live: &AnalysisSeries,
    store: &ScopedBaselineStore,
    config: StatAnomalyConfig,
    next_id: &mut u64,
) -> Vec<crate::anomaly::AnomalyEvent> {
    let mut out = detect_statistical(live, &store.global, config, next_id);

    // Subject slices
    let mut subjects: Vec<SubjectId> = Vec::new();
    for e in &live.timed {
        if let Some(s) = e.subject_id
            && !subjects.contains(&s)
        {
            subjects.push(s);
        }
    }
    for d in &live.durations {
        if let Some(s) = d.subject_id
            && !subjects.contains(&s)
        {
            subjects.push(s);
        }
    }
    for s in subjects {
        if let Some(b) = store.by_subject.get(&s.0) {
            let slice = filter_series_subject(live, s);
            out.extend(detect_statistical(&slice, b, config, next_id));
        }
    }

    // Source slices
    let mut sources: Vec<SourceId> = Vec::new();
    for e in &live.timed {
        if let Some(src) = e.source_id
            && !sources.contains(&src)
        {
            sources.push(src);
        }
    }
    for d in &live.durations {
        if let Some(src) = d.source_id
            && !sources.contains(&src)
        {
            sources.push(src);
        }
    }
    for src in sources {
        if let Some(b) = store.by_source.get(&src.0) {
            let slice = filter_series_source(live, src);
            out.extend(detect_statistical(&slice, b, config, next_id));
        }
    }
    out
}

fn filter_series_subject(series: &AnalysisSeries, subject: SubjectId) -> AnalysisSeries {
    AnalysisSeries {
        timed: series
            .timed
            .iter()
            .copied()
            .filter(|e| e.subject_id == Some(subject))
            .collect(),
        durations: series
            .durations
            .iter()
            .copied()
            .filter(|d| d.subject_id == Some(subject))
            .collect(),
        routes: series
            .routes
            .iter()
            .filter(|r| r.subject_id == subject)
            .cloned()
            .collect(),
        pairs: series
            .pairs
            .iter()
            .copied()
            .filter(|p| p.subject_a == subject || p.subject_b == subject)
            .collect(),
    }
}

fn filter_series_source(series: &AnalysisSeries, source: SourceId) -> AnalysisSeries {
    AnalysisSeries {
        timed: series
            .timed
            .iter()
            .copied()
            .filter(|e| e.source_id == Some(source))
            .collect(),
        durations: series
            .durations
            .iter()
            .copied()
            .filter(|d| d.source_id == Some(source))
            .collect(),
        routes: series.routes.clone(),
        pairs: series
            .pairs
            .iter()
            .copied()
            .filter(|p| p.source_id == Some(source))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{DurationSample, TimedSubjectEvent};

    #[test]
    fn builds_subject_and_source_scopes() {
        let mut series = AnalysisSeries::default();
        for i in 0..10 {
            series.timed.push(TimedSubjectEvent {
                subject_id: Some(SubjectId(1)),
                source_id: Some(SourceId(7)),
                at_ns: i * 3_600_000_000_000,
                event_id: None,
                kind_tag: 0,
            });
            series.durations.push(DurationSample {
                subject_id: Some(SubjectId(1)),
                source_id: Some(SourceId(7)),
                zone_id: None,
                duration_ns: 1_000_000_000 + i * 1_000,
                at_ns: i * 3_600_000_000_000,
                event_id: None,
            });
        }
        for i in 0..10 {
            series.timed.push(TimedSubjectEvent {
                subject_id: Some(SubjectId(2)),
                source_id: Some(SourceId(8)),
                at_ns: i * 3_600_000_000_000,
                event_id: None,
                kind_tag: 0,
            });
        }
        let store = ScopedBaselineStore::from_series(&series, StatAnomalyConfig::default());
        assert!(store.subject_count() >= 1);
        assert!(store.source_count() >= 1);
        let b = store.resolve(Some(SubjectId(1)), Some(SourceId(7)));
        assert!(b.dwell_n >= 5 || b.timed_n >= 5);
    }
}
