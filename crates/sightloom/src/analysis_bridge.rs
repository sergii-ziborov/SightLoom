//! Map a live [`VisionIndex`] into analysis series and run miners/detectors.
//!
//! Keeps `sightloom-analysis` free of index crate cycles: the facade owns the
//! mapping and writes results back into the index document.

use sightloom_analysis::{
    AnalysisSeries, BaselineStats, DurationSample, PairSample, PatternRecord, RouteSample,
    StatAnomalyConfig, TimedSubjectEvent, build_baseline, detect_statistical, mine_patterns,
};
use sightloom_analysis::{AnomalyEvent, AnomalyReason};
use sightloom_core::{SourceId, SubjectId};
use sightloom_index::VisionIndex;

/// Builds an [`AnalysisSeries`] from index tables (effective track samples preferred).
#[must_use]
pub fn analysis_series_from_index(index: &VisionIndex) -> AnalysisSeries {
    let mut series = AnalysisSeries::default();

    for sample in index.tracks.effective_samples() {
        if sample.subject_id.is_none() {
            continue;
        }
        series.timed.push(TimedSubjectEvent {
            subject_id: sample.subject_id,
            source_id: Some(sample.source_id),
            at_ns: sample.pts.as_nanos(),
            event_id: None,
        });
    }

    for stay in &index.zone_stays {
        series.durations.push(DurationSample {
            subject_id: stay.subject_id,
            zone_id: Some(stay.zone_id),
            duration_ns: stay.duration_ns,
            at_ns: stay.end.as_nanos(),
            event_id: None,
        });
    }

    for route in &index.routes {
        series.routes.push(RouteSample {
            subject_id: route.subject_id,
            zones: route.zones.clone(),
            at_ns: route.end.as_nanos(),
            event_id: None,
        });
    }

    for co in &index.co_occurrences {
        series.pairs.push(PairSample {
            subject_a: co.subject_a,
            subject_b: co.subject_b,
            source_id: co.source_id,
            at_ns: co.end.as_nanos(),
            event_id: None,
        });
    }

    // Zone / identity events as additional timed points.
    for event in &index.events {
        if event.subject_id.is_none() {
            continue;
        }
        series.timed.push(TimedSubjectEvent {
            subject_id: event.subject_id,
            source_id: Some(event.stamp.source_id),
            at_ns: event.stamp.pts.as_nanos(),
            event_id: Some(event.event_id),
        });
    }

    series
}

/// Mines patterns from the index and returns new records (does not mutate index).
#[must_use]
pub fn mine_patterns_from_index(
    index: &VisionIndex,
    next_pattern_id: &mut u64,
) -> Vec<PatternRecord> {
    let series = analysis_series_from_index(index);
    mine_patterns(&series, next_pattern_id)
}

/// Builds a statistical baseline from the index.
#[must_use]
pub fn baseline_from_index(index: &VisionIndex, config: StatAnomalyConfig) -> BaselineStats {
    build_baseline(&analysis_series_from_index(index), config)
}

/// Detects statistical anomalies against a baseline using the live index series.
#[must_use]
pub fn detect_anomalies_from_index(
    index: &VisionIndex,
    baseline: &BaselineStats,
    config: StatAnomalyConfig,
    next_anomaly_id: &mut u64,
) -> Vec<AnomalyEvent> {
    detect_statistical(
        &analysis_series_from_index(index),
        baseline,
        config,
        next_anomaly_id,
    )
}

/// Summarizes anomaly reasons for host UIs (English labels).
#[must_use]
pub fn anomaly_reason_label(reason: AnomalyReason) -> &'static str {
    match reason {
        AnomalyReason::UnusualAppearanceTime => "unusual_appearance_time",
        AnomalyReason::UnusualFrequency => "unusual_frequency",
        AnomalyReason::UnusualDwell => "unusual_dwell",
        AnomalyReason::UnusualRoute => "unusual_route",
        AnomalyReason::UnusualCoOccurrence => "unusual_co_occurrence",
        AnomalyReason::MissingExpectedAppearance => "missing_expected_appearance",
        AnomalyReason::SuddenBehaviourChange => "sudden_behaviour_change",
        AnomalyReason::Custom(_) => "custom",
    }
}

/// Demo-oriented identity + span export row (JSON-friendly).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct DemoSpanDto {
    /// Sample id.
    pub sample_id: u64,
    /// Source id.
    pub source_id: u32,
    /// Frame index.
    pub frame_index: u64,
    /// Presentation ticks.
    pub pts_ticks: i64,
    /// Presentation timescale.
    pub pts_timescale: u32,
    /// Local track id.
    pub track_id: u32,
    /// Global track uid when known.
    pub track_uid: Option<u64>,
    /// Subject id when known.
    pub subject_id: Option<u64>,
    /// Box.
    pub left: f32,
    /// Top.
    pub top: f32,
    /// Right.
    pub right: f32,
    /// Bottom.
    pub bottom: f32,
    /// Confidence.
    pub confidence: f32,
    /// Mask handle.
    pub mask_ref: u64,
    /// Revision.
    pub revision: u32,
}

/// Uncertain identity interval DTO for host UI.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct UncertainIntervalDto {
    /// Source.
    pub source_id: u32,
    /// Local track.
    pub track_id: u32,
    /// Subject candidate.
    pub subject_id: Option<u64>,
    /// Start ticks.
    pub start_ticks: i64,
    /// Start timescale.
    pub start_timescale: u32,
    /// End ticks.
    pub end_ticks: i64,
    /// End timescale.
    pub end_timescale: u32,
    /// Peak score.
    pub peak_score: Option<f32>,
}

/// Redaction provenance interval DTO for Intelligence / host audit export.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct RedactionIntervalExportDto {
    /// Interval id.
    pub interval_id: u64,
    /// Subject in focus.
    pub subject_id: Option<u64>,
    /// Source.
    pub source_id: u32,
    /// Local track when known.
    pub track_id: Option<u32>,
    /// Start ticks.
    pub start_ticks: i64,
    /// Start timescale.
    pub start_timescale: u32,
    /// End ticks.
    pub end_ticks: i64,
    /// End timescale.
    pub end_timescale: u32,
    /// Intent wire name.
    pub intent: String,
    /// Evidence handle.
    pub evidence: Option<u64>,
    /// Mask handle.
    pub mask_ref: u64,
    /// Peak confidence.
    pub peak_confidence: f32,
    /// Linked appearance id.
    pub appearance_id: Option<u64>,
    /// Host tag.
    pub tag: u32,
}

/// Helper used by session seed responses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeedResult {
    /// Source.
    pub source_id: SourceId,
    /// Local track id.
    pub track_id: sightloom_core::TrackId,
    /// Global track uid.
    pub track_uid: sightloom_core::TrackUid,
    /// Assigned subject.
    pub subject_id: SubjectId,
}
