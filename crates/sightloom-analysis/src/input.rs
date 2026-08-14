//! Lightweight observation inputs for miners and anomaly detectors.
//!
//! These are intentionally independent of `sightloom-index` to avoid crate cycles.
//! Hosts map `VisionIndex` entities into these views.

extern crate alloc;

use alloc::vec::Vec;
use sightloom_core::{EventId, SourceId, SubjectId, ZoneId};

/// One timed subject observation (appearance start, visit start, etc.).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimedSubjectEvent {
    /// Subject when known.
    pub subject_id: Option<SubjectId>,
    /// Source when known.
    pub source_id: Option<SourceId>,
    /// Event time in nanoseconds (media or wall timeline, host-defined).
    pub at_ns: i64,
    /// Optional supporting event id.
    pub event_id: Option<EventId>,
    /// Host-defined kind tag for `EventBeforeEvent` mining (`0` = untyped).
    pub kind_tag: u32,
}

/// One dwell / visit duration sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DurationSample {
    /// Subject when known.
    pub subject_id: Option<SubjectId>,
    /// Source camera when known (camera-specific baselines).
    pub source_id: Option<SourceId>,
    /// Zone when known.
    pub zone_id: Option<ZoneId>,
    /// Duration nanoseconds.
    pub duration_ns: i64,
    /// Sample time (end of dwell) in nanoseconds.
    pub at_ns: i64,
    /// Optional evidence event.
    pub event_id: Option<EventId>,
}

/// One ordered zone route for a subject.
#[derive(Clone, Debug, PartialEq)]
pub struct RouteSample {
    /// Subject.
    pub subject_id: SubjectId,
    /// Ordered zone ids.
    pub zones: Vec<ZoneId>,
    /// Route end time nanoseconds.
    pub at_ns: i64,
    /// Optional evidence event.
    pub event_id: Option<EventId>,
}

/// One co-occurrence pair observation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairSample {
    /// First subject (order is normalized by miners).
    pub subject_a: SubjectId,
    /// Second subject.
    pub subject_b: SubjectId,
    /// Shared source when known.
    pub source_id: Option<SourceId>,
    /// Observation time nanoseconds.
    pub at_ns: i64,
    /// Optional evidence event.
    pub event_id: Option<EventId>,
}

/// Bundle of host-prepared series used by miners/detectors.
#[derive(Clone, Debug, Default)]
pub struct AnalysisSeries {
    /// Appearance / presence timestamps.
    pub timed: Vec<TimedSubjectEvent>,
    /// Dwell / visit durations.
    pub durations: Vec<DurationSample>,
    /// Route sequences.
    pub routes: Vec<RouteSample>,
    /// Co-occurrence pairs.
    pub pairs: Vec<PairSample>,
}
