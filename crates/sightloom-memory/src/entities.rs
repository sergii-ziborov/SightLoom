//! First-class `VisionIndex` entities (M0 contracts).
//!
//! These records are owned by `SightLoom` and are distinct from host
//! `CaptureProject`, `SemanticEditPlan`, `RenderGraph`, and `ExecutionPlan`
//! documents.

use sightloom_core::{
    AnomalyId, AppearanceId, ClassId, EventId, EvidenceRef, MediaTime, PatternId, SourceId,
    SubjectId, TrackId, VisitId, ZoneId,
};

/// One continuous appearance of a subject on a source timeline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Appearance {
    /// Appearance id.
    pub appearance_id: AppearanceId,
    /// Linked subject when resolved.
    pub subject_id: Option<SubjectId>,
    /// Local track that contributed samples.
    pub track_id: Option<TrackId>,
    /// Media source.
    pub source_id: SourceId,
    /// Inclusive start time.
    pub start: MediaTime,
    /// Inclusive end time.
    pub end: MediaTime,
    /// Optional dominant class.
    pub class_id: Option<ClassId>,
    /// Peak confidence observed during the appearance.
    pub peak_confidence: f32,
    /// Optional evidence handle.
    pub evidence: Option<EvidenceRef>,
}

/// A visit of a subject to a scene or logical place across one or more sources.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Visit {
    /// Visit id.
    pub visit_id: VisitId,
    /// Subject under visit (may be unresolved).
    pub subject_id: Option<SubjectId>,
    /// Visit start.
    pub start: MediaTime,
    /// Visit end.
    pub end: MediaTime,
    /// Number of sources involved.
    pub source_count: u32,
    /// Total dwell-like duration in nanoseconds.
    pub duration_ns: i64,
}

/// Ordered route summary for a subject (sequence of zone or source hops).
#[derive(Clone, Debug, PartialEq)]
pub struct Route {
    /// Subject for the route.
    pub subject_id: SubjectId,
    /// Ordered zone ids (empty when only sources are known).
    pub zones: Vec<ZoneId>,
    /// Ordered source ids traversed.
    pub sources: Vec<SourceId>,
    /// Route window start.
    pub start: MediaTime,
    /// Route window end.
    pub end: MediaTime,
}

/// Time spent by a subject or track inside a zone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoneStay {
    /// Zone.
    pub zone_id: ZoneId,
    /// Optional subject.
    pub subject_id: Option<SubjectId>,
    /// Optional track.
    pub track_id: Option<TrackId>,
    /// Stay start.
    pub start: MediaTime,
    /// Stay end.
    pub end: MediaTime,
    /// Duration nanoseconds.
    pub duration_ns: i64,
}

/// Two subjects observed together within a temporal window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoOccurrence {
    /// First subject.
    pub subject_a: SubjectId,
    /// Second subject.
    pub subject_b: SubjectId,
    /// Shared source when known.
    pub source_id: Option<SourceId>,
    /// Window start.
    pub start: MediaTime,
    /// Window end.
    pub end: MediaTime,
    /// Overlap duration nanoseconds.
    pub overlap_ns: i64,
}

/// Transition of a subject between media sources (multi-camera handoff).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceTransition {
    /// Subject that transitioned.
    pub subject_id: SubjectId,
    /// Source left.
    pub from_source: SourceId,
    /// Source entered.
    pub to_source: SourceId,
    /// Transition time.
    pub at: MediaTime,
    /// Optional evidence.
    pub evidence: Option<EvidenceRef>,
}

/// Compact long-lived subject profile stored in the index.
#[derive(Clone, Debug, PartialEq)]
pub struct SubjectProfile {
    /// Subject id.
    pub subject_id: SubjectId,
    /// Optional display label (host-supplied).
    pub label: Option<String>,
    /// Appearance count.
    pub appearance_count: u32,
    /// Distinct source count.
    pub source_count: u32,
    /// Total observed duration nanoseconds.
    pub total_duration_ns: i64,
    /// First seen.
    pub first_seen: Option<MediaTime>,
    /// Last seen.
    pub last_seen: Option<MediaTime>,
    /// Representative embedding handle when available.
    pub embedding: Option<sightloom_core::EmbeddingRef>,
}

/// Pattern kind tags for M0 schema (analysis fills these later).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PatternKind {
    /// Time-of-day habit.
    TimeOfDay,
    /// Day-of-week habit.
    DayOfWeek,
    /// Visit periodicity.
    VisitPeriodicity,
    /// Dwell distribution.
    DwellDistribution,
    /// Route sequence.
    RouteSequence,
    /// Co-occurrence habit.
    CoOccurrence,
    /// Event-before-event chain.
    EventBeforeEvent,
    /// Expected absence window.
    ExpectedAbsence,
    /// Group formation.
    GroupFormation,
    /// Application-defined.
    Custom,
}

/// Stored pattern record.
#[derive(Clone, Debug, PartialEq)]
pub struct PatternRecord {
    /// Pattern id.
    pub pattern_id: PatternId,
    /// Kind.
    pub kind: PatternKind,
    /// Optional subject scope.
    pub subject_id: Option<SubjectId>,
    /// Confidence in `0.0..=1.0` (not clamped).
    pub confidence: f32,
    /// Supporting event ids.
    pub evidence_events: Vec<EventId>,
    /// Free-form host tag.
    pub tag: u32,
}

/// Anomaly severity for host presentation.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Severity {
    /// Informational.
    Low,
    /// Notable deviation.
    Medium,
    /// Strong deviation.
    High,
    /// Critical / safety-relevant.
    Critical,
}

/// Machine-readable anomaly reason codes (host maps to UI copy).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnomalyReason {
    /// Unusual appearance time.
    UnusualAppearanceTime,
    /// Unusual frequency.
    UnusualFrequency,
    /// Unusual dwell.
    UnusualDwell,
    /// Unusual route.
    UnusualRoute,
    /// Unusual co-occurrence.
    UnusualCoOccurrence,
    /// Missing expected appearance.
    MissingExpectedAppearance,
    /// Sudden behaviour change.
    SuddenBehaviourChange,
    /// Application-defined reason code.
    Custom(u32),
}

/// Backend-neutral anomaly event exposed to host products.
///
/// Hosts must not depend on whether the score came from a statistical rule,
/// classical model, or optional quantum backend.
#[derive(Clone, Debug, PartialEq)]
pub struct AnomalyEvent {
    /// Anomaly id.
    pub anomaly_id: AnomalyId,
    /// Score (higher means more anomalous; scale is backend-defined).
    pub score: f32,
    /// Presentation severity.
    pub severity: Severity,
    /// Reason codes.
    pub reasons: Vec<AnomalyReason>,
    /// Supporting event ids from the same index.
    pub evidence: Vec<EventId>,
    /// Optional subject scope.
    pub subject_id: Option<SubjectId>,
    /// Optional source scope.
    pub source_id: Option<SourceId>,
    /// Event time.
    pub at: MediaTime,
}
