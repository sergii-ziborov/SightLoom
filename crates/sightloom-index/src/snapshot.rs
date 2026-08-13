//! JSON-serializable `VisionIndex` snapshots for materialization and sidecar I/O.

use crate::{
    Appearance, CoOccurrence, EvidenceReel, MemoryError, RedactionIntent, RedactionInterval,
    ReelSegment, Route, SourceTransition, SubjectProfile, TrackSample, VisionIndex,
    VisionIndexHeader, Visit, ZoneStay,
};
use sightloom_analysis::{AnomalyEvent, AnomalyReason, PatternKind, PatternRecord, Severity};
use sightloom_core::{Direction, EventEnvelope, EventKind, EventPayload, FrameStamp, MediaTime};

/// Fully serializable materialization of a [`VisionIndex`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct VisionIndexSnapshot {
    /// Document header.
    pub header: VisionIndexHeader,
    /// Track samples.
    pub tracks: Vec<TrackSampleDto>,
    /// Event envelopes.
    pub events: Vec<EventEnvelopeDto>,
    /// Appearances.
    pub appearances: Vec<AppearanceDto>,
    /// Visits.
    pub visits: Vec<VisitDto>,
    /// Routes.
    pub routes: Vec<RouteDto>,
    /// Zone stays.
    pub zone_stays: Vec<ZoneStayDto>,
    /// Co-occurrences.
    pub co_occurrences: Vec<CoOccurrenceDto>,
    /// Source transitions.
    pub source_transitions: Vec<SourceTransitionDto>,
    /// Subject profiles.
    pub subjects: Vec<SubjectProfileDto>,
    /// Redaction provenance intervals.
    #[cfg_attr(feature = "std", serde(default))]
    pub redaction_intervals: Vec<RedactionIntervalDto>,
    /// Stored evidence reels.
    #[cfg_attr(feature = "std", serde(default))]
    pub evidence_reels: Vec<EvidenceReelDto>,
    /// Patterns.
    pub patterns: Vec<PatternRecordDto>,
    /// Anomalies.
    pub anomalies: Vec<AnomalyEventDto>,
}

#[cfg(feature = "std")]
impl VisionIndexSnapshot {
    /// Captures a snapshot from an in-memory index.
    #[must_use]
    pub fn from_index(index: &VisionIndex) -> Self {
        Self {
            header: index.header.clone(),
            tracks: index
                .tracks
                .samples()
                .iter()
                .copied()
                .map(Into::into)
                .collect(),
            events: index.events.iter().copied().map(Into::into).collect(),
            appearances: index.appearances.iter().copied().map(Into::into).collect(),
            visits: index.visits.iter().copied().map(Into::into).collect(),
            routes: index.routes.iter().cloned().map(Into::into).collect(),
            zone_stays: index.zone_stays.iter().copied().map(Into::into).collect(),
            co_occurrences: index
                .co_occurrences
                .iter()
                .copied()
                .map(Into::into)
                .collect(),
            source_transitions: index
                .source_transitions
                .iter()
                .copied()
                .map(Into::into)
                .collect(),
            subjects: index.subjects.iter().cloned().map(Into::into).collect(),
            redaction_intervals: index
                .redaction_intervals
                .iter()
                .map(|r| (*r).into())
                .collect(),
            evidence_reels: index
                .evidence_reels
                .iter()
                .cloned()
                .map(Into::into)
                .collect(),
            patterns: index.patterns.iter().cloned().map(Into::into).collect(),
            anomalies: index.anomalies.iter().cloned().map(Into::into).collect(),
        }
    }

    /// Serializes the snapshot to pretty JSON.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Serde`] on failure.
    pub fn to_json(&self) -> Result<String, MemoryError> {
        serde_json::to_string_pretty(self).map_err(|error| MemoryError::Serde(error.to_string()))
    }

    /// Parses a snapshot from JSON.
    ///
    /// # Errors
    ///
    /// Returns serde errors.
    pub fn from_json(text: &str) -> Result<Self, MemoryError> {
        serde_json::from_str(text).map_err(|error| MemoryError::Serde(error.to_string()))
    }
}

/// Serializable media time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct MediaTimeDto {
    /// Tick count.
    pub ticks: i64,
    /// Timescale.
    pub timescale: u32,
}

impl From<MediaTime> for MediaTimeDto {
    fn from(value: MediaTime) -> Self {
        Self {
            ticks: value.ticks(),
            timescale: value.timescale(),
        }
    }
}

/// Serializable frame stamp.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameStampDto {
    /// Source id.
    pub source_id: u32,
    /// Frame index.
    pub frame_index: u64,
    /// Presentation time.
    pub pts: MediaTimeDto,
    /// Optional wall clock nanoseconds.
    pub wall_clock_ns: Option<i64>,
}

impl From<FrameStamp> for FrameStampDto {
    fn from(value: FrameStamp) -> Self {
        Self {
            source_id: value.source_id.0,
            frame_index: value.frame_index,
            pts: value.pts.into(),
            wall_clock_ns: value.wall_clock_ns,
        }
    }
}

/// Serializable track sample.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct TrackSampleDto {
    /// Sample id.
    #[serde(default)]
    pub sample_id: u64,
    /// Superseded sample id.
    #[serde(default)]
    pub supersedes: Option<u64>,
    /// Revision number.
    #[serde(default)]
    pub revision: u32,
    /// Idempotency key.
    #[serde(default)]
    pub idempotency_key: u64,
    /// Source id.
    pub source_id: u32,
    /// Frame index.
    pub frame_index: u64,
    /// Presentation time.
    pub pts: MediaTimeDto,
    /// Local track id within the source.
    pub track_id: u32,
    /// Global track uid (`None` if unset).
    pub track_uid: Option<u64>,
    /// Optional subject id.
    pub subject_id: Option<u64>,
    /// Optional class id.
    pub class_id: Option<u16>,
    /// Left edge.
    pub left: f32,
    /// Top edge.
    pub top: f32,
    /// Right edge.
    pub right: f32,
    /// Bottom edge.
    pub bottom: f32,
    /// Confidence.
    pub confidence: f32,
    /// Mask handle (`0` = none).
    pub mask_ref: u64,
}

impl From<TrackSample> for TrackSampleDto {
    fn from(value: TrackSample) -> Self {
        Self {
            sample_id: value.sample_id,
            supersedes: value.supersedes,
            revision: value.revision,
            idempotency_key: value.idempotency_key,
            source_id: value.source_id.0,
            frame_index: value.frame_index,
            pts: value.pts.into(),
            track_id: value.track_id.0,
            track_uid: value.track_uid.map(|id| id.0),
            subject_id: value.subject_id.map(|id| id.0),
            class_id: value.class_id.map(|id| id.0),
            left: value.left,
            top: value.top,
            right: value.right,
            bottom: value.bottom,
            confidence: value.confidence,
            mask_ref: value.mask_ref,
        }
    }
}

/// Serializable event envelope.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct EventEnvelopeDto {
    /// Event id.
    pub event_id: u64,
    /// Stamp.
    pub stamp: FrameStampDto,
    /// Kind name.
    pub kind: String,
    /// Optional track.
    pub track_id: Option<u32>,
    /// Optional subject.
    pub subject_id: Option<u64>,
    /// Optional zone.
    pub zone_id: Option<u16>,
    /// Optional evidence.
    pub evidence: Option<u64>,
    /// Payload kind name.
    pub payload_kind: String,
    /// Payload zone when present.
    pub payload_zone_id: Option<u16>,
    /// Payload class when present.
    pub payload_class_id: Option<u16>,
    /// Payload direction when present (`ltr` / `rtl`).
    pub payload_direction: Option<String>,
    /// Payload duration nanoseconds when present.
    pub payload_duration_ns: Option<i64>,
    /// Payload visit count when present.
    pub payload_visit_count: Option<u32>,
    /// Payload occupancy when present.
    pub payload_occupancy: Option<u32>,
    /// Payload score when present.
    pub payload_score: Option<f32>,
    /// Payload aux when present.
    pub payload_aux: Option<f32>,
    /// Payload tag when present.
    pub payload_tag: Option<u32>,
}

impl From<EventEnvelope> for EventEnvelopeDto {
    fn from(value: EventEnvelope) -> Self {
        let mut dto = Self {
            event_id: value.event_id.0,
            stamp: value.stamp.into(),
            kind: event_kind_name(value.kind).into(),
            track_id: value.track_id.map(|id| id.0),
            subject_id: value.subject_id.map(|id| id.0),
            zone_id: value.zone_id.map(|id| id.0),
            evidence: value.evidence.map(|id| id.0),
            payload_kind: "empty".into(),
            payload_zone_id: None,
            payload_class_id: None,
            payload_direction: None,
            payload_duration_ns: None,
            payload_visit_count: None,
            payload_occupancy: None,
            payload_score: None,
            payload_aux: None,
            payload_tag: None,
        };
        fill_payload(&mut dto, value.payload);
        dto
    }
}

fn fill_payload(dto: &mut EventEnvelopeDto, payload: EventPayload) {
    match payload {
        EventPayload::Empty => {}
        EventPayload::Entered { zone_id, class_id } => {
            dto.payload_kind = "entered".into();
            dto.payload_zone_id = Some(zone_id.0);
            dto.payload_class_id = class_id.map(|id| id.0);
        }
        EventPayload::Exited { zone_id, class_id } => {
            dto.payload_kind = "exited".into();
            dto.payload_zone_id = Some(zone_id.0);
            dto.payload_class_id = class_id.map(|id| id.0);
        }
        EventPayload::Crossed { zone_id, direction } => {
            dto.payload_kind = "crossed".into();
            dto.payload_zone_id = Some(zone_id.0);
            dto.payload_direction = Some(match direction {
                Direction::LeftToRight => "ltr".into(),
                Direction::RightToLeft => "rtl".into(),
            });
        }
        EventPayload::DwellStarted { zone_id } => {
            dto.payload_kind = "dwell_started".into();
            dto.payload_zone_id = Some(zone_id.0);
        }
        EventPayload::DwellEnded {
            zone_id,
            duration_ns,
            visit_count,
        } => {
            dto.payload_kind = "dwell_ended".into();
            dto.payload_zone_id = Some(zone_id.0);
            dto.payload_duration_ns = Some(duration_ns);
            dto.payload_visit_count = Some(visit_count);
        }
        EventPayload::Occupancy { zone_id, occupancy } => {
            dto.payload_kind = "occupancy".into();
            dto.payload_zone_id = Some(zone_id.0);
            dto.payload_occupancy = Some(occupancy);
        }
        EventPayload::Metrics { score, aux, tag } => {
            dto.payload_kind = "metrics".into();
            dto.payload_score = Some(score);
            dto.payload_aux = Some(aux);
            dto.payload_tag = Some(tag);
        }
    }
}

fn event_kind_name(kind: EventKind) -> &'static str {
    match kind {
        EventKind::Zone => "zone",
        EventKind::Dwell => "dwell",
        EventKind::Occupancy => "occupancy",
        EventKind::Identity => "identity",
        EventKind::Pattern => "pattern",
        EventKind::Anomaly => "anomaly",
        EventKind::Custom => "custom",
    }
}

/// Serializable appearance.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct AppearanceDto {
    /// Appearance id.
    pub appearance_id: u64,
    /// Subject id.
    pub subject_id: Option<u64>,
    /// Track id.
    pub track_id: Option<u32>,
    /// Source id.
    pub source_id: u32,
    /// Start time.
    pub start: MediaTimeDto,
    /// End time.
    pub end: MediaTimeDto,
    /// Class id.
    pub class_id: Option<u16>,
    /// Peak confidence.
    pub peak_confidence: f32,
    /// Evidence handle.
    pub evidence: Option<u64>,
}

impl From<Appearance> for AppearanceDto {
    fn from(value: Appearance) -> Self {
        Self {
            appearance_id: value.appearance_id.0,
            subject_id: value.subject_id.map(|id| id.0),
            track_id: value.track_id.map(|id| id.0),
            source_id: value.source_id.0,
            start: value.start.into(),
            end: value.end.into(),
            class_id: value.class_id.map(|id| id.0),
            peak_confidence: value.peak_confidence,
            evidence: value.evidence.map(|id| id.0),
        }
    }
}

/// Serializable visit.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct VisitDto {
    /// Visit id.
    pub visit_id: u64,
    /// Subject id.
    pub subject_id: Option<u64>,
    /// Start.
    pub start: MediaTimeDto,
    /// End.
    pub end: MediaTimeDto,
    /// Source count.
    pub source_count: u32,
    /// Duration nanoseconds.
    pub duration_ns: i64,
}

impl From<Visit> for VisitDto {
    fn from(value: Visit) -> Self {
        Self {
            visit_id: value.visit_id.0,
            subject_id: value.subject_id.map(|id| id.0),
            start: value.start.into(),
            end: value.end.into(),
            source_count: value.source_count,
            duration_ns: value.duration_ns,
        }
    }
}

/// Serializable route.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct RouteDto {
    /// Subject id.
    pub subject_id: u64,
    /// Zones.
    pub zones: Vec<u16>,
    /// Sources.
    pub sources: Vec<u32>,
    /// Start.
    pub start: MediaTimeDto,
    /// End.
    pub end: MediaTimeDto,
}

impl From<Route> for RouteDto {
    fn from(value: Route) -> Self {
        Self {
            subject_id: value.subject_id.0,
            zones: value.zones.iter().map(|zone| zone.0).collect(),
            sources: value.sources.iter().map(|source| source.0).collect(),
            start: value.start.into(),
            end: value.end.into(),
        }
    }
}

/// Serializable zone stay.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct ZoneStayDto {
    /// Zone id.
    pub zone_id: u16,
    /// Subject id.
    pub subject_id: Option<u64>,
    /// Track id.
    pub track_id: Option<u32>,
    /// Start.
    pub start: MediaTimeDto,
    /// End.
    pub end: MediaTimeDto,
    /// Duration nanoseconds.
    pub duration_ns: i64,
}

impl From<ZoneStay> for ZoneStayDto {
    fn from(value: ZoneStay) -> Self {
        Self {
            zone_id: value.zone_id.0,
            subject_id: value.subject_id.map(|id| id.0),
            track_id: value.track_id.map(|id| id.0),
            start: value.start.into(),
            end: value.end.into(),
            duration_ns: value.duration_ns,
        }
    }
}

/// Serializable co-occurrence.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct CoOccurrenceDto {
    /// Subject A.
    pub subject_a: u64,
    /// Subject B.
    pub subject_b: u64,
    /// Source id.
    pub source_id: Option<u32>,
    /// Start.
    pub start: MediaTimeDto,
    /// End.
    pub end: MediaTimeDto,
    /// Overlap nanoseconds.
    pub overlap_ns: i64,
}

impl From<CoOccurrence> for CoOccurrenceDto {
    fn from(value: CoOccurrence) -> Self {
        Self {
            subject_a: value.subject_a.0,
            subject_b: value.subject_b.0,
            source_id: value.source_id.map(|id| id.0),
            start: value.start.into(),
            end: value.end.into(),
            overlap_ns: value.overlap_ns,
        }
    }
}

/// Serializable source transition.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct SourceTransitionDto {
    /// Subject id.
    pub subject_id: u64,
    /// From source.
    pub from_source: u32,
    /// To source.
    pub to_source: u32,
    /// Transition time.
    pub at: MediaTimeDto,
    /// Evidence.
    pub evidence: Option<u64>,
}

impl From<SourceTransition> for SourceTransitionDto {
    fn from(value: SourceTransition) -> Self {
        Self {
            subject_id: value.subject_id.0,
            from_source: value.from_source.0,
            to_source: value.to_source.0,
            at: value.at.into(),
            evidence: value.evidence.map(|id| id.0),
        }
    }
}

/// Serializable subject profile.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct SubjectProfileDto {
    /// Subject id.
    pub subject_id: u64,
    /// Label.
    pub label: Option<String>,
    /// Appearance count.
    pub appearance_count: u32,
    /// Source count.
    pub source_count: u32,
    /// Total duration nanoseconds.
    pub total_duration_ns: i64,
    /// First seen.
    pub first_seen: Option<MediaTimeDto>,
    /// Last seen.
    pub last_seen: Option<MediaTimeDto>,
    /// Embedding handle.
    pub embedding: Option<u64>,
}

impl From<SubjectProfile> for SubjectProfileDto {
    fn from(value: SubjectProfile) -> Self {
        Self {
            subject_id: value.subject_id.0,
            label: value.label,
            appearance_count: value.appearance_count,
            source_count: value.source_count,
            total_duration_ns: value.total_duration_ns,
            first_seen: value.first_seen.map(Into::into),
            last_seen: value.last_seen.map(Into::into),
            embedding: value.embedding.map(|id| id.0),
        }
    }
}

/// Serializable redaction / provenance interval.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct RedactionIntervalDto {
    /// Interval id.
    pub interval_id: u64,
    /// Subject id.
    pub subject_id: Option<u64>,
    /// Source id.
    pub source_id: u32,
    /// Track id.
    pub track_id: Option<u32>,
    /// Start.
    pub start: MediaTimeDto,
    /// End.
    pub end: MediaTimeDto,
    /// Intent wire name (`blur_subject`, `blur_others`, `uncertain_hold`, `custom`).
    pub intent: String,
    /// Evidence handle.
    pub evidence: Option<u64>,
    /// Mask handle.
    pub mask_ref: u64,
    /// Peak confidence.
    pub peak_confidence: f32,
    /// Appearance id.
    pub appearance_id: Option<u64>,
    /// Host tag.
    pub tag: u32,
}

impl From<RedactionInterval> for RedactionIntervalDto {
    fn from(value: RedactionInterval) -> Self {
        Self {
            interval_id: value.interval_id.0,
            subject_id: value.subject_id.map(|id| id.0),
            source_id: value.source_id.0,
            track_id: value.track_id.map(|id| id.0),
            start: value.start.into(),
            end: value.end.into(),
            intent: value.intent.as_str().into(),
            evidence: value.evidence.map(|id| id.0),
            mask_ref: value.mask_ref,
            peak_confidence: value.peak_confidence,
            appearance_id: value.appearance_id.map(|id| id.0),
            tag: value.tag,
        }
    }
}

/// Parses intent wire name (unknown → Custom).
#[must_use]
pub fn redaction_intent_from_dto(name: &str) -> RedactionIntent {
    RedactionIntent::from_str_name(name).unwrap_or(RedactionIntent::Custom)
}

/// Serializable reel segment.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct ReelSegmentDto {
    /// Source id.
    pub source_id: u32,
    /// Track id.
    pub track_id: Option<u32>,
    /// Track uid.
    pub track_uid: Option<u64>,
    /// Start.
    pub start: MediaTimeDto,
    /// End.
    pub end: MediaTimeDto,
    /// Mask handle.
    pub mask_ref: u64,
    /// Evidence handle.
    pub evidence: Option<u64>,
    /// Peak confidence.
    pub peak_confidence: f32,
    /// Sample id.
    pub sample_id: Option<u64>,
}

impl From<ReelSegment> for ReelSegmentDto {
    fn from(value: ReelSegment) -> Self {
        Self {
            source_id: value.source_id.0,
            track_id: value.track_id.map(|id| id.0),
            track_uid: value.track_uid.map(|id| id.0),
            start: value.start.into(),
            end: value.end.into(),
            mask_ref: value.mask_ref,
            evidence: value.evidence.map(|id| id.0),
            peak_confidence: value.peak_confidence,
            sample_id: value.sample_id,
        }
    }
}

/// Serializable evidence reel.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct EvidenceReelDto {
    /// Reel id.
    pub reel_id: u64,
    /// Subject id.
    pub subject_id: Option<u64>,
    /// Segments.
    pub segments: Vec<ReelSegmentDto>,
    /// Host tag.
    pub tag: u32,
}

impl From<EvidenceReel> for EvidenceReelDto {
    fn from(value: EvidenceReel) -> Self {
        Self {
            reel_id: value.reel_id.0,
            subject_id: value.subject_id.map(|id| id.0),
            segments: value.segments.into_iter().map(Into::into).collect(),
            tag: value.tag,
        }
    }
}

/// Serializable pattern record.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct PatternRecordDto {
    /// Pattern id.
    pub pattern_id: u64,
    /// Kind name.
    pub kind: String,
    /// Subject id.
    pub subject_id: Option<u64>,
    /// Confidence.
    pub confidence: f32,
    /// Evidence event ids.
    pub evidence_events: Vec<u64>,
    /// Tag.
    pub tag: u32,
}

impl From<PatternRecord> for PatternRecordDto {
    fn from(value: PatternRecord) -> Self {
        Self {
            pattern_id: value.pattern_id.0,
            kind: pattern_kind_name(value.kind).into(),
            subject_id: value.subject_id.map(|id| id.0),
            confidence: value.confidence,
            evidence_events: value.evidence_events.iter().map(|id| id.0).collect(),
            tag: value.tag,
        }
    }
}

fn pattern_kind_name(kind: PatternKind) -> &'static str {
    match kind {
        PatternKind::TimeOfDay => "time_of_day",
        PatternKind::DayOfWeek => "day_of_week",
        PatternKind::VisitPeriodicity => "visit_periodicity",
        PatternKind::DwellDistribution => "dwell_distribution",
        PatternKind::RouteSequence => "route_sequence",
        PatternKind::CoOccurrence => "co_occurrence",
        PatternKind::EventBeforeEvent => "event_before_event",
        PatternKind::ExpectedAbsence => "expected_absence",
        PatternKind::GroupFormation => "group_formation",
        PatternKind::Custom => "custom",
    }
}

/// Serializable anomaly event.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct AnomalyEventDto {
    /// Anomaly id.
    pub anomaly_id: u64,
    /// Score.
    pub score: f32,
    /// Severity name.
    pub severity: String,
    /// Reason codes.
    pub reasons: Vec<String>,
    /// Evidence event ids.
    pub evidence: Vec<u64>,
    /// Subject id.
    pub subject_id: Option<u64>,
    /// Source id.
    pub source_id: Option<u32>,
    /// Event time.
    pub at: MediaTimeDto,
}

impl From<AnomalyEvent> for AnomalyEventDto {
    fn from(value: AnomalyEvent) -> Self {
        Self {
            anomaly_id: value.anomaly_id.0,
            score: value.score,
            severity: severity_name(value.severity).into(),
            reasons: value
                .reasons
                .iter()
                .map(|reason| reason_name(*reason))
                .collect(),
            evidence: value.evidence.iter().map(|id| id.0).collect(),
            subject_id: value.subject_id.map(|id| id.0),
            source_id: value.source_id.map(|id| id.0),
            at: value.at.into(),
        }
    }
}

fn severity_name(severity: Severity) -> &'static str {
    match severity {
        Severity::Low => "low",
        Severity::Medium => "medium",
        Severity::High => "high",
        Severity::Critical => "critical",
    }
}

fn reason_name(reason: AnomalyReason) -> String {
    match reason {
        AnomalyReason::UnusualAppearanceTime => "unusual_appearance_time".into(),
        AnomalyReason::UnusualFrequency => "unusual_frequency".into(),
        AnomalyReason::UnusualDwell => "unusual_dwell".into(),
        AnomalyReason::UnusualRoute => "unusual_route".into(),
        AnomalyReason::UnusualCoOccurrence => "unusual_co_occurrence".into(),
        AnomalyReason::MissingExpectedAppearance => "missing_expected_appearance".into(),
        AnomalyReason::SuddenBehaviourChange => "sudden_behaviour_change".into(),
        AnomalyReason::Custom(code) => format!("custom:{code}"),
    }
}
