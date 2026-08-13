//! First-class `VisionIndex` entities (memory records).
//!
//! Pattern and anomaly types live in `sightloom-analysis` and are stored by
//! reference from the index document.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use sightloom_core::{
    AppearanceId, ClassId, EvidenceRef, MediaTime, RedactionIntervalId, SourceId, SubjectId,
    TrackId, VisitId, ZoneId,
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

/// Host / Intelligence intent for a redaction provenance interval.
///
/// `SightLoom` stores the interval + evidence handles only; pixel blur is out of
/// scope (sibling render product).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum RedactionIntent {
    /// Blur the referenced subject on this interval.
    BlurSubject = 0,
    /// Blur everyone except the referenced subject on this interval.
    BlurOthers = 1,
    /// Hold for review (uncertain identity).
    UncertainHold = 2,
    /// Host-defined / free-form intent (`tag` carries meaning).
    Custom = 3,
}

impl RedactionIntent {
    /// Stable wire name for JSON export.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BlurSubject => "blur_subject",
            Self::BlurOthers => "blur_others",
            Self::UncertainHold => "uncertain_hold",
            Self::Custom => "custom",
        }
    }

    /// Parses a wire name (case-sensitive).
    #[must_use]
    pub fn from_str_name(name: &str) -> Option<Self> {
        match name {
            "blur_subject" => Some(Self::BlurSubject),
            "blur_others" => Some(Self::BlurOthers),
            "uncertain_hold" => Some(Self::UncertainHold),
            "custom" => Some(Self::Custom),
            _ => None,
        }
    }
}

/// First-class provenance row for a redaction-relevant media interval.
///
/// Links subject / track / time / evidence so hosts and Intelligence can audit
/// what would be redacted without storing pixels here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RedactionInterval {
    /// Interval id.
    pub interval_id: RedactionIntervalId,
    /// Subject in focus (blur target, keep-subject, or uncertain candidate).
    pub subject_id: Option<SubjectId>,
    /// Media source.
    pub source_id: SourceId,
    /// Local track when known.
    pub track_id: Option<TrackId>,
    /// Inclusive start.
    pub start: MediaTime,
    /// Inclusive end.
    pub end: MediaTime,
    /// Redaction intent.
    pub intent: RedactionIntent,
    /// Optional host evidence handle (crop / mask blob / reel segment).
    pub evidence: Option<EvidenceRef>,
    /// Optional mask handle (`0` = none).
    pub mask_ref: u64,
    /// Peak confidence / score on the interval.
    pub peak_confidence: f32,
    /// Linked appearance when derived from memory entities.
    pub appearance_id: Option<AppearanceId>,
    /// Free-form host tag.
    pub tag: u32,
}
