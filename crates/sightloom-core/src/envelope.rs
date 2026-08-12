//! Portable event envelope shared by index serialization and analytics.

use crate::{ClassId, Direction, EventId, EvidenceRef, FrameStamp, SubjectId, TrackId, ZoneId};

/// Coarse kind tag for an indexed event.
///
/// Payload-specific details live in [`EventPayload`]. Hosts match on `kind`
/// for cheap filtering without decoding full payloads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventKind {
    /// Zone membership or line crossing.
    Zone,
    /// Dwell timing within a zone.
    Dwell,
    /// Occupancy change inside a zone.
    Occupancy,
    /// Identity / subject resolution lifecycle.
    Identity,
    /// Pattern detector output.
    Pattern,
    /// Anomaly detector output.
    Anomaly,
    /// Application-defined custom event.
    Custom,
}

/// Compact payload attached to an [`EventEnvelope`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EventPayload {
    /// No additional fields.
    Empty,
    /// Zone enter.
    Entered {
        /// Zone that was entered.
        zone_id: ZoneId,
        /// Optional class at the time of the event.
        class_id: Option<ClassId>,
    },
    /// Zone exit.
    Exited {
        /// Zone that was exited.
        zone_id: ZoneId,
        /// Optional class at the time of the event.
        class_id: Option<ClassId>,
    },
    /// Line crossing.
    Crossed {
        /// Line zone that was crossed.
        zone_id: ZoneId,
        /// Crossing direction.
        direction: Direction,
    },
    /// Dwell started.
    DwellStarted {
        /// Zone under dwell.
        zone_id: ZoneId,
    },
    /// Dwell ended.
    DwellEnded {
        /// Zone under dwell.
        zone_id: ZoneId,
        /// Duration in nanoseconds.
        duration_ns: i64,
        /// Visit count after this dwell.
        visit_count: u32,
    },
    /// Occupancy snapshot.
    Occupancy {
        /// Zone whose occupancy changed.
        zone_id: ZoneId,
        /// Confirmed occupants.
        occupancy: u32,
    },
    /// Free numeric payload for custom / analysis events.
    Metrics {
        /// Primary score or magnitude.
        score: f32,
        /// Optional secondary value.
        aux: f32,
        /// Application tag.
        tag: u32,
    },
}

/// Versioned, queryable event record for a `VisionIndex`.
///
/// This is the shared contract between tracking, analytics, storage, and host
/// products. It does not embed pixels or edit instructions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EventEnvelope {
    /// Stable event id within an index document.
    pub event_id: EventId,
    /// Temporal and source stamp for the event.
    pub stamp: FrameStamp,
    /// Coarse kind for filtering.
    pub kind: EventKind,
    /// Optional track association.
    pub track_id: Option<TrackId>,
    /// Optional long-lived subject association.
    pub subject_id: Option<SubjectId>,
    /// Optional zone association (also often present in payload).
    pub zone_id: Option<ZoneId>,
    /// Optional evidence handle for reels / audit.
    pub evidence: Option<EvidenceRef>,
    /// Kind-specific payload.
    pub payload: EventPayload,
}

impl EventEnvelope {
    /// Creates an envelope with empty payload and no optional associations.
    #[must_use]
    pub fn new(event_id: EventId, stamp: FrameStamp, kind: EventKind) -> Self {
        Self {
            event_id,
            stamp,
            kind,
            track_id: None,
            subject_id: None,
            zone_id: None,
            evidence: None,
            payload: EventPayload::Empty,
        }
    }

    /// Builder-style track association.
    #[must_use]
    pub fn with_track(mut self, track_id: TrackId) -> Self {
        self.track_id = Some(track_id);
        self
    }

    /// Builder-style subject association.
    #[must_use]
    pub fn with_subject(mut self, subject_id: SubjectId) -> Self {
        self.subject_id = Some(subject_id);
        self
    }

    /// Builder-style zone association.
    #[must_use]
    pub fn with_zone(mut self, zone_id: ZoneId) -> Self {
        self.zone_id = Some(zone_id);
        self
    }

    /// Builder-style evidence handle.
    #[must_use]
    pub fn with_evidence(mut self, evidence: EvidenceRef) -> Self {
        self.evidence = Some(evidence);
        self
    }

    /// Builder-style payload.
    #[must_use]
    pub fn with_payload(mut self, payload: EventPayload) -> Self {
        self.payload = payload;
        self
    }
}
