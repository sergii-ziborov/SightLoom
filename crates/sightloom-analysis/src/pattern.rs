//! Pattern records produced by analysis backends.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use sightloom_core::{EventId, PatternId, SubjectId};

/// Pattern kind tags.
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
