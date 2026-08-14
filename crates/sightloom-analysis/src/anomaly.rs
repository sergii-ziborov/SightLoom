//! Backend-neutral anomaly events.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use sightloom_core::{AnomalyId, EventId, MediaTime, SourceId, SubjectId};

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
    /// Cross-camera hop violates topology travel window (impossible).
    ImpossibleCrossCameraHop,
    /// Rare camera-to-camera transition vs relational baseline.
    UnusualCameraTransition,
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
