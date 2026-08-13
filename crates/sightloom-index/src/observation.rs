//! Rich frame-level observations above compact detections.

use crate::{ObservationAttributes, OrientedRect};
use sightloom_core::{
    ClassId, CoreError, Detection, EmbeddingRef, EvidenceRef, FrameStamp, KeypointSetRef, MaskRef,
    ObservationId, Rect, SubjectId, TrackId,
};

/// A rich, model-neutral observation for video understanding pipelines.
///
/// Compact [`Detection`] remains in `sightloom-core` for hot paths. This type
/// lives one layer above and carries identity, evidence, and optional mask /
/// embedding handles without owning pixel buffers.
///
/// Append-only revision semantics mirror track samples: corrections set
/// [`Self::supersedes`] to the prior observation id and raise [`Self::revision`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Observation {
    /// Unique observation id within a processing context.
    pub id: ObservationId,
    /// When set, this row supersedes that observation id (correction).
    pub supersedes: Option<u64>,
    /// Revision along this lineage (`1` = first).
    pub revision: u32,
    /// Optional host idempotency key (`0` = none).
    pub idempotency_key: u64,
    /// Source and temporal stamp.
    pub stamp: FrameStamp,
    /// Axis-aligned bounding box.
    pub bbox: Rect,
    /// Detector or tracker confidence in `0.0..=1.0` (not clamped).
    pub confidence: f32,
    /// Optional class label.
    pub class_id: Option<ClassId>,
    /// Optional track association.
    pub track_id: Option<TrackId>,
    /// Optional stable subject / identity.
    pub subject_id: Option<SubjectId>,
    /// Optional out-of-line mask handle.
    pub mask: Option<MaskRef>,
    /// Optional oriented box (P1 geometry ops deferred).
    pub oriented_box: Option<OrientedRect>,
    /// Optional keypoints handle (P1 deferred).
    pub keypoints: Option<KeypointSetRef>,
    /// Optional embedding handle for re-id / memory.
    pub embedding: Option<EmbeddingRef>,
    /// Compact attributes.
    pub attributes: ObservationAttributes,
    /// Provenance / evidence handle for reels and audit.
    pub provenance: EvidenceRef,
}

impl Observation {
    /// Creates an observation when confidence is finite.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::NonFinite`] when `confidence` is NaN or infinite.
    pub fn new(
        id: ObservationId,
        stamp: FrameStamp,
        bbox: Rect,
        confidence: f32,
        provenance: EvidenceRef,
    ) -> Result<Self, CoreError> {
        if !confidence.is_finite() {
            return Err(CoreError::NonFinite);
        }
        Ok(Self {
            id,
            supersedes: None,
            revision: 1,
            idempotency_key: 0,
            stamp,
            bbox,
            confidence,
            class_id: None,
            track_id: None,
            subject_id: None,
            mask: None,
            oriented_box: None,
            keypoints: None,
            embedding: None,
            attributes: ObservationAttributes::empty(),
            provenance,
        })
    }

    /// Builder-style idempotency key.
    #[must_use]
    pub fn with_idempotency_key(mut self, key: u64) -> Self {
        self.idempotency_key = key;
        self
    }

    /// Marks this row as superseding a prior observation id and bumps revision.
    #[must_use]
    pub fn with_supersedes(mut self, prior_id: u64, prior_revision: u32) -> Self {
        self.supersedes = Some(prior_id);
        self.revision = prior_revision.saturating_add(1).max(1);
        self
    }

    /// Builds a rich observation from a compact detection and frame stamp.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::NonFinite`] when the detection score is non-finite
    /// (already guaranteed by `Detection`, kept for API symmetry).
    pub fn from_detection(
        id: ObservationId,
        stamp: FrameStamp,
        detection: Detection,
        provenance: EvidenceRef,
    ) -> Result<Self, CoreError> {
        let mut observation =
            Self::new(id, stamp, detection.bbox(), detection.score(), provenance)?;
        observation.class_id = detection.class_id();
        observation.track_id = detection.track_id();
        Ok(observation)
    }

    /// Projects back to a compact detection for core algorithms.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::NonFinite`] when confidence is non-finite.
    pub fn to_detection(self) -> Result<Detection, CoreError> {
        Detection::new(self.bbox, self.confidence, self.class_id, self.track_id)
    }

    /// Builder-style track assignment.
    #[must_use]
    pub fn with_track_id(mut self, track_id: TrackId) -> Self {
        self.track_id = Some(track_id);
        self
    }

    /// Builder-style subject assignment.
    #[must_use]
    pub fn with_subject_id(mut self, subject_id: SubjectId) -> Self {
        self.subject_id = Some(subject_id);
        self
    }

    /// Builder-style mask handle.
    #[must_use]
    pub fn with_mask(mut self, mask: MaskRef) -> Self {
        self.mask = Some(mask);
        self
    }

    /// Builder-style class label.
    #[must_use]
    pub fn with_class_id(mut self, class_id: ClassId) -> Self {
        self.class_id = Some(class_id);
        self
    }
}

/// Effective observation view: rows not superseded by a later observation.
#[must_use]
pub fn effective_observations(items: &[Observation]) -> Vec<Observation> {
    let superseded: Vec<u64> = items.iter().filter_map(|o| o.supersedes).collect();
    items
        .iter()
        .copied()
        .filter(|o| !superseded.contains(&o.id.0))
        .collect()
}

/// True when `key != 0` already appears on any observation (host idempotency).
#[must_use]
pub fn observation_idempotency_seen(items: &[Observation], key: u64) -> bool {
    key != 0 && items.iter().any(|o| o.idempotency_key == key)
}
