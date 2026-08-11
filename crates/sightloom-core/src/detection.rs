//! Compact detections and caller-owned detection batches.

use crate::{CoreError, Rect};

/// A model-specific class identifier.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ClassId(pub u16);

/// An externally assigned track identifier.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TrackId(pub u32);

/// An application-specific zone identifier.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ZoneId(pub u16);

/// A validated object detection with optional typed metadata.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Detection {
    bbox: Rect,
    score: f32,
    class_id: Option<ClassId>,
    track_id: Option<TrackId>,
}

impl Detection {
    /// Creates a detection when its score is finite.
    ///
    /// Finite scores are preserved without clamping.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::NonFinite`] when `score` is NaN or infinite.
    pub fn new(
        bbox: Rect,
        score: f32,
        class_id: Option<ClassId>,
        track_id: Option<TrackId>,
    ) -> Result<Self, CoreError> {
        if !score.is_finite() {
            return Err(CoreError::NonFinite);
        }

        Ok(Self {
            bbox,
            score,
            class_id,
            track_id,
        })
    }

    /// Returns the detection bounding box.
    #[must_use]
    pub const fn bbox(self) -> Rect {
        self.bbox
    }

    /// Returns the model confidence score.
    #[must_use]
    pub const fn score(self) -> f32 {
        self.score
    }

    /// Returns the optional class identifier.
    #[must_use]
    pub const fn class_id(self) -> Option<ClassId> {
        self.class_id
    }

    /// Returns the optional external track identifier.
    #[must_use]
    pub const fn track_id(self) -> Option<TrackId> {
        self.track_id
    }
}

/// A detection batch backed by mutable storage owned by the caller.
#[derive(Debug)]
pub struct DetectionBatch<'a> {
    storage: &'a mut [Detection],
    len: usize,
}

impl<'a> DetectionBatch<'a> {
    /// Creates an empty batch over caller-owned storage.
    #[must_use]
    pub fn new(storage: &'a mut [Detection]) -> Self {
        Self { storage, len: 0 }
    }

    /// Creates a batch whose complete caller-owned slice contains valid data.
    #[must_use]
    pub fn from_filled(storage: &'a mut [Detection]) -> Self {
        let len = storage.len();
        Self { storage, len }
    }

    /// Appends a detection without reallocating or truncating existing data.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InsufficientCapacity`] without modifying the batch
    /// when the caller-owned storage is full.
    pub fn push(&mut self, detection: Detection) -> Result<(), CoreError> {
        let slot = self
            .storage
            .get_mut(self.len)
            .ok_or(CoreError::InsufficientCapacity)?;
        *slot = detection;
        self.len += 1;
        Ok(())
    }

    /// Returns the valid prefix of the caller-owned storage.
    #[must_use]
    pub fn as_slice(&self) -> &[Detection] {
        &self.storage[..self.len]
    }
}
