//! Keypoint sets for pose / landmarks (handles + compact store).
//!
//! Pixels stay with the host; `SightLoom` stores coordinates and optional scores.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use sightloom_core::{CoreError, KeypointSetRef, Point};

/// One 2D keypoint with optional confidence and class/index tag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Keypoint {
    /// Image coordinate.
    pub point: Point,
    /// Confidence in `0.0..=1.0` (not clamped).
    pub score: f32,
    /// Host-defined joint / landmark index.
    pub index: u16,
}

impl Keypoint {
    /// Creates a keypoint when score is finite.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::NonFinite`] for non-finite scores.
    pub fn new(point: Point, score: f32, index: u16) -> Result<Self, CoreError> {
        if !score.is_finite() {
            return Err(CoreError::NonFinite);
        }
        Ok(Self {
            point,
            score,
            index,
        })
    }
}

/// A set of keypoints for one detection / observation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KeypointSet {
    /// Points.
    pub points: Vec<Keypoint>,
}

impl KeypointSet {
    /// Empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of keypoints.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// True when empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// In-memory store of keypoint sets (out-of-line from observations).
#[derive(Clone, Debug, Default)]
pub struct KeypointStore {
    next_id: u64,
    entries: Vec<(KeypointSetRef, KeypointSet)>,
}

impl KeypointStore {
    /// Empty store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: 1,
            entries: Vec::new(),
        }
    }

    /// Inserts a set and returns a handle.
    pub fn insert(&mut self, set: KeypointSet) -> KeypointSetRef {
        let id = KeypointSetRef(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        self.entries.push((id, set));
        id
    }

    /// Looks up a set.
    #[must_use]
    pub fn get(&self, handle: KeypointSetRef) -> Option<&KeypointSet> {
        self.entries
            .iter()
            .find(|(k, _)| *k == handle)
            .map(|(_, v)| v)
    }

    /// Number of stored sets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
