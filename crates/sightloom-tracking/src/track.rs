//! Single-object track state for `ByteTrack`.

use crate::kalman::KalmanState;
use sightloom_core::{ClassId, Rect, TrackId};

/// Lifecycle state of a track.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrackState {
    /// Newly created; not yet confirmed.
    New,
    /// Actively matched in recent frames.
    Tracked,
    /// Temporarily unmatched; retained for re-association.
    Lost,
    /// Expired and eligible for removal.
    Removed,
}

/// One tracked object with Kalman state and metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Track {
    /// Stable track identifier.
    pub id: TrackId,
    /// Current lifecycle state.
    pub state: TrackState,
    /// Optional class label carried from the activating detection.
    pub class_id: Option<ClassId>,
    /// Kalman filter state.
    pub kalman: KalmanState,
    /// Last associated detection score.
    pub score: f32,
    /// Frames since the last successful match.
    pub time_since_update: u32,
    /// Total frames this track has been observed.
    pub hits: u32,
    /// Frame index when the track was created.
    pub start_frame: u64,
    /// Frame index of the latest update.
    pub frame_id: u64,
}

impl Track {
    /// Creates a new track from a detection box.
    #[must_use]
    pub fn new(
        id: TrackId,
        rect: Rect,
        score: f32,
        class_id: Option<ClassId>,
        frame_id: u64,
    ) -> Self {
        Self {
            id,
            state: TrackState::New,
            class_id,
            kalman: KalmanState::initiate(rect),
            score,
            time_since_update: 0,
            hits: 1,
            start_frame: frame_id,
            frame_id,
        }
    }

    /// Predicted bounding box.
    #[must_use]
    pub fn predicted_bbox(self) -> Rect {
        self.kalman.to_rect()
    }

    /// Runs the Kalman predict step.
    pub fn predict(&mut self) {
        self.kalman.predict();
        self.time_since_update = self.time_since_update.saturating_add(1);
    }

    /// Associates a detection and marks the track as tracked.
    pub fn update(&mut self, rect: Rect, score: f32, class_id: Option<ClassId>, frame_id: u64) {
        self.kalman.update(rect);
        self.score = score;
        if class_id.is_some() {
            self.class_id = class_id;
        }
        self.time_since_update = 0;
        self.hits = self.hits.saturating_add(1);
        self.frame_id = frame_id;
        self.state = if self.hits >= 2 {
            TrackState::Tracked
        } else {
            TrackState::New
        };
    }

    /// Marks the track as lost.
    pub fn mark_lost(&mut self) {
        self.state = TrackState::Lost;
    }

    /// Marks the track as removed.
    pub fn mark_removed(&mut self) {
        self.state = TrackState::Removed;
    }
}
