//! `ByteTrack` configuration.

use crate::TrackError;

/// Runtime parameters for [`crate::ByteTracker`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ByteTrackConfig {
    /// Detections at or above this score enter high-confidence matching.
    pub track_high_thresh: f32,
    /// Detections at or above this score may create new tracks.
    pub track_activation_thresh: f32,
    /// Floor for low-confidence second-stage matching.
    pub track_low_thresh: f32,
    /// Minimum intersection-over-union required for a valid match.
    pub match_thresh: f32,
    /// Frames a lost track is retained for re-association.
    pub max_time_lost: u32,
    /// When true, only same-class detections may match a track.
    pub class_aware: bool,
}

impl Default for ByteTrackConfig {
    fn default() -> Self {
        Self {
            track_high_thresh: 0.5,
            track_activation_thresh: 0.6,
            track_low_thresh: 0.1,
            match_thresh: 0.8,
            max_time_lost: 30,
            class_aware: false,
        }
    }
}

impl ByteTrackConfig {
    /// Validates finite thresholds in `0.0..=1.0` and non-zero lost buffer.
    ///
    /// # Errors
    ///
    /// Returns [`TrackError::InvalidConfig`] when any field is out of range.
    pub fn validate(self) -> Result<Self, TrackError> {
        let thresholds = [
            self.track_high_thresh,
            self.track_activation_thresh,
            self.track_low_thresh,
            self.match_thresh,
        ];
        for value in thresholds {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(TrackError::InvalidConfig);
            }
        }
        if self.track_low_thresh > self.track_high_thresh {
            return Err(TrackError::InvalidConfig);
        }
        if self.max_time_lost == 0 {
            return Err(TrackError::InvalidConfig);
        }
        Ok(self)
    }
}
