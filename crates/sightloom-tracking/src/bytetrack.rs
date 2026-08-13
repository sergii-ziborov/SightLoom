//! ByteTrack-compatible multi-object tracker.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};

use crate::{
    ByteTrackConfig, Track, TrackError, TrackState,
    matching::{AssignScratch, MatchCandidate, greedy_iou_assign},
};
use sightloom_core::{ClassId, Detection, Rect, TrackId};

/// Serializable snapshot of a single-source tracker runtime.
#[derive(Clone, Debug, PartialEq)]
pub struct TrackerSnapshot {
    /// Internal frame counter.
    pub frame_id: u64,
    /// Next local track id.
    pub next_id: u32,
    /// Active and lost tracks.
    pub tracks: Vec<Track>,
}

/// `ByteTrack` multi-object tracker with stable IDs.
///
/// Accepts external detections each frame and returns the active tracked set.
/// Does not run inference or decode video.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
pub struct ByteTracker {
    config: ByteTrackConfig,
    frame_id: u64,
    next_id: u32,
    tracks: Vec<Track>,
}

#[cfg(feature = "alloc")]
impl ByteTracker {
    /// Creates a tracker from a validated configuration.
    ///
    /// # Errors
    ///
    /// Returns [`TrackError::InvalidConfig`] when thresholds are invalid.
    pub fn new(config: ByteTrackConfig) -> Result<Self, TrackError> {
        let config = config.validate()?;
        Ok(Self {
            config,
            frame_id: 0,
            next_id: 1,
            tracks: Vec::new(),
        })
    }

    /// Returns the current configuration.
    #[must_use]
    pub const fn config(&self) -> ByteTrackConfig {
        self.config
    }

    /// Returns all non-removed tracks.
    #[must_use]
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// Internal frame counter after the last [`Self::update`].
    #[must_use]
    pub const fn frame_id(&self) -> u64 {
        self.frame_id
    }

    /// Next local track id that will be assigned.
    #[must_use]
    pub const fn next_id(&self) -> u32 {
        self.next_id
    }

    /// Captures a full runtime snapshot for checkpointing.
    #[must_use]
    pub fn snapshot(&self) -> TrackerSnapshot {
        TrackerSnapshot {
            frame_id: self.frame_id,
            next_id: self.next_id,
            tracks: self.tracks.clone(),
        }
    }

    /// Restores counters and track list from a snapshot (config unchanged).
    pub fn restore(&mut self, snapshot: TrackerSnapshot) {
        self.frame_id = snapshot.frame_id;
        self.next_id = snapshot.next_id.max(1);
        self.tracks = snapshot.tracks;
    }

    /// Restores a tracker from a snapshot and validated config.
    ///
    /// # Errors
    ///
    /// Returns config validation errors.
    pub fn from_snapshot(
        config: ByteTrackConfig,
        snapshot: TrackerSnapshot,
    ) -> Result<Self, TrackError> {
        let mut tracker = Self::new(config)?;
        tracker.restore(snapshot);
        Ok(tracker)
    }

    /// Updates the tracker with one frame of detections.
    ///
    /// Assigns `track_id` on the returned detections for matched/activated
    /// objects. Input order of returned detections follows the active track
    /// list (tracked and new), not the original detection order.
    ///
    /// # Errors
    ///
    /// Returns [`TrackError::NonFinite`] only if internal assumptions break;
    /// detections are expected to already be validated by `sightloom-core`.
    pub fn update(&mut self, detections: &[Detection]) -> Result<Vec<Detection>, TrackError> {
        self.frame_id = self.frame_id.saturating_add(1);
        let frame_id = self.frame_id;

        for track in &mut self.tracks {
            track.predict();
        }

        let mut high = Vec::new();
        let mut low = Vec::new();
        for (index, detection) in detections.iter().enumerate() {
            if detection.score() >= self.config.track_high_thresh {
                high.push((index, *detection));
            } else if detection.score() >= self.config.track_low_thresh {
                low.push((index, *detection));
            }
        }

        let mut tracked_indices = Vec::new();
        let mut lost_indices = Vec::new();
        let mut new_indices = Vec::new();
        for (index, track) in self.tracks.iter().enumerate() {
            match track.state {
                TrackState::Tracked => tracked_indices.push(index),
                TrackState::Lost => lost_indices.push(index),
                TrackState::New => new_indices.push(index),
                TrackState::Removed => {}
            }
        }
        // Pool for first association: tracked + new + lost
        let mut pool = Vec::new();
        pool.extend(tracked_indices.iter().copied());
        pool.extend(new_indices.iter().copied());
        pool.extend(lost_indices.iter().copied());

        let (matched_high, unmatched_pool, unmatched_high) =
            self.associate(&pool, &high, self.config.match_thresh);

        for &(track_i, det_i) in &matched_high {
            let detection = high[det_i].1;
            self.tracks[track_i].update(
                detection.bbox(),
                detection.score(),
                detection.class_id(),
                frame_id,
            );
        }

        // Second stage: remaining pool tracks vs low-confidence detections.
        let (matched_low, still_unmatched_pool, _) =
            self.associate(&unmatched_pool, &low, self.config.match_thresh);

        for &(track_i, det_i) in &matched_low {
            let detection = low[det_i].1;
            self.tracks[track_i].update(
                detection.bbox(),
                detection.score(),
                detection.class_id(),
                frame_id,
            );
        }

        for track_i in still_unmatched_pool {
            let track = &mut self.tracks[track_i];
            if track.state == TrackState::Tracked || track.state == TrackState::New {
                track.mark_lost();
            }
        }

        // Activate new tracks from unmatched high-confidence detections.
        for det_i in unmatched_high {
            let detection = high[det_i].1;
            if detection.score() < self.config.track_activation_thresh {
                continue;
            }
            let id = TrackId(self.next_id);
            self.next_id = self.next_id.saturating_add(1);
            self.tracks.push(Track::new(
                id,
                detection.bbox(),
                detection.score(),
                detection.class_id(),
                frame_id,
            ));
        }

        // Expire lost tracks past the buffer.
        for track in &mut self.tracks {
            if track.state == TrackState::Lost
                && track.time_since_update > self.config.max_time_lost
            {
                track.mark_removed();
            }
        }
        self.tracks
            .retain(|track| track.state != TrackState::Removed);

        let mut output = Vec::new();
        for track in &self.tracks {
            if track.state != TrackState::Tracked && track.state != TrackState::New {
                continue;
            }
            let detection = Detection::new(
                track.predicted_bbox(),
                track.score,
                track.class_id,
                Some(track.id),
            )
            .map_err(|_| TrackError::NonFinite)?;
            output.push(detection);
        }
        Ok(output)
    }

    fn associate(
        &self,
        track_indices: &[usize],
        detections: &[(usize, Detection)],
        iou_threshold: f32,
    ) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
        if track_indices.is_empty() || detections.is_empty() {
            return (
                Vec::new(),
                track_indices.to_vec(),
                (0..detections.len()).collect(),
            );
        }

        let track_boxes: Vec<Rect> = track_indices
            .iter()
            .map(|&i| self.tracks[i].predicted_bbox())
            .collect();
        let det_boxes: Vec<Rect> = detections.iter().map(|(_, d)| d.bbox()).collect();
        let track_classes: Vec<Option<ClassId>> = track_indices
            .iter()
            .map(|&i| self.tracks[i].class_id)
            .collect();
        let det_classes: Vec<Option<ClassId>> =
            detections.iter().map(|(_, d)| d.class_id()).collect();

        let cand_cap = track_boxes.len().saturating_mul(det_boxes.len()).max(1);
        let mut candidates = vec![
            MatchCandidate {
                track_index: 0,
                detection_index: 0,
                iou: 0.0,
            };
            cand_cap
        ];
        let mut track_used = vec![false; track_boxes.len()];
        let mut det_used = vec![false; det_boxes.len()];
        let mut matches = vec![(0_usize, 0_usize); track_boxes.len().min(det_boxes.len()).max(1)];
        let mut unmatched_tracks = vec![0_usize; track_boxes.len()];
        let mut unmatched_dets = vec![0_usize; det_boxes.len()];
        let mut scratch = AssignScratch {
            candidates: &mut candidates,
            track_used: &mut track_used,
            detection_used: &mut det_used,
        };

        let result = greedy_iou_assign(
            &track_boxes,
            &det_boxes,
            &track_classes,
            &det_classes,
            self.config.class_aware,
            iou_threshold,
            &mut scratch,
            &mut matches,
            &mut unmatched_tracks,
            &mut unmatched_dets,
        );

        let paired = matches[..result.match_count]
            .iter()
            .map(|&(ti, di)| (track_indices[ti], di))
            .collect();
        let unmatched_pool = unmatched_tracks[..result.unmatched_track_count]
            .iter()
            .map(|&ti| track_indices[ti])
            .collect();
        let unmatched_high = unmatched_dets[..result.unmatched_detection_count].to_vec();
        (paired, unmatched_pool, unmatched_high)
    }
}
