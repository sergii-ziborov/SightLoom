//! Per-source tracker pool with globally unique track uids.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::collections::HashMap;

use crate::{ByteTrackConfig, ByteTracker, TrackError, TrackerSnapshot};
use sightloom_core::{Detection, SourceId, TrackKey, TrackUid};

/// One detection result annotated with multi-source identity keys.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackedDetection {
    /// Detection with local [`TrackId`] set by the per-source tracker.
    pub detection: Detection,
    /// Composite source-local key.
    pub track_key: TrackKey,
    /// Globally unique track uid for the session.
    pub track_uid: TrackUid,
}

/// Multi-source tracker: one independent motion model per [`SourceId`].
///
/// Local track ids may collide across sources; [`TrackUid`] values do not.
#[cfg(feature = "std")]
#[derive(Clone, Debug)]
pub struct MultiSourceTracker {
    config: ByteTrackConfig,
    trackers: HashMap<u32, ByteTracker>,
    /// `(source_id, local_track_id) -> TrackUid`
    uids: HashMap<(u32, u32), TrackUid>,
    next_uid: u64,
}

#[cfg(feature = "std")]
impl MultiSourceTracker {
    /// Creates an empty multi-source tracker pool.
    ///
    /// # Errors
    ///
    /// Returns invalid config errors.
    pub fn new(config: ByteTrackConfig) -> Result<Self, TrackError> {
        let _ = config.validate()?;
        Ok(Self {
            config,
            trackers: HashMap::new(),
            uids: HashMap::new(),
            next_uid: 1,
        })
    }

    /// Tracker configuration shared by every source.
    #[must_use]
    pub const fn config(&self) -> ByteTrackConfig {
        self.config
    }

    /// Next global track uid counter.
    #[must_use]
    pub const fn next_uid(&self) -> u64 {
        self.next_uid
    }

    /// Snapshot of uid map and per-source trackers for checkpoints.
    #[must_use]
    pub fn checkpoint(&self) -> MultiSourceCheckpoint {
        let mut sources = Vec::new();
        for (source, tracker) in &self.trackers {
            sources.push(SourceTrackerCheckpoint {
                source_id: *source,
                tracker: tracker.snapshot(),
            });
        }
        sources.sort_by_key(|s| s.source_id);
        let mut uids = Vec::new();
        for ((source, local), uid) in &self.uids {
            uids.push(UidMapEntry {
                source_id: *source,
                local_track_id: *local,
                track_uid: uid.0,
            });
        }
        uids.sort_by_key(|e| (e.source_id, e.local_track_id));
        MultiSourceCheckpoint {
            next_uid: self.next_uid,
            sources,
            uids,
        }
    }

    /// Restores multi-source state from a checkpoint.
    ///
    /// # Errors
    ///
    /// Propagates tracker restore errors.
    pub fn restore(
        config: ByteTrackConfig,
        checkpoint: MultiSourceCheckpoint,
    ) -> Result<Self, TrackError> {
        let mut pool = Self::new(config)?;
        pool.next_uid = checkpoint.next_uid.max(1);
        for source in checkpoint.sources {
            let tracker = ByteTracker::from_snapshot(config, source.tracker)?;
            pool.trackers.insert(source.source_id, tracker);
        }
        for entry in checkpoint.uids {
            pool.uids.insert(
                (entry.source_id, entry.local_track_id),
                TrackUid(entry.track_uid),
            );
            pool.next_uid = pool.next_uid.max(entry.track_uid.saturating_add(1));
        }
        Ok(pool)
    }

    /// Updates only the tracker for `source_id`.
    ///
    /// # Errors
    ///
    /// Propagates single-source tracker errors.
    pub fn update(
        &mut self,
        source_id: SourceId,
        detections: &[Detection],
    ) -> Result<Vec<TrackedDetection>, TrackError> {
        if !self.trackers.contains_key(&source_id.0) {
            let tracker = ByteTracker::new(self.config)?;
            self.trackers.insert(source_id.0, tracker);
        }
        let tracker = self
            .trackers
            .get_mut(&source_id.0)
            .ok_or(TrackError::InvalidConfig)?;
        let tracked = tracker.update(detections)?;
        let mut out = Vec::with_capacity(tracked.len());
        for detection in tracked {
            let Some(local) = detection.track_id() else {
                continue;
            };
            let key = TrackKey::new(source_id, local);
            let uid = *self.uids.entry((source_id.0, local.0)).or_insert_with(|| {
                let uid = TrackUid(self.next_uid);
                self.next_uid = self.next_uid.saturating_add(1);
                uid
            });
            out.push(TrackedDetection {
                detection,
                track_key: key,
                track_uid: uid,
            });
        }
        Ok(out)
    }

    /// Looks up the global uid for a track key.
    #[must_use]
    pub fn uid_of(&self, key: TrackKey) -> Option<TrackUid> {
        self.uids
            .get(&(key.source_id.0, key.local_track_id.0))
            .copied()
    }

    /// Number of independent source trackers currently allocated.
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.trackers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ByteTrackConfig;
    use sightloom_core::{Detection, Rect, SourceId};

    fn det(left: f32, top: f32, right: f32, bottom: f32) -> Detection {
        Detection::new(
            Rect::new(left, top, right, bottom).unwrap(),
            0.9,
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn distinct_uids_for_same_local_id_across_sources() {
        let mut pool = MultiSourceTracker::new(ByteTrackConfig {
            track_high_thresh: 0.5,
            track_activation_thresh: 0.5,
            track_low_thresh: 0.1,
            match_thresh: 0.3,
            max_time_lost: 30,
            class_aware: false,
        })
        .unwrap();
        let a = pool
            .update(SourceId(1), &[det(0.0, 0.0, 10.0, 20.0)])
            .unwrap();
        let b = pool
            .update(SourceId(2), &[det(100.0, 0.0, 110.0, 20.0)])
            .unwrap();
        assert_eq!(a[0].track_key.local_track_id.0, 1);
        assert_eq!(b[0].track_key.local_track_id.0, 1);
        assert_ne!(a[0].track_uid, b[0].track_uid);
        assert_eq!(pool.source_count(), 2);
    }
}

/// Checkpoint payload for multi-source tracking.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiSourceCheckpoint {
    /// Next global uid.
    pub next_uid: u64,
    /// Per-source tracker snapshots.
    pub sources: Vec<SourceTrackerCheckpoint>,
    /// Composite key to uid map.
    pub uids: Vec<UidMapEntry>,
}

/// One source tracker inside a multi-source checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceTrackerCheckpoint {
    /// Source id.
    pub source_id: u32,
    /// Tracker snapshot.
    pub tracker: TrackerSnapshot,
}

/// One uid map row.
#[derive(Clone, Debug, PartialEq)]
pub struct UidMapEntry {
    /// Source id.
    pub source_id: u32,
    /// Local track id.
    pub local_track_id: u32,
    /// Global uid.
    pub track_uid: u64,
}
