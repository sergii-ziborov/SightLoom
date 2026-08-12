//! In-memory session that builds a `VisionIndex` from detections and zones.

use sightloom_analytics::{AnalyticsEvent, ZoneAnalytics, analytics_to_envelope};
use sightloom_core::{Detection, EventId, FrameStamp, Point, Rect, TrackId};
use sightloom_memory::{SourceEntry, TrackSample, VisionIndex, VisionIndexSnapshot};
use sightloom_track::{ByteTrackConfig, ByteTracker, TrackError};

/// Errors raised while materializing a `VisionIndex` session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SessionError {
    /// Tracker configuration or runtime failure.
    Track(TrackError),
    /// Zone analytics capacity or config failure.
    Analytics,
    /// Snapshot serialization failure message.
    Serialize(String),
}

impl From<TrackError> for SessionError {
    fn from(value: TrackError) -> Self {
        Self::Track(value)
    }
}

/// Host-facing session: tracker + `VisionIndex` accumulation.
pub struct IndexSession {
    tracker: ByteTracker,
    index: VisionIndex,
    next_event_id: u64,
}

impl IndexSession {
    /// Creates a session with a validated tracker config and empty index.
    ///
    /// # Errors
    ///
    /// Returns tracker configuration errors.
    pub fn new(
        name: impl Into<String>,
        track_config: ByteTrackConfig,
    ) -> Result<Self, SessionError> {
        Ok(Self {
            tracker: ByteTracker::new(track_config)?,
            index: VisionIndex::new(name),
            next_event_id: 1,
        })
    }

    /// Registers a media source on the index header.
    pub fn add_source(&mut self, entry: SourceEntry) {
        self.index.add_source(entry);
    }

    /// Returns a shared reference to the live index.
    #[must_use]
    pub fn index(&self) -> &VisionIndex {
        &self.index
    }

    /// Mutable access for advanced entity writers (appearances, subjects, …).
    pub fn index_mut(&mut self) -> &mut VisionIndex {
        &mut self.index
    }

    /// Ingests one frame of detections: track → append track samples.
    ///
    /// Returns the tracked detections (with stable track ids) for zone/mask
    /// follow-up by the host.
    ///
    /// # Errors
    ///
    /// Propagates tracker errors.
    pub fn ingest_detections(
        &mut self,
        stamp: FrameStamp,
        detections: &[Detection],
    ) -> Result<Vec<Detection>, SessionError> {
        let tracked = self.tracker.update(detections)?;
        for detection in &tracked {
            let Some(track_id) = detection.track_id() else {
                continue;
            };
            let bbox = detection.bbox();
            self.index.push_track(TrackSample {
                source_id: stamp.source_id,
                frame_index: stamp.frame_index,
                pts: stamp.pts,
                track_id,
                subject_id: None,
                class_id: detection.class_id(),
                left: bbox.left(),
                top: bbox.top(),
                right: bbox.right(),
                bottom: bbox.bottom(),
                confidence: detection.score(),
                mask_ref: 0,
            });
        }
        Ok(tracked)
    }

    /// Feeds tracked boxes into a zone monitor and records envelopes.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::Analytics`] when the zone monitor cannot accept
    /// an update (capacity / output).
    pub fn ingest_zone_updates<const N: usize>(
        &mut self,
        stamp: FrameStamp,
        zone: &mut ZoneAnalytics<'_, N>,
        tracked: &[Detection],
    ) -> Result<usize, SessionError> {
        let mut total = 0_usize;
        let mut output = [AnalyticsEvent::OccupancyChanged {
            zone_id: sightloom_core::ZoneId(0),
            occupancy: 0,
        }; 8];

        for detection in tracked {
            let Some(track_id) = detection.track_id() else {
                continue;
            };
            let count = zone
                .update(
                    track_id,
                    detection.bbox(),
                    detection.class_id(),
                    None,
                    stamp.frame_index,
                    stamp.pts,
                    &mut output,
                )
                .map_err(|_| SessionError::Analytics)?;
            for event in output.iter().take(count).copied() {
                let event_id = EventId(self.next_event_id);
                self.next_event_id = self.next_event_id.saturating_add(1);
                let envelope = analytics_to_envelope(event_id, stamp, event, None);
                self.index.push_event(envelope);
                total = total.saturating_add(1);
            }
        }
        Ok(total)
    }

    /// Stores a compact mask blob and returns its handle value for track rows.
    pub fn store_mask_bytes(&mut self, bytes: impl Into<Vec<u8>>) -> u64 {
        self.index.masks.insert(bytes).0
    }

    /// Attaches a mask handle to the latest track sample for `track_id` if any.
    pub fn attach_mask_to_latest_track(&mut self, track_id: TrackId, mask_ref: u64) -> bool {
        let samples = self.index.tracks.samples();
        // Rebuild with last matching sample updated — TrackStream is append-only.
        // Hosts can also write mask_ref at ingest time; this is a convenience.
        let Some((index, sample)) = samples
            .iter()
            .enumerate()
            .rev()
            .find(|(_, sample)| sample.track_id == track_id)
            .map(|(i, s)| (i, *s))
        else {
            return false;
        };
        let mut updated = sample;
        updated.mask_ref = mask_ref;
        // TrackStream has no in-place update; push a correction sample.
        let _ = index;
        self.index.push_track(updated);
        true
    }

    /// Materializes a JSON `VisionIndex` snapshot.
    ///
    /// # Errors
    ///
    /// Returns serialization failures.
    pub fn materialize_json(&self) -> Result<String, SessionError> {
        self.index
            .validate()
            .map_err(|error| SessionError::Serialize(format!("invalid index: {error:?}")))?;
        VisionIndexSnapshot::from_index(&self.index)
            .to_json()
            .map_err(|error| SessionError::Serialize(format!("{error:?}")))
    }

    /// Helper for tests: bottom-center anchor point of a box.
    #[must_use]
    pub fn bottom_center(bbox: Rect) -> Point {
        Point::new(bbox.left() * 0.5 + bbox.right() * 0.5, bbox.bottom())
            .unwrap_or_else(|_| bbox.center())
    }
}
