//! In-memory session that builds a `VisionIndex` from detections, zones, and re-id.

use std::collections::HashMap;

use sightloom_analysis::{AnalyticsEvent, ZoneAnalytics, analytics_to_envelope};
use sightloom_core::{
    Detection, EmbeddingRef, EventId, FrameStamp, MediaTime, Point, Rect, SourceId, SubjectId,
    TrackId,
};
use sightloom_index::{SourceEntry, TrackSample, VisionIndex, VisionIndexSnapshot};
use sightloom_reid::{
    EmbeddingError, EmbeddingObservation, IdentityMatch, ReferenceSample, ResolveConfig,
    SubjectGallery, SubjectModality, TrackFragment, aggregate_fragment,
};
use sightloom_tracking::{ByteTrackConfig, ByteTracker, TrackError};

/// Errors raised while materializing a `VisionIndex` session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SessionError {
    /// Tracker configuration or runtime failure.
    Track(TrackError),
    /// Zone analytics capacity or config failure.
    Analytics,
    /// Identity / embedding failure.
    Identity(EmbeddingError),
    /// Snapshot serialization failure message.
    Serialize(String),
}

impl From<TrackError> for SessionError {
    fn from(value: TrackError) -> Self {
        Self::Track(value)
    }
}

impl From<EmbeddingError> for SessionError {
    fn from(value: EmbeddingError) -> Self {
        Self::Identity(value)
    }
}

/// Host-facing session: tracker + identity gallery + `VisionIndex` accumulation.
pub struct IndexSession {
    tracker: ByteTracker,
    index: VisionIndex,
    gallery: SubjectGallery,
    next_event_id: u64,
    /// Stable subject assignment per local track id.
    track_subjects: HashMap<u32, SubjectId>,
    /// Pending embedding observations keyed by track id.
    pending_embeddings: HashMap<u32, Vec<EmbeddingObservation>>,
    /// When true, accepted matches auto-write `subject_id` onto tracks.
    auto_assign_subjects: bool,
    /// Default modality for fragment resolution.
    default_modality: SubjectModality,
}

impl IndexSession {
    /// Creates a session with a validated tracker config and empty index/gallery.
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
            gallery: SubjectGallery::new(),
            next_event_id: 1,
            track_subjects: HashMap::new(),
            pending_embeddings: HashMap::new(),
            auto_assign_subjects: true,
            default_modality: SubjectModality::PersonAppearance,
        })
    }

    /// Registers a media source on the index header.
    pub fn add_source(&mut self, entry: SourceEntry) {
        self.index.add_source(entry);
    }

    /// Enables or disables automatic subject assignment on Accept matches.
    pub fn set_auto_assign_subjects(&mut self, enabled: bool) {
        self.auto_assign_subjects = enabled;
    }

    /// Sets the default re-id modality used when callers omit one.
    pub fn set_default_modality(&mut self, modality: SubjectModality) {
        self.default_modality = modality;
    }

    /// Sets identity resolve thresholds.
    ///
    /// # Errors
    ///
    /// Propagates gallery config validation errors.
    pub fn set_resolve_config(&mut self, config: ResolveConfig) -> Result<(), SessionError> {
        self.gallery.set_resolve_config(config)?;
        Ok(())
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

    /// Returns the identity gallery.
    #[must_use]
    pub fn gallery(&self) -> &SubjectGallery {
        &self.gallery
    }

    /// Mutable access to the identity gallery.
    pub fn gallery_mut(&mut self) -> &mut SubjectGallery {
        &mut self.gallery
    }

    /// Registers a new subject identity in the gallery.
    pub fn register_subject(&mut self, modality: SubjectModality) -> SubjectId {
        self.gallery.register_subject(modality)
    }

    /// Adds a positive/negative/unlabeled reference sample to a subject.
    ///
    /// # Errors
    ///
    /// Propagates gallery errors when the subject is unknown.
    pub fn add_subject_reference(
        &mut self,
        subject_id: SubjectId,
        sample: ReferenceSample,
    ) -> Result<(), SessionError> {
        self.gallery.add_reference(subject_id, sample)?;
        Ok(())
    }

    /// Inserts an embedding vector and records it for a track.
    ///
    /// # Errors
    ///
    /// Propagates embedding store validation errors.
    pub fn note_track_embedding(
        &mut self,
        track_id: TrackId,
        vector: impl Into<Vec<f32>>,
        at: MediaTime,
    ) -> Result<EmbeddingRef, SessionError> {
        let handle = self.gallery.embeddings.insert(vector)?;
        self.pending_embeddings
            .entry(track_id.0)
            .or_default()
            .push(EmbeddingObservation {
                embedding: handle,
                at,
            });
        Ok(handle)
    }

    /// Returns the subject currently assigned to a local track, if any.
    #[must_use]
    pub fn subject_for_track(&self, track_id: TrackId) -> Option<SubjectId> {
        self.track_subjects.get(&track_id.0).copied()
    }

    /// Ingests one frame of detections: track → append track samples.
    ///
    /// When a track already has an assigned subject, samples carry that id.
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
            let subject_id = self.track_subjects.get(&track_id.0).copied();
            self.index.push_track(TrackSample {
                source_id: stamp.source_id,
                frame_index: stamp.frame_index,
                pts: stamp.pts,
                track_id,
                subject_id,
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

    /// Aggregates pending embeddings for `track_id`, resolves identity, and
    /// patches the latest track sample when a subject is assigned.
    ///
    /// # Errors
    ///
    /// Returns identity errors when no pending embeddings exist or pooling fails.
    pub fn resolve_track_identity(
        &mut self,
        track_id: TrackId,
        source_id: SourceId,
        modality: Option<SubjectModality>,
        at: MediaTime,
    ) -> Result<(TrackFragment, Vec<IdentityMatch>), SessionError> {
        let observations = self
            .pending_embeddings
            .remove(&track_id.0)
            .ok_or(EmbeddingError::InvalidVector)?;
        if observations.is_empty() {
            return Err(EmbeddingError::InvalidVector.into());
        }
        let modality = modality.unwrap_or(self.default_modality);
        let known_subject = self.track_subjects.get(&track_id.0).copied();
        let fragment = aggregate_fragment(
            &mut self.gallery.embeddings,
            track_id,
            source_id,
            modality,
            &observations,
            known_subject,
        )?;
        let (fragment, matches) =
            self.gallery
                .resolve_and_audit(fragment, self.auto_assign_subjects, at);
        if let Some(subject_id) = fragment.subject_id {
            self.track_subjects.insert(track_id.0, subject_id);
            self.patch_latest_track_subject(track_id, subject_id);
        }
        Ok((fragment, matches))
    }

    /// Resolves identity for every track that has pending embeddings.
    ///
    /// Returns the number of tracks resolved.
    ///
    /// # Errors
    ///
    /// Propagates the first identity resolution error.
    pub fn resolve_pending_identities(
        &mut self,
        stamp: FrameStamp,
        modality: Option<SubjectModality>,
    ) -> Result<usize, SessionError> {
        let track_ids: Vec<u32> = self.pending_embeddings.keys().copied().collect();
        let mut resolved = 0_usize;
        for track_id in track_ids {
            self.resolve_track_identity(TrackId(track_id), stamp.source_id, modality, stamp.pts)?;
            resolved = resolved.saturating_add(1);
        }
        Ok(resolved)
    }

    /// Applies a manual identity confirmation and patches the track map/sample.
    ///
    /// # Errors
    ///
    /// Propagates gallery audit lookup errors.
    pub fn confirm_identity(
        &mut self,
        audit_id: u64,
        confirm: bool,
        subject_id: Option<SubjectId>,
        track_id: TrackId,
    ) -> Result<(), SessionError> {
        self.gallery.confirm_manual(audit_id, confirm, subject_id)?;
        if confirm {
            if let Some(subject_id) = subject_id {
                self.track_subjects.insert(track_id.0, subject_id);
                self.patch_latest_track_subject(track_id, subject_id);
            }
        } else {
            self.track_subjects.remove(&track_id.0);
        }
        Ok(())
    }

    /// Feeds tracked boxes into a zone monitor and records envelopes.
    ///
    /// Envelopes inherit any subject already assigned to the track.
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
            let subject_id = self.track_subjects.get(&track_id.0).copied();
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
                let envelope = analytics_to_envelope(event_id, stamp, event, subject_id);
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

    /// Attaches a mask handle by appending a correction sample for `track_id`.
    pub fn attach_mask_to_latest_track(&mut self, track_id: TrackId, mask_ref: u64) -> bool {
        let samples = self.index.tracks.samples();
        let Some(sample) = samples
            .iter()
            .rev()
            .find(|sample| sample.track_id == track_id)
            .copied()
        else {
            return false;
        };
        let mut updated = sample;
        updated.mask_ref = mask_ref;
        if let Some(subject_id) = self.track_subjects.get(&track_id.0).copied() {
            updated.subject_id = Some(subject_id);
        }
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

    fn patch_latest_track_subject(&mut self, track_id: TrackId, subject_id: SubjectId) {
        let samples = self.index.tracks.samples();
        let Some(sample) = samples
            .iter()
            .rev()
            .find(|sample| sample.track_id == track_id)
            .copied()
        else {
            return;
        };
        let mut updated = sample;
        updated.subject_id = Some(subject_id);
        self.index.push_track(updated);
    }
}
