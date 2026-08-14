//! Compact track sample records for sidecar streams.

use sightloom_core::{ClassId, MediaTime, SourceId, SubjectId, TrackId, TrackKey, TrackUid};

/// One track sample suitable for CBOR and Arrow-shaped columnar export.
///
/// Append-only stream: corrections push a new row with [`Self::supersedes`]
/// pointing at the prior [`Self::sample_id`] and a higher [`Self::revision`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackSample {
    /// Monotonic sample id within the stream (`0` = assign on push).
    pub sample_id: u64,
    /// When set, this row supersedes that sample id (correction / revision).
    pub supersedes: Option<u64>,
    /// Revision number for this logical observation lineage (`1` = first).
    pub revision: u32,
    /// Optional host idempotency key (hash); `0` = none.
    pub idempotency_key: u64,
    /// Source camera or file.
    pub source_id: SourceId,
    /// Frame index within the source.
    pub frame_index: u64,
    /// Presentation timestamp.
    pub pts: MediaTime,
    /// Local track id within [`Self::source_id`].
    pub track_id: TrackId,
    /// Globally unique track id across sources (`None` if not assigned yet).
    pub track_uid: Option<TrackUid>,
    /// Optional subject linkage.
    pub subject_id: Option<SubjectId>,
    /// Optional class.
    pub class_id: Option<ClassId>,
    /// Bounding box edges.
    pub left: f32,
    /// Top edge.
    pub top: f32,
    /// Right edge.
    pub right: f32,
    /// Bottom edge.
    pub bottom: f32,
    /// Confidence.
    pub confidence: f32,
    /// Optional mask store handle (`0` means none).
    pub mask_ref: u64,
}

impl TrackSample {
    /// Composite source-local track key.
    #[must_use]
    pub const fn track_key(self) -> TrackKey {
        TrackKey::new(self.source_id, self.track_id)
    }
}

/// In-memory append-only track stream (host conveniences).
#[cfg(feature = "std")]
#[derive(Clone, Debug, Default)]
pub struct TrackStream {
    samples: Vec<TrackSample>,
    next_sample_id: u64,
}

#[cfg(feature = "std")]
impl TrackStream {
    /// Creates an empty stream.
    #[must_use]
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            next_sample_id: 1,
        }
    }

    /// Appends a sample, assigning [`TrackSample::sample_id`] when it is `0`.
    pub fn push(&mut self, mut sample: TrackSample) {
        if sample.sample_id == 0 {
            sample.sample_id = self.next_sample_id;
            self.next_sample_id = self.next_sample_id.saturating_add(1);
        } else {
            self.next_sample_id = self.next_sample_id.max(sample.sample_id.saturating_add(1));
        }
        if sample.revision == 0 {
            sample.revision = 1;
        }
        self.samples.push(sample);
    }

    /// Appends a correction that supersedes `prior_id`.
    pub fn push_revision(&mut self, mut sample: TrackSample, prior_id: u64) {
        sample.supersedes = Some(prior_id);
        // Always allocate a new sample id for the revision row.
        sample.sample_id = 0;
        let prior_rev = self
            .samples
            .iter()
            .find(|s| s.sample_id == prior_id)
            .map_or(0, |s| s.revision);
        sample.revision = prior_rev.saturating_add(1).max(1);
        self.push(sample);
    }

    /// Returns all samples (immutable audit view, including superseded rows).
    #[must_use]
    pub fn samples(&self) -> &[TrackSample] {
        &self.samples
    }

    /// Effective/current view: samples not superseded by a later row.
    #[must_use]
    pub fn effective_samples(&self) -> Vec<TrackSample> {
        let superseded: Vec<u64> = self.samples.iter().filter_map(|s| s.supersedes).collect();
        self.samples
            .iter()
            .copied()
            .filter(|s| !superseded.contains(&s.sample_id))
            .collect()
    }

    /// True when `key != 0` already appears on any sample (host idempotency).
    #[must_use]
    pub fn idempotency_seen(&self, key: u64) -> bool {
        key != 0 && self.samples.iter().any(|s| s.idempotency_key == key)
    }

    /// Next sample id that will be assigned.
    #[must_use]
    pub const fn next_sample_id(&self) -> u64 {
        self.next_sample_id
    }

    /// Filters samples for a local track id (all sources).
    #[must_use]
    pub fn for_track(&self, track_id: TrackId) -> Vec<TrackSample> {
        self.samples
            .iter()
            .copied()
            .filter(|sample| sample.track_id == track_id)
            .collect()
    }

    /// Filters samples for a composite track key.
    #[must_use]
    pub fn for_track_key(&self, key: TrackKey) -> Vec<TrackSample> {
        self.samples
            .iter()
            .copied()
            .filter(|sample| sample.track_key() == key)
            .collect()
    }

    /// Filters samples for a global track uid.
    #[must_use]
    pub fn for_track_uid(&self, track_uid: TrackUid) -> Vec<TrackSample> {
        self.samples
            .iter()
            .copied()
            .filter(|sample| sample.track_uid == Some(track_uid))
            .collect()
    }

    /// Filters samples for a subject id.
    #[must_use]
    pub fn for_subject(&self, subject_id: SubjectId) -> Vec<TrackSample> {
        self.samples
            .iter()
            .copied()
            .filter(|sample| sample.subject_id == Some(subject_id))
            .collect()
    }

    /// Rebuilds a stream from an existing sample list.
    #[must_use]
    pub fn from_samples(samples: Vec<TrackSample>) -> Self {
        let next_sample_id = samples
            .iter()
            .map(|s| s.sample_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
            .max(1);
        Self {
            samples,
            next_sample_id,
        }
    }

    /// Encodes all samples as Arrow-shaped `SLARROW1` bytes.
    ///
    /// # Errors
    ///
    /// Propagates encode failures.
    pub fn to_arrow_bytes(&self) -> Result<Vec<u8>, crate::MemoryError> {
        crate::encode_track_arrow(&self.samples)
    }

    /// Rebuilds a stream from Arrow-shaped `SLARROW1` bytes.
    ///
    /// # Errors
    ///
    /// Bad codec / truncated buffer.
    pub fn from_arrow_bytes(bytes: &[u8]) -> Result<Self, crate::MemoryError> {
        Ok(Self::from_samples(crate::decode_track_arrow(bytes)?))
    }
}
