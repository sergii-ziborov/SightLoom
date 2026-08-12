//! Compact track sample records for sidecar streams.

use sightloom_core::{ClassId, MediaTime, SourceId, SubjectId, TrackId};

/// One track sample suitable for CBOR/Arrow streaming later.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackSample {
    /// Source camera or file.
    pub source_id: SourceId,
    /// Frame index within the source.
    pub frame_index: u64,
    /// Presentation timestamp.
    pub pts: MediaTime,
    /// Track id.
    pub track_id: TrackId,
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

/// In-memory append-only track stream (host conveniences).
#[cfg(feature = "std")]
#[derive(Clone, Debug, Default)]
pub struct TrackStream {
    samples: Vec<TrackSample>,
}

#[cfg(feature = "std")]
impl TrackStream {
    /// Creates an empty stream.
    #[must_use]
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    /// Appends a sample.
    pub fn push(&mut self, sample: TrackSample) {
        self.samples.push(sample);
    }

    /// Returns all samples.
    #[must_use]
    pub fn samples(&self) -> &[TrackSample] {
        &self.samples
    }

    /// Filters samples for a track id.
    #[must_use]
    pub fn for_track(&self, track_id: TrackId) -> Vec<TrackSample> {
        self.samples
            .iter()
            .copied()
            .filter(|sample| sample.track_id == track_id)
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
}
