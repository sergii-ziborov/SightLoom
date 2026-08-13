//! Evidence reel builders — structured handles only (no pixels / no video decode).
//!
//! A reel is a queryable timeline of references a host can use to assemble
//! clips, privacy previews, or audit UIs. Pixel crops live behind
//! [`EvidenceRef`] / mask handles owned by the host or package.

use crate::{TrackSample, VisionIndex};
use sightloom_core::{EvidenceRef, MediaTime, SourceId, SubjectId, TrackId, TrackUid};

/// Opaque reel identifier within one index / session.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ReelId(pub u64);

/// One segment inside an evidence reel (still no pixels).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReelSegment {
    /// Source for this segment.
    pub source_id: SourceId,
    /// Local track when known.
    pub track_id: Option<TrackId>,
    /// Global track uid when known.
    pub track_uid: Option<TrackUid>,
    /// Inclusive start.
    pub start: MediaTime,
    /// Inclusive end.
    pub end: MediaTime,
    /// Optional mask handle (`0` = none).
    pub mask_ref: u64,
    /// Optional host evidence blob (crop / thumb).
    pub evidence: Option<EvidenceRef>,
    /// Peak confidence in the segment when derived from tracks.
    pub peak_confidence: f32,
    /// Track sample id when derived from a single sample.
    pub sample_id: Option<u64>,
}

/// Evidence reel: ordered segments for one subject (or multi-subject host set).
#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceReel {
    /// Reel id.
    pub reel_id: ReelId,
    /// Primary subject (None for mixed host-built reels).
    pub subject_id: Option<SubjectId>,
    /// Ordered segments.
    pub segments: Vec<ReelSegment>,
    /// Free-form host tag.
    pub tag: u32,
}

impl EvidenceReel {
    /// Total media span from first start to last end (nanoseconds), if any.
    #[must_use]
    pub fn span_ns(&self) -> Option<i64> {
        let first = self.segments.first()?;
        let last = self.segments.last()?;
        Some(last.end.as_nanos().saturating_sub(first.start.as_nanos()))
    }

    /// Number of segments.
    #[must_use]
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// True when empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}

/// Builds reels from index data.
#[derive(Clone, Debug, Default)]
pub struct EvidenceReelBuilder {
    next_id: u64,
}

impl EvidenceReelBuilder {
    /// Creates a builder starting ids at 1.
    #[must_use]
    pub fn new() -> Self {
        Self { next_id: 1 }
    }

    fn alloc_id(&mut self) -> ReelId {
        let id = ReelId(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        id
    }

    /// One segment per effective track sample for `subject_id` (chronological).
    #[must_use]
    pub fn from_subject_samples(
        &mut self,
        index: &VisionIndex,
        subject_id: SubjectId,
        tag: u32,
    ) -> EvidenceReel {
        let mut samples: Vec<TrackSample> = index
            .tracks
            .effective_samples()
            .into_iter()
            .filter(|s| s.subject_id == Some(subject_id))
            .collect();
        samples.sort_by(|a, b| {
            a.pts
                .as_nanos()
                .cmp(&b.pts.as_nanos())
                .then_with(|| a.frame_index.cmp(&b.frame_index))
        });
        let segments = samples
            .into_iter()
            .map(|s| ReelSegment {
                source_id: s.source_id,
                track_id: Some(s.track_id),
                track_uid: s.track_uid,
                start: s.pts,
                end: s.pts,
                mask_ref: s.mask_ref,
                evidence: None,
                peak_confidence: s.confidence,
                sample_id: Some(s.sample_id),
            })
            .collect();
        EvidenceReel {
            reel_id: self.alloc_id(),
            subject_id: Some(subject_id),
            segments,
            tag,
        }
    }

    /// Coalesces consecutive samples on the same track key within `gap_ns`.
    #[must_use]
    pub fn from_subject_coalesced(
        &mut self,
        index: &VisionIndex,
        subject_id: SubjectId,
        max_gap_ns: i64,
        tag: u32,
    ) -> EvidenceReel {
        let mut samples: Vec<TrackSample> = index
            .tracks
            .effective_samples()
            .into_iter()
            .filter(|s| s.subject_id == Some(subject_id))
            .collect();
        samples.sort_by(|a, b| {
            a.source_id
                .0
                .cmp(&b.source_id.0)
                .then_with(|| a.track_id.0.cmp(&b.track_id.0))
                .then_with(|| a.pts.as_nanos().cmp(&b.pts.as_nanos()))
        });

        let mut segments: Vec<ReelSegment> = Vec::new();
        for sample in samples {
            if let Some(last) = segments.last_mut() {
                let same_track = last.source_id == sample.source_id
                    && last.track_id == Some(sample.track_id);
                let gap = sample
                    .pts
                    .as_nanos()
                    .saturating_sub(last.end.as_nanos());
                if same_track && gap <= max_gap_ns {
                    last.end = sample.pts;
                    last.peak_confidence = last.peak_confidence.max(sample.confidence);
                    if sample.mask_ref != 0 {
                        last.mask_ref = sample.mask_ref;
                    }
                    continue;
                }
            }
            segments.push(ReelSegment {
                source_id: sample.source_id,
                track_id: Some(sample.track_id),
                track_uid: sample.track_uid,
                start: sample.pts,
                end: sample.pts,
                mask_ref: sample.mask_ref,
                evidence: None,
                peak_confidence: sample.confidence,
                sample_id: Some(sample.sample_id),
            });
        }
        EvidenceReel {
            reel_id: self.alloc_id(),
            subject_id: Some(subject_id),
            segments,
            tag,
        }
    }

    /// Builds one reel from an explicit segment list (host / Intelligence).
    #[must_use]
    pub fn from_segments(
        &mut self,
        subject_id: Option<SubjectId>,
        segments: Vec<ReelSegment>,
        tag: u32,
    ) -> EvidenceReel {
        EvidenceReel {
            reel_id: self.alloc_id(),
            subject_id,
            segments,
            tag,
        }
    }
}

/// Convenience: build a coalesced reel for a subject on an index.
#[must_use]
pub fn build_subject_reel(
    index: &VisionIndex,
    subject_id: SubjectId,
    max_gap_ns: i64,
) -> EvidenceReel {
    EvidenceReelBuilder::new().from_subject_coalesced(index, subject_id, max_gap_ns, 0)
}
