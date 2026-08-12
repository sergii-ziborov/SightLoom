#![cfg_attr(not(feature = "std"), no_std)]
//! Subject references and identity-resolution contracts.
//!
//! Pipeline target (later implementations):
//! `Detection → TrackId → TrackFragment → embedding aggregation → SubjectId`.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use sightloom_core::{EmbeddingRef, EvidenceRef, MediaTime, SourceId, SubjectId, TrackId};

/// How a subject is described by reference material.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubjectModality {
    /// Face identity cues.
    Face,
    /// Full-body / clothing appearance.
    PersonAppearance,
    /// Vehicle appearance.
    VehicleAppearance,
    /// License plate text/appearance.
    LicensePlate,
    /// Generic object embedding.
    GenericObject,
}

/// One reference sample contributing to a subject identity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReferenceSample {
    /// Source of the sample when known.
    pub source_id: Option<SourceId>,
    /// Optional track that produced the sample.
    pub track_id: Option<TrackId>,
    /// When the sample was captured.
    pub at: Option<MediaTime>,
    /// Embedding handle (out-of-line vector storage).
    pub embedding: Option<EmbeddingRef>,
    /// Evidence crop / reel handle.
    pub evidence: Option<EvidenceRef>,
    /// Positive example when true; negative when false; unlabeled when `None`.
    pub is_positive: Option<bool>,
}

/// A subject identity with modality-specific reference samples.
#[derive(Clone, Debug, PartialEq)]
pub struct SubjectReference {
    /// Long-lived subject id.
    pub subject_id: SubjectId,
    /// Identity modality.
    pub modality: SubjectModality,
    /// Reference samples (positive, negative, or unlabeled).
    #[cfg(feature = "alloc")]
    pub samples: Vec<ReferenceSample>,
}

#[cfg(feature = "alloc")]
impl SubjectReference {
    /// Creates an empty reference set for a subject and modality.
    #[must_use]
    pub fn new(subject_id: SubjectId, modality: SubjectModality) -> Self {
        Self {
            subject_id,
            modality,
            samples: Vec::new(),
        }
    }

    /// Appends a reference sample.
    pub fn push_sample(&mut self, sample: ReferenceSample) {
        self.samples.push(sample);
    }
}

/// Confidence band for an identity match decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatchDecision {
    /// Accepted as the same subject.
    Accept,
    /// Rejected.
    Reject,
    /// Needs manual review / more evidence.
    Uncertain,
}

/// Result of comparing an embedding or track fragment to a subject.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdentityMatch {
    /// Candidate subject.
    pub subject_id: SubjectId,
    /// Similarity score (backend-defined scale).
    pub score: f32,
    /// Accept / reject / uncertain.
    pub decision: MatchDecision,
}

/// A contiguous track fragment used during identity aggregation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackFragment {
    /// Local track id.
    pub track_id: TrackId,
    /// Source.
    pub source_id: SourceId,
    /// Start time.
    pub start: MediaTime,
    /// End time.
    pub end: MediaTime,
    /// Aggregated embedding when available.
    pub embedding: Option<EmbeddingRef>,
    /// Linked subject when already resolved.
    pub subject_id: Option<SubjectId>,
}

/// Identity resolver contract (algorithms land in later milestones).
#[cfg(feature = "alloc")]
pub trait IdentityResolver {
    /// Propose matches for a track fragment against known subjects.
    fn resolve_fragment(
        &self,
        fragment: &TrackFragment,
        candidates: &[SubjectReference],
    ) -> Vec<IdentityMatch>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use sightloom_core::SubjectId;

    #[test]
    fn subject_reference_accepts_samples() {
        let mut reference = SubjectReference::new(SubjectId(7), SubjectModality::PersonAppearance);
        reference.push_sample(ReferenceSample {
            source_id: None,
            track_id: None,
            at: None,
            embedding: None,
            evidence: None,
            is_positive: Some(true),
        });
        assert_eq!(reference.samples.len(), 1);
        assert_eq!(reference.modality, SubjectModality::PersonAppearance);
    }
}
