//! Reference-photo / embedding search against a subject gallery.
//!
//! Hosts convert photos to embedding vectors externally; `SightLoom` only ranks
//! gallery subjects (and optional track fragments) by multi-factor identity score.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::collections::BTreeMap;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::{
    CameraTopology, EmbeddingError, EmbeddingStore, IdentityMatch, IdentityResolver,
    IdentityScoreFactors, MatchDecision, ResolveConfig, ScoreContext, SubjectLastSeen,
    SubjectModality, SubjectReference, ThresholdResolver, TrackFragment, cosine_similarity,
};
use sightloom_core::{ClassId, EmbeddingRef, MediaTime, SourceId, SubjectId, TrackId};

/// One ranked hit from a photo / embedding search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhotoSearchHit {
    /// Matched subject when scoring the gallery.
    pub subject_id: SubjectId,
    /// Fused score.
    pub score: f32,
    /// Accept / reject / uncertain.
    pub decision: MatchDecision,
    /// Factor breakdown.
    pub factors: IdentityScoreFactors,
}

/// Search query produced from one or more reference photos (embeddings).
#[derive(Clone, Debug, PartialEq)]
pub struct PhotoQuery {
    /// Query embedding handle in the gallery store.
    pub embedding: EmbeddingRef,
    /// Quality of the photo embedding (`1.0` if unknown).
    pub quality: f32,
    /// Modality used for matching.
    pub modality: SubjectModality,
    /// Optional class filter.
    pub class_id: Option<ClassId>,
    /// Logical query source (for topology; often a synthetic source).
    pub source_id: SourceId,
    /// Query time (for temporal factors).
    pub at: MediaTime,
}

impl PhotoQuery {
    /// Builds a photo query with unit quality and default source `0`.
    #[must_use]
    pub fn new(embedding: EmbeddingRef, modality: SubjectModality, at: MediaTime) -> Self {
        Self {
            embedding,
            quality: 1.0,
            modality,
            class_id: None,
            source_id: SourceId(0),
            at,
        }
    }
}

/// Scores a photo embedding against gallery subjects (linear scan, multi-factor).
///
/// # Errors
///
/// Returns store / config errors.
#[cfg(feature = "alloc")]
#[allow(clippy::too_many_arguments)]
pub fn search_gallery_by_photo(
    store: &EmbeddingStore,
    subjects: &[SubjectReference],
    config: ResolveConfig,
    query: &PhotoQuery,
    topology: Option<&CameraTopology>,
    last_seen: Option<&SubjectLastSeen>,
    source_accept: Option<&BTreeMap<u32, f32>>,
    top_k: usize,
) -> Result<Vec<PhotoSearchHit>, EmbeddingError> {
    let fragment = TrackFragment {
        track_id: TrackId(0),
        source_id: query.source_id,
        start: query.at,
        end: query.at,
        embedding: Some(query.embedding),
        subject_id: None,
        modality: query.modality,
        embedding_quality: query.quality,
        class_id: query.class_id,
    };
    let ctx = ScoreContext {
        query_source: query.source_id,
        query_at: query.at,
        embedding_quality: query.quality,
        prior_identity_confidence: 1.0,
        class_id: query.class_id,
    };
    let resolver =
        ThresholdResolver::with_context(store, config, ctx, topology, last_seen, source_accept)?;
    let matches = resolver.resolve_fragment(&fragment, subjects);
    let hits: Vec<PhotoSearchHit> = matches
        .into_iter()
        .map(|m: IdentityMatch| PhotoSearchHit {
            subject_id: m.subject_id,
            score: m.score,
            decision: m.decision,
            factors: m.factors,
        })
        .collect();
    if top_k == 0 || hits.len() <= top_k {
        Ok(hits)
    } else {
        Ok(hits[..top_k].to_vec())
    }
}

/// Direct cosine ranking of a query vector against all positive reference samples.
///
/// Useful when multi-factor context is not yet available; still reports subject ids.
///
/// # Errors
///
/// Returns store lookup errors when the query handle is unknown.
#[cfg(feature = "alloc")]
pub fn rank_subjects_by_cosine(
    store: &EmbeddingStore,
    subjects: &[SubjectReference],
    query: EmbeddingRef,
    top_k: usize,
) -> Result<Vec<(SubjectId, f32)>, EmbeddingError> {
    let q = store.get(query)?;
    let mut ranks: Vec<(SubjectId, f32)> = Vec::new();
    for subject in subjects {
        let mut best: Option<f32> = None;
        for sample in subject.positives() {
            let Some(handle) = sample.embedding else {
                continue;
            };
            let Ok(vector) = store.get(handle) else {
                continue;
            };
            let Some(sim) = cosine_similarity(q, vector) else {
                continue;
            };
            best = Some(best.map_or(sim, |b| b.max(sim)));
        }
        if let Some(sim) = best {
            ranks.push((subject.subject_id, sim));
        }
    }
    ranks.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| a.0.0.cmp(&b.0.0))
    });
    if top_k == 0 || ranks.len() <= top_k {
        Ok(ranks)
    } else {
        Ok(ranks[..top_k].to_vec())
    }
}
