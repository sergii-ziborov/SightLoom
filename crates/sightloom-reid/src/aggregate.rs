//! Track fragment aggregation from embedding samples.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};

use crate::{EmbeddingError, EmbeddingStore, SubjectModality, TrackFragment, mean_pool};
use sightloom_core::{EmbeddingRef, MediaTime, SourceId, SubjectId, TrackId};

/// One embedding observation on a track used for aggregation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmbeddingObservation {
    /// Embedding handle in a store.
    pub embedding: EmbeddingRef,
    /// Observation time.
    pub at: MediaTime,
}

/// Aggregates ordered embedding observations into a track fragment.
///
/// Mean-pools the vectors into a new embedding in `store`.
///
/// # Errors
///
/// Returns embedding store or pooling errors when inputs are empty/invalid.
///
/// # Panics
///
/// Does not panic for valid non-empty `observations`; empty input is an error.
#[cfg(feature = "alloc")]
pub fn aggregate_fragment(
    store: &mut EmbeddingStore,
    track_id: TrackId,
    source_id: SourceId,
    modality: SubjectModality,
    observations: &[EmbeddingObservation],
    subject_id: Option<SubjectId>,
) -> Result<TrackFragment, EmbeddingError> {
    if observations.is_empty() {
        return Err(EmbeddingError::InvalidVector);
    }
    let mut vectors = Vec::with_capacity(observations.len());
    for observation in observations {
        vectors.push(store.get(observation.embedding)?);
    }
    let dim = vectors[0].len();
    let mut mean = vec![0.0_f32; dim];
    mean_pool(&vectors, &mut mean)?;
    let embedding = store.insert(mean)?;
    let start = observations[0].at;
    let end = observations[observations.len() - 1].at;
    Ok(TrackFragment {
        track_id,
        source_id,
        start,
        end,
        embedding: Some(embedding),
        subject_id,
        modality,
        embedding_quality: 1.0,
        class_id: None,
    })
}
