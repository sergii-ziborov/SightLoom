//! In-memory embedding storage and cosine similarity helpers.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
use sightloom_core::EmbeddingRef;

/// Errors produced by embedding operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddingError {
    /// Empty or mismatched vector dimensions.
    InvalidVector,
    /// Handle was not found in the store.
    NotFound,
    /// Caller-owned output buffer is too small.
    InsufficientCapacity,
}

/// Host-owned store mapping [`EmbeddingRef`] handles to dense vectors.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug, Default)]
pub struct EmbeddingStore {
    next_id: u64,
    entries: Vec<(EmbeddingRef, Vec<f32>)>,
    /// Model identity for version separation across galleries.
    model: Option<EmbeddingModelId>,
}

/// Identity of the embedding model that produced vectors in a store.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EmbeddingModelId {
    /// Model name or URI.
    pub name: alloc::string::String,
    /// Model version or digest.
    pub version: alloc::string::String,
}

#[cfg(feature = "alloc")]
impl EmbeddingStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: 1,
            entries: Vec::new(),
            model: None,
        }
    }

    /// Sets the embedding model identity for this store.
    pub fn set_model(&mut self, model: EmbeddingModelId) {
        self.model = Some(model);
    }

    /// Returns the configured model identity when present.
    #[must_use]
    pub fn model(&self) -> Option<&EmbeddingModelId> {
        self.model.as_ref()
    }

    /// Inserts a vector and returns a new handle.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError::InvalidVector`] when the vector is empty or
    /// contains a non-finite component.
    pub fn insert(&mut self, vector: impl Into<Vec<f32>>) -> Result<EmbeddingRef, EmbeddingError> {
        let vector = vector.into();
        validate_vector(&vector)?;
        let handle = EmbeddingRef(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        self.entries.push((handle, vector));
        Ok(handle)
    }

    /// Looks up a vector by handle.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError::NotFound`] when the handle is unknown.
    pub fn get(&self, handle: EmbeddingRef) -> Result<&[f32], EmbeddingError> {
        self.entries
            .iter()
            .find(|(key, _)| *key == handle)
            .map(|(_, vector)| vector.as_slice())
            .ok_or(EmbeddingError::NotFound)
    }

    /// Returns all stored entries.
    #[must_use]
    pub fn entries(&self) -> &[(EmbeddingRef, Vec<f32>)] {
        &self.entries
    }

    /// Next handle counter (for checkpoints).
    #[must_use]
    pub const fn next_id(&self) -> u64 {
        self.next_id
    }

    /// Restores store contents from a checkpoint payload.
    pub fn restore_from(&mut self, next_id: u64, entries: Vec<(EmbeddingRef, Vec<f32>)>) {
        self.next_id = next_id.max(1);
        self.entries = entries;
    }

    /// Exact top-k cosine search over all stored vectors.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError::InvalidVector`] when the query is invalid.
    pub fn search_top_k(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<(EmbeddingRef, f32)>, EmbeddingError> {
        use crate::ann::{AnnIndex, BruteForceAnn};
        let mut ann = BruteForceAnn::new();
        for (handle, vector) in &self.entries {
            ann.upsert(handle.0, vector)?;
        }
        let hits = ann.search(query, top_k)?;
        Ok(hits
            .into_iter()
            .map(|h| (EmbeddingRef(h.id), h.score))
            .collect())
    }

    /// Builds an ANN backend of `kind` over all store entries.
    ///
    /// # Errors
    ///
    /// Propagates vector validation errors during rebuild.
    pub fn build_ann(&self, kind: crate::AnnKind) -> Result<crate::AnnBackend, EmbeddingError> {
        let mut backend = crate::AnnBackend::new(kind);
        backend.rebuild_from(self.entries.iter().map(|(h, v)| (h.0, v.as_slice())))?;
        Ok(backend)
    }
}

/// Cosine similarity in `[-1.0, 1.0]`.
///
/// Returns `None` when either vector is empty, lengths differ, or a norm is zero.
#[must_use]
pub fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.is_empty() || left.len() != right.len() {
        return None;
    }
    let mut dot = 0.0_f32;
    let mut left_norm = 0.0_f32;
    let mut right_norm = 0.0_f32;
    for (a, b) in left.iter().zip(right.iter()) {
        if !a.is_finite() || !b.is_finite() {
            return None;
        }
        dot += *a * *b;
        left_norm += *a * *a;
        right_norm += *b * *b;
    }
    if left_norm <= 0.0 || right_norm <= 0.0 {
        return None;
    }
    // Avoid libm: score = dot / (sqrt(l) * sqrt(r)).
    let denom = sqrt_approx(left_norm) * sqrt_approx(right_norm);
    if denom <= 0.0 {
        return None;
    }
    Some(dot / denom)
}

/// Mean-pools finite equal-dimension vectors into `output`.
///
/// # Errors
///
/// Returns dimension or capacity errors.
pub fn mean_pool(vectors: &[&[f32]], output: &mut [f32]) -> Result<(), EmbeddingError> {
    if vectors.is_empty() {
        return Err(EmbeddingError::InvalidVector);
    }
    let dim = vectors[0].len();
    if dim == 0 || output.len() < dim {
        return Err(EmbeddingError::InsufficientCapacity);
    }
    for vector in vectors {
        if vector.len() != dim {
            return Err(EmbeddingError::InvalidVector);
        }
        for &value in *vector {
            if !value.is_finite() {
                return Err(EmbeddingError::InvalidVector);
            }
        }
    }
    for slot in output.iter_mut().take(dim) {
        *slot = 0.0;
    }
    let count = vectors.len() as f32;
    for vector in vectors {
        for (index, value) in vector.iter().enumerate() {
            output[index] += *value / count;
        }
    }
    Ok(())
}

fn validate_vector(vector: &[f32]) -> Result<(), EmbeddingError> {
    if vector.is_empty() {
        return Err(EmbeddingError::InvalidVector);
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(EmbeddingError::InvalidVector);
    }
    Ok(())
}

/// Newton square-root approximation for portable cosine.
fn sqrt_approx(value: f32) -> f32 {
    if value <= 0.0 || !value.is_finite() {
        return 0.0;
    }
    let mut y = value;
    for _ in 0..8 {
        y = 0.5 * (y + value / y);
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_have_unit_cosine() {
        let a = [1.0_f32, 0.0, 0.0];
        let b = [2.0_f32, 0.0, 0.0];
        let score = cosine_similarity(&a, &b).unwrap();
        assert!((score - 1.0).abs() < 1e-3);
    }

    #[test]
    fn orthogonal_vectors_are_near_zero() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        let score = cosine_similarity(&a, &b).unwrap();
        assert!(score.abs() < 1e-3);
    }
}
