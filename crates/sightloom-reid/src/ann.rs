//! Approximate nearest-neighbor (ANN) indexes for embedding search.
//!
//! Pure-Rust, dependency-free foundation:
//! - [`BruteForceAnn`] — exact cosine scan (default / correctness baseline)
//! - [`LshAnn`] — random-projection sign LSH + multiprobe candidate set + exact re-rank
//!
//! Not a FAISS/HNSW replacement; hosts may plug external ANN later via rebuild.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::collections::BTreeMap;
#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::{EmbeddingError, cosine_similarity};

/// One ANN hit: external id + cosine similarity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnnHit {
    /// Caller-defined id (e.g. `EmbeddingRef.0` or packed track key).
    pub id: u64,
    /// Cosine similarity in approximately `[-1.0, 1.0]`.
    pub score: f32,
}

/// Which ANN implementation to use when building an index.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AnnKind {
    /// Exact linear scan.
    #[default]
    BruteForce,
    /// Random-projection LSH (approximate).
    Lsh {
        /// Number of random projection bits (1..=32 recommended).
        bits: u8,
        /// Extra hash probes around the query hash (0 = exact bucket only).
        multiprobe: u8,
    },
}

/// Common operations over ANN backends.
pub trait AnnIndex {
    /// Inserts or replaces a vector for `id`.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError::InvalidVector`] on empty / non-finite / dim mismatch.
    fn upsert(&mut self, id: u64, vector: &[f32]) -> Result<(), EmbeddingError>;

    /// Removes `id` when present.
    fn remove(&mut self, id: u64);

    /// Top-k cosine hits (best first).
    ///
    /// # Errors
    ///
    /// Returns vector validation errors.
    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<AnnHit>, EmbeddingError>;

    /// Number of indexed vectors.
    fn len(&self) -> usize;

    /// True when empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drops all entries.
    fn clear(&mut self);
}

/// Exact cosine ANN (linear scan).
#[derive(Clone, Debug, Default)]
pub struct BruteForceAnn {
    dim: Option<usize>,
    items: Vec<(u64, Vec<f32>)>,
}

impl BruteForceAnn {
    /// Empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl AnnIndex for BruteForceAnn {
    fn upsert(&mut self, id: u64, vector: &[f32]) -> Result<(), EmbeddingError> {
        validate_and_dim(&mut self.dim, vector)?;
        if let Some(slot) = self.items.iter_mut().find(|(k, _)| *k == id) {
            slot.1 = vector.to_vec();
        } else {
            self.items.push((id, vector.to_vec()));
        }
        Ok(())
    }

    fn remove(&mut self, id: u64) {
        self.items.retain(|(k, _)| *k != id);
    }

    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<AnnHit>, EmbeddingError> {
        if query.is_empty() || query.iter().any(|v| !v.is_finite()) {
            return Err(EmbeddingError::InvalidVector);
        }
        if let Some(dim) = self.dim
            && query.len() != dim
            && !self.items.is_empty()
        {
            return Err(EmbeddingError::InvalidVector);
        }
        let mut hits = Vec::new();
        for (id, vec) in &self.items {
            if let Some(score) = cosine_similarity(query, vec) {
                hits.push(AnnHit { id: *id, score });
            }
        }
        sort_hits(&mut hits);
        if top_k == 0 || hits.len() <= top_k {
            Ok(hits)
        } else {
            Ok(hits[..top_k].to_vec())
        }
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn clear(&mut self) {
        self.items.clear();
        self.dim = None;
    }
}

/// Random-projection sign LSH with exact re-rank of candidates.
#[derive(Clone, Debug)]
pub struct LshAnn {
    dim: Option<usize>,
    bits: u8,
    multiprobe: u8,
    /// Deterministic projection rows (`bits` × `dim`).
    projections: Vec<f32>,
    buckets: BTreeMap<u32, Vec<u64>>,
    items: Vec<(u64, Vec<f32>)>,
}

impl LshAnn {
    /// Creates an LSH index with the given bit width and multiprobe radius.
    #[must_use]
    pub fn new(bits: u8, multiprobe: u8) -> Self {
        let bits = bits.clamp(1, 32);
        Self {
            dim: None,
            bits,
            multiprobe,
            projections: Vec::new(),
            buckets: BTreeMap::new(),
            items: Vec::new(),
        }
    }

    fn ensure_projections(&mut self, dim: usize) {
        if !self.projections.is_empty() {
            return;
        }
        let n = usize::from(self.bits) * dim;
        self.projections = Vec::with_capacity(n);
        // Deterministic pseudo-random projections (LCG) — no external RNG.
        let mut state = 0xA5A5_1234_u32.wrapping_add(dim as u32);
        for _ in 0..n {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // Map to (-1, 1)
            let unit = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            self.projections.push(unit);
        }
    }

    fn hash_of(&self, vector: &[f32]) -> u32 {
        let dim = vector.len();
        let bits = usize::from(self.bits);
        let mut hash = 0_u32;
        for b in 0..bits {
            let mut dot = 0.0_f32;
            let row = b * dim;
            for (i, &v) in vector.iter().enumerate() {
                dot += v * self.projections[row + i];
            }
            if dot >= 0.0 {
                hash |= 1_u32 << b;
            }
        }
        hash
    }

    fn probe_hashes(&self, center: u32) -> Vec<u32> {
        let mut out = vec![center];
        let bits = u32::from(self.bits);
        let probes = u32::from(self.multiprobe).min(bits);
        for i in 0..probes {
            out.push(center ^ (1_u32 << i));
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    fn reindex_buckets(&mut self) {
        self.buckets.clear();
        for (id, vec) in &self.items {
            let h = self.hash_of(vec);
            self.buckets.entry(h).or_default().push(*id);
        }
    }
}

impl AnnIndex for LshAnn {
    fn upsert(&mut self, id: u64, vector: &[f32]) -> Result<(), EmbeddingError> {
        validate_and_dim(&mut self.dim, vector)?;
        let dim = vector.len();
        self.ensure_projections(dim);
        if let Some(slot) = self.items.iter_mut().find(|(k, _)| *k == id) {
            slot.1 = vector.to_vec();
        } else {
            self.items.push((id, vector.to_vec()));
        }
        self.reindex_buckets();
        Ok(())
    }

    fn remove(&mut self, id: u64) {
        self.items.retain(|(k, _)| *k != id);
        self.reindex_buckets();
    }

    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<AnnHit>, EmbeddingError> {
        if query.is_empty() || query.iter().any(|v| !v.is_finite()) {
            return Err(EmbeddingError::InvalidVector);
        }
        if self.items.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(dim) = self.dim
            && query.len() != dim
        {
            return Err(EmbeddingError::InvalidVector);
        }
        // If projections not ready (empty index path), fall back to brute.
        if self.projections.is_empty() {
            return BruteForceAnn {
                dim: self.dim,
                items: self.items.clone(),
            }
            .search(query, top_k);
        }
        let center = self.hash_of(query);
        let probes = self.probe_hashes(center);
        let mut candidate_ids: Vec<u64> = Vec::new();
        for h in probes {
            if let Some(ids) = self.buckets.get(&h) {
                for id in ids {
                    if !candidate_ids.contains(id) {
                        candidate_ids.push(*id);
                    }
                }
            }
        }
        // Safety net: if LSH found nothing, scan all (approx quality floor).
        if candidate_ids.is_empty() {
            candidate_ids = self.items.iter().map(|(id, _)| *id).collect();
        }
        let mut hits = Vec::new();
        for id in candidate_ids {
            if let Some((_, vec)) = self.items.iter().find(|(k, _)| *k == id)
                && let Some(score) = cosine_similarity(query, vec)
            {
                hits.push(AnnHit { id, score });
            }
        }
        sort_hits(&mut hits);
        if top_k == 0 || hits.len() <= top_k {
            Ok(hits)
        } else {
            Ok(hits[..top_k].to_vec())
        }
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn clear(&mut self) {
        self.items.clear();
        self.buckets.clear();
        self.projections.clear();
        self.dim = None;
    }
}

/// Owned ANN backend selected by [`AnnKind`].
#[derive(Clone, Debug)]
pub enum AnnBackend {
    /// Exact scan.
    BruteForce(BruteForceAnn),
    /// LSH approximate.
    Lsh(LshAnn),
}

impl AnnBackend {
    /// Builds an empty backend of the requested kind.
    #[must_use]
    pub fn new(kind: AnnKind) -> Self {
        match kind {
            AnnKind::BruteForce => Self::BruteForce(BruteForceAnn::new()),
            AnnKind::Lsh { bits, multiprobe } => Self::Lsh(LshAnn::new(bits, multiprobe)),
        }
    }

    /// Rebuilds from `(id, vector)` pairs.
    ///
    /// # Errors
    ///
    /// Propagates upsert validation errors.
    pub fn rebuild_from<'a, I>(&mut self, items: I) -> Result<(), EmbeddingError>
    where
        I: IntoIterator<Item = (u64, &'a [f32])>,
    {
        self.clear();
        for (id, vec) in items {
            self.upsert(id, vec)?;
        }
        Ok(())
    }
}

impl AnnIndex for AnnBackend {
    fn upsert(&mut self, id: u64, vector: &[f32]) -> Result<(), EmbeddingError> {
        match self {
            Self::BruteForce(i) => i.upsert(id, vector),
            Self::Lsh(i) => i.upsert(id, vector),
        }
    }

    fn remove(&mut self, id: u64) {
        match self {
            Self::BruteForce(i) => i.remove(id),
            Self::Lsh(i) => i.remove(id),
        }
    }

    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<AnnHit>, EmbeddingError> {
        match self {
            Self::BruteForce(i) => i.search(query, top_k),
            Self::Lsh(i) => i.search(query, top_k),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::BruteForce(i) => i.len(),
            Self::Lsh(i) => i.len(),
        }
    }

    fn clear(&mut self) {
        match self {
            Self::BruteForce(i) => i.clear(),
            Self::Lsh(i) => i.clear(),
        }
    }
}

fn validate_and_dim(dim: &mut Option<usize>, vector: &[f32]) -> Result<(), EmbeddingError> {
    if vector.is_empty() || vector.iter().any(|v| !v.is_finite()) {
        return Err(EmbeddingError::InvalidVector);
    }
    match *dim {
        None => *dim = Some(vector.len()),
        Some(d) if d != vector.len() => return Err(EmbeddingError::InvalidVector),
        Some(_) => {}
    }
    Ok(())
}

fn sort_hits(hits: &mut [AnnHit]) {
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    #[test]
    fn brute_force_ranks_identical_first() {
        let mut ann = BruteForceAnn::new();
        ann.upsert(1, &[1.0, 0.0, 0.0]).unwrap();
        ann.upsert(2, &[0.0, 1.0, 0.0]).unwrap();
        ann.upsert(3, &[0.9, 0.1, 0.0]).unwrap();
        let hits = ann.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(hits[0].id, 1);
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn lsh_finds_near_neighbor() {
        let mut ann = LshAnn::new(16, 2);
        ann.upsert(10, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        ann.upsert(20, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        ann.upsert(30, &[0.95, 0.05, 0.0, 0.0]).unwrap();
        let hits = ann.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert!(!hits.is_empty());
        // Best should be exact or near match to axis-x.
        assert!(hits[0].id == 10 || hits[0].id == 30);
    }

    #[test]
    fn backend_rebuild() {
        let mut backend = AnnBackend::new(AnnKind::BruteForce);
        backend
            .rebuild_from([(1, &[1.0_f32, 0.0] as &[f32]), (2, &[0.0, 1.0])])
            .unwrap();
        assert_eq!(backend.len(), 2);
        let hits = backend.search(&[0.0, 1.0], 1).unwrap();
        assert_eq!(hits[0].id, 2);
    }
}
