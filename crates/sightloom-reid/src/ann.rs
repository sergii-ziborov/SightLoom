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
use alloc::collections::{BTreeMap, BTreeSet};
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
    /// Hierarchical Navigable Small World graph (pure Rust, not FAISS).
    Hnsw {
        /// Max neighbors per node on layers &gt; 0 (typical 8–32).
        m: u8,
        /// Candidate list size while inserting (typical 100–200).
        ef_construction: u16,
        /// Candidate list size while searching (typical 32–128).
        ef_search: u16,
    },
}

impl AnnKind {
    /// Default HNSW parameters suitable for medium galleries.
    #[must_use]
    pub const fn hnsw_default() -> Self {
        Self::Hnsw {
            m: 16,
            ef_construction: 100,
            ef_search: 64,
        }
    }
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

/// Hierarchical NSW graph ANN (cosine via distance `1 - sim`).
///
/// Simplified Malkov & Yashunin HNSW: multi-layer proximity graph, greedy
/// descent + ef-bounded search. Pure Rust — **not** a FAISS/HNSWlib binding.
#[derive(Clone, Debug)]
pub struct HnswAnn {
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    ef_search: usize,
    dim: Option<usize>,
    entry: Option<usize>,
    max_layer: usize,
    nodes: Vec<HnswNode>,
    id_to_idx: BTreeMap<u64, usize>,
}

#[derive(Clone, Debug)]
struct HnswNode {
    id: u64,
    vector: Vec<f32>,
    /// Neighbors per layer (index into `nodes`).
    neighbors: Vec<Vec<usize>>,
    /// Soft-deleted.
    dead: bool,
}

impl HnswAnn {
    /// Creates an empty HNSW index.
    #[must_use]
    pub fn new(m: u8, ef_construction: u16, ef_search: u16) -> Self {
        let m = usize::from(m).clamp(2, 64);
        Self {
            m,
            m_max0: m.saturating_mul(2),
            ef_construction: usize::from(ef_construction).max(m),
            ef_search: usize::from(ef_search).max(1),
            dim: None,
            entry: None,
            max_layer: 0,
            nodes: Vec::new(),
            id_to_idx: BTreeMap::new(),
        }
    }

    fn level_for(id: u64, m: usize) -> usize {
        // Deterministic geometric level from id hash (no RNG needed).
        let mut x = id
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0xBF58_476D_1CE4_E5B9);
        let mut level = 0_usize;
        // Approx P(level >= l) ≈ 1/m^l
        let modulus = m.max(2) as u64;
        while x.is_multiple_of(modulus) && level < 16 {
            level = level.saturating_add(1);
            x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        }
        level
    }

    fn dist(a: &[f32], b: &[f32]) -> f32 {
        // Lower is better.
        1.0 - cosine_similarity(a, b).unwrap_or(-1.0)
    }

    fn search_layer(
        &self,
        query: &[f32],
        entry: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f32)> {
        let mut visited = BTreeSet::new();
        visited.insert(entry);
        let d0 = Self::dist(query, &self.nodes[entry].vector);
        // candidates: min-heap by dist (manual sorted vec)
        let mut candidates = vec![(entry, d0)];
        let mut w = vec![(entry, d0)]; // nearest found, sorted ascending dist

        while let Some((c_idx, c_dist)) = candidates.first().copied() {
            candidates.remove(0);
            let f_dist = w.last().map_or(f32::MAX, |(_, d)| *d);
            if c_dist > f_dist && w.len() >= ef {
                break;
            }
            if layer >= self.nodes[c_idx].neighbors.len() {
                continue;
            }
            for &n in &self.nodes[c_idx].neighbors[layer] {
                if self.nodes[n].dead || visited.contains(&n) {
                    continue;
                }
                visited.insert(n);
                let d = Self::dist(query, &self.nodes[n].vector);
                let f_dist = w.last().map_or(f32::MAX, |(_, x)| *x);
                if d < f_dist || w.len() < ef {
                    insert_sorted_asc(&mut candidates, (n, d));
                    insert_sorted_asc(&mut w, (n, d));
                    if w.len() > ef {
                        w.pop();
                    }
                }
            }
        }
        w
    }

    fn select_neighbors(candidates: &[(usize, f32)], m: usize) -> Vec<usize> {
        candidates.iter().take(m).map(|(i, _)| *i).collect()
    }

    fn connect(&mut self, a: usize, b: usize, layer: usize, m_max: usize) {
        if a == b {
            return;
        }
        while self.nodes[a].neighbors.len() <= layer {
            self.nodes[a].neighbors.push(Vec::new());
        }
        while self.nodes[b].neighbors.len() <= layer {
            self.nodes[b].neighbors.push(Vec::new());
        }
        if !self.nodes[a].neighbors[layer].contains(&b) {
            self.nodes[a].neighbors[layer].push(b);
        }
        if !self.nodes[b].neighbors[layer].contains(&a) {
            self.nodes[b].neighbors[layer].push(a);
        }
        // Prune to m_max closest by distance to node vector.
        for idx in [a, b] {
            let neighbors = self.nodes[idx].neighbors[layer].clone();
            if neighbors.len() <= m_max {
                continue;
            }
            let v = self.nodes[idx].vector.clone();
            let mut scored: Vec<(usize, f32)> = neighbors
                .into_iter()
                .map(|n| (n, Self::dist(&v, &self.nodes[n].vector)))
                .collect();
            scored.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(core::cmp::Ordering::Equal));
            self.nodes[idx].neighbors[layer] =
                scored.into_iter().take(m_max).map(|(n, _)| n).collect();
        }
    }
}

impl AnnIndex for HnswAnn {
    fn upsert(&mut self, id: u64, vector: &[f32]) -> Result<(), EmbeddingError> {
        validate_and_dim(&mut self.dim, vector)?;
        if let Some(&idx) = self.id_to_idx.get(&id) {
            // In-place vector update (graph edges retained).
            self.nodes[idx].vector = vector.to_vec();
            self.nodes[idx].dead = false;
            return Ok(());
        }
        let level = Self::level_for(id, self.m);
        let idx = self.nodes.len();
        self.nodes.push(HnswNode {
            id,
            vector: vector.to_vec(),
            neighbors: (0..=level).map(|_| Vec::new()).collect(),
            dead: false,
        });
        self.id_to_idx.insert(id, idx);

        if self.entry.is_none() {
            self.entry = Some(idx);
            self.max_layer = level;
            return Ok(());
        }

        let mut ep = self.entry.unwrap();
        // Greedy search from top layer down to level+1.
        for layer in (level.saturating_add(1)..=self.max_layer).rev() {
            let nearest = self.search_layer(vector, ep, 1, layer);
            if let Some((n, _)) = nearest.first() {
                ep = *n;
            }
        }
        // Insert into layers level..0
        for layer in (0..=level).rev() {
            let candidates = self.search_layer(vector, ep, self.ef_construction, layer);
            let m_max = if layer == 0 { self.m_max0 } else { self.m };
            let selected = Self::select_neighbors(&candidates, self.m.min(m_max));
            for &n in &selected {
                self.connect(idx, n, layer, m_max);
            }
            if let Some((n, _)) = candidates.first() {
                ep = *n;
            }
        }
        if level > self.max_layer {
            self.max_layer = level;
            self.entry = Some(idx);
        }
        Ok(())
    }

    fn remove(&mut self, id: u64) {
        if let Some(&idx) = self.id_to_idx.get(&id) {
            self.nodes[idx].dead = true;
            self.id_to_idx.remove(&id);
            if self.entry == Some(idx) {
                self.entry = self
                    .nodes
                    .iter()
                    .enumerate()
                    .find(|(_, n)| !n.dead)
                    .map(|(i, _)| i);
            }
        }
    }

    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<AnnHit>, EmbeddingError> {
        if query.is_empty() || query.iter().any(|v| !v.is_finite()) {
            return Err(EmbeddingError::InvalidVector);
        }
        if self.nodes.is_empty() || self.entry.is_none() {
            return Ok(Vec::new());
        }
        if let Some(dim) = self.dim
            && query.len() != dim
        {
            return Err(EmbeddingError::InvalidVector);
        }
        let mut ep = self.entry.unwrap();
        // If entry is dead, fall back to brute.
        if self.nodes[ep].dead {
            return Ok(brute_from_hnsw_nodes(&self.nodes, query, top_k));
        }
        for layer in (1..=self.max_layer).rev() {
            let nearest = self.search_layer(query, ep, 1, layer);
            if let Some((n, _)) = nearest.first() {
                ep = *n;
            }
        }
        let found = self.search_layer(query, ep, self.ef_search.max(top_k.max(1)), 0);
        let mut hits: Vec<AnnHit> = found
            .into_iter()
            .filter(|(i, _)| !self.nodes[*i].dead)
            .map(|(i, dist)| AnnHit {
                id: self.nodes[i].id,
                score: 1.0 - dist,
            })
            .collect();
        sort_hits(&mut hits);
        if top_k == 0 || hits.len() <= top_k {
            Ok(hits)
        } else {
            Ok(hits[..top_k].to_vec())
        }
    }

    fn len(&self) -> usize {
        self.id_to_idx.len()
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.id_to_idx.clear();
        self.entry = None;
        self.max_layer = 0;
        self.dim = None;
    }
}

fn insert_sorted_asc(list: &mut Vec<(usize, f32)>, item: (usize, f32)) {
    let pos = list
        .iter()
        .position(|(_, d)| item.1 < *d)
        .unwrap_or(list.len());
    list.insert(pos, item);
}

fn brute_from_hnsw_nodes(nodes: &[HnswNode], query: &[f32], top_k: usize) -> Vec<AnnHit> {
    let mut hits = Vec::new();
    for n in nodes {
        if n.dead {
            continue;
        }
        if let Some(score) = cosine_similarity(query, &n.vector) {
            hits.push(AnnHit { id: n.id, score });
        }
    }
    sort_hits(&mut hits);
    if top_k == 0 || hits.len() <= top_k {
        hits
    } else {
        hits[..top_k].to_vec()
    }
}

/// Host-supplied ANN (e.g. FAISS via FFI). `SightLoom` does not link FAISS.
///
/// Hosts implement this trait and call [`search_with_host_ann`] instead of
/// owning FAISS inside the library.
pub trait HostAnnAdapter {
    /// Approximate top-k search.
    ///
    /// # Errors
    ///
    /// Host-defined failures mapped to [`EmbeddingError`].
    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<AnnHit>, EmbeddingError>;
}

/// Runs search through a host FAISS/HNSWlib/etc. adapter.
///
/// # Errors
///
/// Propagates adapter errors.
pub fn search_with_host_ann(
    adapter: &dyn HostAnnAdapter,
    query: &[f32],
    top_k: usize,
) -> Result<Vec<AnnHit>, EmbeddingError> {
    adapter.search(query, top_k)
}

/// Owned ANN backend selected by [`AnnKind`].
#[derive(Clone, Debug)]
pub enum AnnBackend {
    /// Exact scan.
    BruteForce(BruteForceAnn),
    /// LSH approximate.
    Lsh(LshAnn),
    /// Hierarchical NSW graph.
    Hnsw(HnswAnn),
}

impl AnnBackend {
    /// Builds an empty backend of the requested kind.
    #[must_use]
    pub fn new(kind: AnnKind) -> Self {
        match kind {
            AnnKind::BruteForce => Self::BruteForce(BruteForceAnn::new()),
            AnnKind::Lsh { bits, multiprobe } => Self::Lsh(LshAnn::new(bits, multiprobe)),
            AnnKind::Hnsw {
                m,
                ef_construction,
                ef_search,
            } => Self::Hnsw(HnswAnn::new(m, ef_construction, ef_search)),
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
            Self::Hnsw(i) => i.upsert(id, vector),
        }
    }

    fn remove(&mut self, id: u64) {
        match self {
            Self::BruteForce(i) => i.remove(id),
            Self::Lsh(i) => i.remove(id),
            Self::Hnsw(i) => i.remove(id),
        }
    }

    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<AnnHit>, EmbeddingError> {
        match self {
            Self::BruteForce(i) => i.search(query, top_k),
            Self::Lsh(i) => i.search(query, top_k),
            Self::Hnsw(i) => i.search(query, top_k),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::BruteForce(i) => i.len(),
            Self::Lsh(i) => i.len(),
            Self::Hnsw(i) => i.len(),
        }
    }

    fn clear(&mut self) {
        match self {
            Self::BruteForce(i) => i.clear(),
            Self::Lsh(i) => i.clear(),
            Self::Hnsw(i) => i.clear(),
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

    #[test]
    fn hnsw_finds_nearest_among_many() {
        let mut ann = HnswAnn::new(8, 50, 32);
        for i in 0..40_u64 {
            let angle = (i as f32) * 0.15;
            let v = [angle.cos(), angle.sin(), 0.0, 0.0];
            ann.upsert(i, &v).unwrap();
        }
        // Query near i=0 → [1,0,0,0]
        let hits = ann.search(&[1.0, 0.0, 0.0, 0.0], 3).unwrap();
        assert!(!hits.is_empty());
        assert!(hits[0].score > 0.9, "score={}", hits[0].score);
        // Best id should be small (near axis-x).
        assert!(hits[0].id < 5, "id={}", hits[0].id);
    }
}
