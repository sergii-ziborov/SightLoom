//! Multi-factor identity scoring beyond raw cosine similarity.
//!
//! Fused score approximates:
//! ```text
//! embedding_similarity
//!   × embedding_quality
//!   × temporal_plausibility
//!   × camera_topology
//!   × class_compatibility
//!   × prior_identity_confidence
//! ```
//!
//! ANN backends, ROC/EER calibration, and retention policy remain host-side
//! or later milestones; this module defines the score contract and pure fusion.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use sightloom_core::{ClassId, MediaTime, SourceId};

/// Per-factor identity evidence used for accept / reject / uncertain decisions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdentityScoreFactors {
    /// Cosine (or backend) similarity, typically in `[-1.0, 1.0]`.
    pub embedding_similarity: f32,
    /// Embedding quality in `[0.0, 1.0]` (blur, size, detector conf proxy).
    pub embedding_quality: f32,
    /// Temporal plausibility in `[0.0, 1.0]` given last sighting time.
    pub temporal_plausibility: f32,
    /// Camera topology factor in `[0.0, 1.0]` (0 = physically impossible).
    pub camera_topology: f32,
    /// Class compatibility in `[0.0, 1.0]`.
    pub class_compatibility: f32,
    /// Prior identity confidence in `[0.0, 1.0]`.
    pub prior_identity_confidence: f32,
}

impl Default for IdentityScoreFactors {
    fn default() -> Self {
        Self {
            embedding_similarity: 0.0,
            embedding_quality: 1.0,
            temporal_plausibility: 1.0,
            camera_topology: 1.0,
            class_compatibility: 1.0,
            prior_identity_confidence: 1.0,
        }
    }
}

impl IdentityScoreFactors {
    /// Multiplies finite factors clamped to non-negative contribution.
    ///
    /// Similarity is mapped from `[-1, 1]` to `[0, 1]` before fusion so a
    /// negative cosine cannot be rescued by other unit factors alone.
    #[must_use]
    pub fn fused(self) -> f32 {
        let sim = ((self.embedding_similarity + 1.0) * 0.5).clamp(0.0, 1.0);
        let parts = [
            sim,
            self.embedding_quality.clamp(0.0, 1.0),
            self.temporal_plausibility.clamp(0.0, 1.0),
            self.camera_topology.clamp(0.0, 1.0),
            self.class_compatibility.clamp(0.0, 1.0),
            self.prior_identity_confidence.clamp(0.0, 1.0),
        ];
        if parts.iter().any(|v| !v.is_finite()) {
            return 0.0;
        }
        parts.iter().product()
    }
}

/// Optional context that modulates cosine scores before thresholding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreContext {
    /// Source of the query fragment.
    pub query_source: SourceId,
    /// Query time.
    pub query_at: MediaTime,
    /// Embedding quality for the query vector.
    pub embedding_quality: f32,
    /// Prior confidence that the track already has a correct subject.
    pub prior_identity_confidence: f32,
    /// Optional class of the query observation.
    pub class_id: Option<ClassId>,
}

impl ScoreContext {
    /// Default context with unit quality/prior and zero class.
    #[must_use]
    pub const fn new(query_source: SourceId, query_at: MediaTime) -> Self {
        Self {
            query_source,
            query_at,
            embedding_quality: 1.0,
            prior_identity_confidence: 1.0,
            class_id: None,
        }
    }
}

/// One directed travel constraint between cameras.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CameraEdge {
    /// Source camera.
    pub from: SourceId,
    /// Destination camera.
    pub to: SourceId,
    /// Minimum physically plausible travel time in nanoseconds.
    pub min_travel_ns: i64,
}

/// Sparse camera topology for cross-camera identity gating.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CameraTopology {
    edges: Vec<CameraEdge>,
}

impl CameraTopology {
    /// Empty topology (same-source always allowed; cross-source unconstrained).
    #[must_use]
    pub fn new() -> Self {
        Self { edges: Vec::new() }
    }

    /// Adds or replaces a directed edge.
    pub fn set_edge(&mut self, from: SourceId, to: SourceId, min_travel_ns: i64) {
        if let Some(edge) = self
            .edges
            .iter_mut()
            .find(|e| e.from == from && e.to == to)
        {
            edge.min_travel_ns = min_travel_ns;
        } else {
            self.edges.push(CameraEdge {
                from,
                to,
                min_travel_ns,
            });
        }
    }

    /// Topology factor in `[0.0, 1.0]`.
    ///
    /// - Same source → `1.0`
    /// - No edge and `strict` → `0.0` (unknown hop impossible)
    /// - No edge and not strict → `1.0` (open world)
    /// - Edge present: `0.0` if elapsed &lt; min travel, else `1.0`
    #[must_use]
    pub fn factor(
        &self,
        from: SourceId,
        to: SourceId,
        elapsed_ns: i64,
        strict_unknown: bool,
    ) -> f32 {
        if from == to {
            return 1.0;
        }
        match self
            .edges
            .iter()
            .find(|e| e.from == from && e.to == to)
            .map(|e| e.min_travel_ns)
        {
            Some(min_travel) if elapsed_ns < min_travel => 0.0,
            None if strict_unknown => 0.0,
            Some(_) | None => 1.0,
        }
    }

    /// Edges in insertion order.
    #[must_use]
    pub fn edges(&self) -> &[CameraEdge] {
        &self.edges
    }
}

/// Temporal plausibility from last-seen time on a subject.
///
/// Returns `1.0` when no prior sighting exists. When `max_gap_ns` is set and
/// the gap exceeds it, returns `0.0`; otherwise `1.0`.
#[must_use]
pub fn temporal_plausibility(
    last_seen: Option<MediaTime>,
    now: MediaTime,
    max_gap_ns: Option<i64>,
) -> f32 {
    let Some(last) = last_seen else {
        return 1.0;
    };
    let gap = now.duration_since_ns(last).unsigned_abs();
    let gap = i64::try_from(gap).unwrap_or(i64::MAX);
    match max_gap_ns {
        Some(max) if gap > max => 0.0,
        _ => 1.0,
    }
}

/// Class compatibility: matching classes or either missing → 1.0; mismatch → 0.0.
#[must_use]
pub fn class_compatibility(query: Option<ClassId>, reference: Option<ClassId>) -> f32 {
    match (query, reference) {
        (Some(a), Some(b)) if a != b => 0.0,
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sightloom_core::SourceId;

    #[test]
    fn topology_blocks_impossible_hop() {
        let mut topo = CameraTopology::new();
        topo.set_edge(SourceId(1), SourceId(2), 60_000_000_000);
        assert!((topo.factor(SourceId(1), SourceId(2), 2_000_000_000, true) - 0.0).abs() < 1e-6);
        assert!((topo.factor(SourceId(1), SourceId(2), 120_000_000_000, true) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fused_score_zero_when_topology_impossible() {
        let factors = IdentityScoreFactors {
            embedding_similarity: 0.99,
            embedding_quality: 1.0,
            temporal_plausibility: 1.0,
            camera_topology: 0.0,
            class_compatibility: 1.0,
            prior_identity_confidence: 1.0,
        };
        assert!((factors.fused() - 0.0).abs() < 1e-6);
    }
}
