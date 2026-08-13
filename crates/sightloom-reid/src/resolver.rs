//! Multi-factor threshold identity resolver with positive/negative references.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::collections::BTreeMap;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::{
    CameraTopology, EmbeddingError, EmbeddingStore, IdentityMatch, IdentityResolver,
    IdentityScoreFactors, MatchDecision, ScoreContext, SubjectReference, TrackFragment,
    class_compatibility, cosine_similarity, temporal_plausibility,
};
use sightloom_core::{MediaTime, SourceId};

/// Configuration for [`ThresholdResolver`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolveConfig {
    /// Fused scores at or above this value are accepted.
    pub accept_threshold: f32,
    /// Fused scores at or below this value are rejected.
    pub reject_threshold: f32,
    /// When true, only same-modality candidates are considered.
    pub require_same_modality: bool,
    /// If raw cosine to any negative example exceeds this, force reject.
    pub negative_reject_threshold: f32,
    /// When true, unknown camera hops score topology factor `0`.
    pub strict_camera_topology: bool,
    /// Optional max gap between subject sightings (nanoseconds); `None` = open.
    pub max_identity_gap_ns: Option<i64>,
    /// Optional per-source accept threshold override (use via gallery API).
    pub default_source_accept: Option<f32>,
}

impl Default for ResolveConfig {
    fn default() -> Self {
        Self {
            accept_threshold: 0.75,
            reject_threshold: 0.40,
            require_same_modality: true,
            negative_reject_threshold: 0.70,
            strict_camera_topology: false,
            max_identity_gap_ns: None,
            default_source_accept: None,
        }
    }
}

impl ResolveConfig {
    /// Validates finite thresholds and ordering `reject <= accept`.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError::InvalidVector`] for invalid ranges.
    pub fn validate(self) -> Result<Self, EmbeddingError> {
        let values = [
            self.accept_threshold,
            self.reject_threshold,
            self.negative_reject_threshold,
        ];
        for value in values {
            if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
                return Err(EmbeddingError::InvalidVector);
            }
        }
        if self.reject_threshold > self.accept_threshold {
            return Err(EmbeddingError::InvalidVector);
        }
        if let Some(src) = self.default_source_accept
            && (!src.is_finite() || !(0.0..=1.0).contains(&src))
        {
            return Err(EmbeddingError::InvalidVector);
        }
        Ok(self)
    }
}

/// Cosine + multi-factor resolver using an embedding store and thresholds.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
pub struct ThresholdResolver<'a> {
    store: &'a EmbeddingStore,
    config: ResolveConfig,
    topology: Option<&'a CameraTopology>,
    /// Last sighting time per subject (for temporal gating).
    last_seen: Option<&'a BTreeMap<u64, (SourceId, MediaTime)>>,
    /// Optional per-source accept thresholds.
    source_accept: Option<&'a BTreeMap<u32, f32>>,
    context: ScoreContext,
}

#[cfg(feature = "alloc")]
impl<'a> ThresholdResolver<'a> {
    /// Creates a resolver from a validated configuration (unit score context).
    ///
    /// # Errors
    ///
    /// Propagates config validation errors.
    pub fn new(store: &'a EmbeddingStore, config: ResolveConfig) -> Result<Self, EmbeddingError> {
        Self::with_context(
            store,
            config,
            ScoreContext::new(SourceId(0), MediaTime::default()),
            None,
            None,
            None,
        )
    }

    /// Creates a resolver with multi-factor context and optional topology.
    ///
    /// # Errors
    ///
    /// Propagates config validation errors.
    pub fn with_context(
        store: &'a EmbeddingStore,
        config: ResolveConfig,
        context: ScoreContext,
        topology: Option<&'a CameraTopology>,
        last_seen: Option<&'a BTreeMap<u64, (SourceId, MediaTime)>>,
        source_accept: Option<&'a BTreeMap<u32, f32>>,
    ) -> Result<Self, EmbeddingError> {
        Ok(Self {
            store,
            config: config.validate()?,
            topology,
            last_seen,
            source_accept,
            context,
        })
    }

    fn accept_threshold_for(&self, source: SourceId) -> f32 {
        self.source_accept
            .and_then(|m| m.get(&source.0).copied())
            .or(self.config.default_source_accept)
            .unwrap_or(self.config.accept_threshold)
    }

    #[allow(clippy::too_many_lines)]
    fn score_subject(
        &self,
        query: &[f32],
        subject: &SubjectReference,
        fragment: &TrackFragment,
    ) -> Option<IdentityMatch> {
        if self.config.require_same_modality && subject.modality != fragment.modality {
            return None;
        }

        let mut pos_best = None;
        let mut ref_class = None;
        let mut ref_quality = 1.0_f32;
        for sample in subject.positives() {
            let Some(handle) = sample.embedding else {
                continue;
            };
            let Ok(vector) = self.store.get(handle) else {
                continue;
            };
            let Some(score) = cosine_similarity(query, vector) else {
                continue;
            };
            if pos_best.is_none_or(|best: f32| score > best) {
                pos_best = Some(score);
                ref_class = sample.class_id;
                ref_quality = sample.quality.unwrap_or(1.0);
            }
        }
        if pos_best.is_none() {
            for sample in &subject.samples {
                if sample.is_positive.is_some() {
                    continue;
                }
                let Some(handle) = sample.embedding else {
                    continue;
                };
                let Ok(vector) = self.store.get(handle) else {
                    continue;
                };
                let Some(score) = cosine_similarity(query, vector) else {
                    continue;
                };
                if pos_best.is_none_or(|best: f32| score > best) {
                    pos_best = Some(score);
                    ref_class = sample.class_id;
                    ref_quality = sample.quality.unwrap_or(1.0);
                }
            }
        }

        let mut forced_reject = false;
        for sample in subject.negatives() {
            let Some(handle) = sample.embedding else {
                continue;
            };
            let Ok(vector) = self.store.get(handle) else {
                continue;
            };
            let Some(score) = cosine_similarity(query, vector) else {
                continue;
            };
            if score >= self.config.negative_reject_threshold {
                forced_reject = true;
            }
        }

        let similarity = pos_best?;
        let quality = self
            .context
            .embedding_quality
            .min(fragment.embedding_quality)
            .min(ref_quality)
            .clamp(0.0, 1.0);

        let (last_source, last_at) = self
            .last_seen
            .and_then(|m| m.get(&subject.subject_id.0).copied())
            .map_or((None, None), |(s, t)| (Some(s), Some(t)));

        let temporal = temporal_plausibility(
            last_at,
            self.context.query_at,
            self.config.max_identity_gap_ns,
        );

        let elapsed = last_at.map_or(0, |t| {
            self.context
                .query_at
                .duration_since_ns(t)
                .unsigned_abs()
                .min(i64::MAX as u64) as i64
        });
        let topo_from = last_source.unwrap_or(fragment.source_id);
        let topology = self.topology.map_or(1.0, |topo| {
            topo.factor(
                topo_from,
                fragment.source_id,
                elapsed,
                self.config.strict_camera_topology,
            )
        });

        let class_c = class_compatibility(fragment.class_id.or(self.context.class_id), ref_class);
        let prior = self.context.prior_identity_confidence.clamp(0.0, 1.0);

        let factors = IdentityScoreFactors {
            embedding_similarity: similarity,
            embedding_quality: quality,
            temporal_plausibility: temporal,
            camera_topology: topology,
            class_compatibility: class_c,
            prior_identity_confidence: prior,
        };
        let fused = factors.fused();

        // Topology hard gate: impossible hop cannot Accept even if cosine is high.
        let decision = if forced_reject || topology <= 0.0 || temporal <= 0.0 || class_c <= 0.0 {
            MatchDecision::Reject
        } else {
            let accept = self.accept_threshold_for(fragment.source_id);
            // Map legacy cosine thresholds: if config still uses cosine-like
            // ranges above 0.5, also allow accept on raw similarity path when
            // all factors are unit.
            let accept_fused = if accept > 1.0 {
                1.0
            } else if accept > 0.5 && quality >= 0.999 && topology >= 0.999 {
                // Backward-compatible path: treat accept_threshold as cosine.
                let cosine_accept = accept;
                if similarity >= cosine_accept {
                    return Some(IdentityMatch {
                        subject_id: subject.subject_id,
                        score: similarity,
                        decision: MatchDecision::Accept,
                        factors,
                    });
                }
                // Convert cosine threshold to fused scale for mixed paths.
                ((accept + 1.0) * 0.5).clamp(0.0, 1.0)
            } else {
                accept
            };
            let reject_fused = if self.config.reject_threshold > 0.5 {
                ((self.config.reject_threshold + 1.0) * 0.5).clamp(0.0, 1.0)
            } else {
                self.config.reject_threshold.clamp(0.0, 1.0)
            };
            if fused >= accept_fused || similarity >= accept {
                MatchDecision::Accept
            } else if fused <= reject_fused || similarity <= self.config.reject_threshold {
                MatchDecision::Reject
            } else {
                MatchDecision::Uncertain
            }
        };

        // Prefer fused score for ranking; keep cosine visible in factors.
        let score = if decision == MatchDecision::Accept
            && similarity >= self.accept_threshold_for(fragment.source_id)
        {
            similarity.max(fused)
        } else {
            fused
        };

        Some(IdentityMatch {
            subject_id: subject.subject_id,
            score,
            decision,
            factors,
        })
    }
}

#[cfg(feature = "alloc")]
impl IdentityResolver for ThresholdResolver<'_> {
    fn resolve_fragment(
        &self,
        fragment: &TrackFragment,
        candidates: &[SubjectReference],
    ) -> Vec<IdentityMatch> {
        let Some(handle) = fragment.embedding else {
            return Vec::new();
        };
        let Ok(query) = self.store.get(handle) else {
            return Vec::new();
        };

        let mut matches = Vec::new();
        for subject in candidates {
            if let Some(result) = self.score_subject(query, subject, fragment) {
                matches.push(result);
            }
        }
        matches.sort_unstable_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| left.subject_id.0.cmp(&right.subject_id.0))
        });
        matches
    }
}

/// Subject last-seen map keyed by [`SubjectId::0`].
pub type SubjectLastSeen = BTreeMap<u64, (SourceId, MediaTime)>;
