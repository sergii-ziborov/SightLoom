//! Threshold-based identity resolver with positive/negative references.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::{
    EmbeddingError, EmbeddingStore, IdentityMatch, IdentityResolver, MatchDecision,
    SubjectModality, SubjectReference, TrackFragment, cosine_similarity,
};

/// Configuration for [`ThresholdResolver`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolveConfig {
    /// Scores at or above this value are accepted.
    pub accept_threshold: f32,
    /// Scores at or below this value are rejected.
    pub reject_threshold: f32,
    /// When true, only same-modality candidates are considered.
    pub require_same_modality: bool,
    /// If similarity to any negative example exceeds this, force reject.
    pub negative_reject_threshold: f32,
}

impl Default for ResolveConfig {
    fn default() -> Self {
        Self {
            accept_threshold: 0.75,
            reject_threshold: 0.40,
            require_same_modality: true,
            negative_reject_threshold: 0.70,
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
        Ok(self)
    }
}

/// Cosine similarity resolver using an embedding store and thresholds.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
pub struct ThresholdResolver<'a> {
    store: &'a EmbeddingStore,
    config: ResolveConfig,
}

#[cfg(feature = "alloc")]
impl<'a> ThresholdResolver<'a> {
    /// Creates a resolver from a validated configuration.
    ///
    /// # Errors
    ///
    /// Propagates config validation errors.
    pub fn new(store: &'a EmbeddingStore, config: ResolveConfig) -> Result<Self, EmbeddingError> {
        Ok(Self {
            store,
            config: config.validate()?,
        })
    }

    fn score_subject(
        &self,
        query: &[f32],
        subject: &SubjectReference,
        query_modality: SubjectModality,
    ) -> Option<IdentityMatch> {
        if self.config.require_same_modality && subject.modality != query_modality {
            return None;
        }

        let mut pos_best = None;
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
            pos_best = Some(pos_best.map_or(score, |best: f32| best.max(score)));
        }
        // Unlabeled samples act as weak positives when no explicit positives exist.
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
                pos_best = Some(pos_best.map_or(score, |best: f32| best.max(score)));
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

        let score = pos_best?;
        let decision = if forced_reject {
            MatchDecision::Reject
        } else if score >= self.config.accept_threshold {
            MatchDecision::Accept
        } else if score <= self.config.reject_threshold {
            MatchDecision::Reject
        } else {
            MatchDecision::Uncertain
        };
        Some(IdentityMatch {
            subject_id: subject.subject_id,
            score,
            decision,
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
            if let Some(result) = self.score_subject(query, subject, fragment.modality) {
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
