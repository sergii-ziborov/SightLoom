//! ROC / EER calibration for identity score thresholds.
//!
//! Hosts supply labeled pair scores (genuine vs impostor). `SightLoom` computes
//! an ROC curve, equal-error rate, and recommended accept/reject thresholds.
//! This is **not** a production biometrics lab report — it is a portable
//! foundation for tuning [`crate::ResolveConfig`].

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
use crate::EmbeddingStore;
use crate::{EmbeddingError, ResolveConfig, cosine_similarity};
#[cfg(feature = "alloc")]
use sightloom_core::EmbeddingRef;

/// One labeled similarity score for calibration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LabeledScore {
    /// Similarity / fused score in approximately `[-1.0, 1.0]` or `[0.0, 1.0]`.
    pub score: f32,
    /// `true` = same identity (genuine), `false` = different (impostor).
    pub genuine: bool,
}

/// One operating point on the ROC curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RocPoint {
    /// Decision threshold: scores ≥ threshold count as positive (same id).
    pub threshold: f32,
    /// True positive rate (sensitivity / recall).
    pub tpr: f32,
    /// False positive rate (1 − specificity).
    pub fpr: f32,
    /// False negative rate (`1 - tpr`).
    pub fnr: f32,
}

/// Result of ROC / EER calibration.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationReport {
    /// Sampled ROC points (thresholds descending).
    pub roc: Vec<RocPoint>,
    /// Equal-error rate ≈ FPR ≈ FNR at the chosen threshold.
    pub eer: f32,
    /// Threshold where |FPR − FNR| is minimized.
    pub eer_threshold: f32,
    /// Suggested accept threshold (at or above EER, biased toward lower FPR).
    pub recommended_accept: f32,
    /// Suggested reject threshold (below EER).
    pub recommended_reject: f32,
    /// Number of genuine pairs.
    pub genuine_count: u32,
    /// Number of impostor pairs.
    pub impostor_count: u32,
}

/// Builds an ROC curve and EER estimate from labeled scores.
///
/// `n_thresholds` controls sampling density along unique score values
/// (clamped to at least 2 and at most the number of distinct scores).
///
/// # Errors
///
/// Returns [`EmbeddingError::InvalidVector`] when scores are empty, non-finite,
/// or only one class is present.
pub fn compute_roc(
    scores: &[LabeledScore],
    n_thresholds: usize,
) -> Result<CalibrationReport, EmbeddingError> {
    if scores.is_empty() {
        return Err(EmbeddingError::InvalidVector);
    }
    let mut genuine_count = 0_u32;
    let mut impostor_count = 0_u32;
    for s in scores {
        if !s.score.is_finite() {
            return Err(EmbeddingError::InvalidVector);
        }
        if s.genuine {
            genuine_count = genuine_count.saturating_add(1);
        } else {
            impostor_count = impostor_count.saturating_add(1);
        }
    }
    if genuine_count == 0 || impostor_count == 0 {
        return Err(EmbeddingError::InvalidVector);
    }

    let mut thresholds: Vec<f32> = scores.iter().map(|s| s.score).collect();
    thresholds.sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
    thresholds.dedup_by(|a, b| (*a - *b).abs() < 1e-7);

    let n = n_thresholds.clamp(2, thresholds.len().max(2));
    let step = (thresholds.len().saturating_sub(1)).max(1) as f32 / (n.saturating_sub(1) as f32);
    let mut sampled = Vec::with_capacity(n);
    for i in 0..n {
        // Portable (no libm/`round`): nearest index for non-negative values.
        let idx = ((i as f32) * step + 0.5) as usize;
        let idx = idx.min(thresholds.len().saturating_sub(1));
        let t = thresholds[idx];
        if sampled
            .last()
            .is_none_or(|p: &RocPoint| (p.threshold - t).abs() > 1e-7)
        {
            sampled.push(eval_threshold(scores, t, genuine_count, impostor_count));
        }
    }
    // Ensure extremes.
    let lo = scores.iter().map(|s| s.score).fold(f32::INFINITY, f32::min);
    let hi = scores
        .iter()
        .map(|s| s.score)
        .fold(f32::NEG_INFINITY, f32::max);
    sampled.insert(
        0,
        eval_threshold(scores, hi + 1e-3, genuine_count, impostor_count),
    );
    sampled.push(eval_threshold(
        scores,
        lo - 1e-3,
        genuine_count,
        impostor_count,
    ));

    // EER: minimize |fpr - fnr|
    let mut best = sampled[0];
    let mut best_gap = (best.fpr - best.fnr).abs();
    for p in &sampled {
        let gap = (p.fpr - p.fnr).abs();
        if gap < best_gap {
            best_gap = gap;
            best = *p;
        }
    }
    let eer = 0.5 * (best.fpr + best.fnr);
    // Accept slightly above EER (stricter), reject below with a band.
    let band = ((hi - lo) * 0.05).clamp(0.02, 0.15);
    let recommended_accept = (best.threshold + band * 0.5).clamp(lo, hi);
    let recommended_reject = (best.threshold - band).clamp(lo, hi);

    Ok(CalibrationReport {
        roc: sampled,
        eer,
        eer_threshold: best.threshold,
        recommended_accept,
        recommended_reject,
        genuine_count,
        impostor_count,
    })
}

fn eval_threshold(
    scores: &[LabeledScore],
    threshold: f32,
    genuine_count: u32,
    impostor_count: u32,
) -> RocPoint {
    let mut tp = 0_u32;
    let mut fp = 0_u32;
    for s in scores {
        let pred_pos = s.score >= threshold;
        if pred_pos && s.genuine {
            tp = tp.saturating_add(1);
        } else if pred_pos && !s.genuine {
            fp = fp.saturating_add(1);
        }
    }
    let tpr = tp as f32 / genuine_count as f32;
    let fpr = fp as f32 / impostor_count as f32;
    RocPoint {
        threshold,
        tpr,
        fpr,
        fnr: 1.0 - tpr,
    }
}

/// Applies calibration recommendations onto a [`ResolveConfig`].
///
/// Sets `accept_threshold` / `reject_threshold` from the report (clamped so
/// `reject <= accept`). Leaves other fields unchanged.
#[must_use]
pub fn resolve_config_from_calibration(
    mut config: ResolveConfig,
    report: &CalibrationReport,
) -> ResolveConfig {
    let mut accept = report.recommended_accept;
    let mut reject = report.recommended_reject;
    if reject > accept {
        core::mem::swap(&mut accept, &mut reject);
    }
    config.accept_threshold = accept;
    config.reject_threshold = reject;
    config
}

/// Builds labeled cosine scores from pairs of embedding handles.
///
/// # Errors
///
/// Returns store lookup / vector errors.
#[cfg(feature = "alloc")]
pub fn labeled_scores_from_pairs(
    store: &EmbeddingStore,
    pairs: &[(EmbeddingRef, EmbeddingRef, bool)],
) -> Result<Vec<LabeledScore>, EmbeddingError> {
    let mut out = Vec::with_capacity(pairs.len());
    for &(a, b, genuine) in pairs {
        let va = store.get(a)?;
        let vb = store.get(b)?;
        let Some(sim) = cosine_similarity(va, vb) else {
            return Err(EmbeddingError::InvalidVector);
        };
        out.push(LabeledScore {
            score: sim,
            genuine,
        });
    }
    Ok(out)
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    #[test]
    fn eer_separates_well_split_scores() {
        let mut scores = Vec::new();
        // Genuine around 0.9
        for i in 0..20 {
            scores.push(LabeledScore {
                score: 0.85 + (i as f32) * 0.005,
                genuine: true,
            });
        }
        // Impostor around 0.2
        for i in 0..20 {
            scores.push(LabeledScore {
                score: 0.10 + (i as f32) * 0.01,
                genuine: false,
            });
        }
        let report = compute_roc(&scores, 32).unwrap();
        assert!(report.eer < 0.15, "eer={}", report.eer);
        assert!(report.recommended_accept > report.recommended_reject);
        assert!(report.recommended_accept > 0.4);
        let cfg = resolve_config_from_calibration(ResolveConfig::default(), &report);
        assert!(cfg.accept_threshold >= cfg.reject_threshold);
        assert!(cfg.validate().is_ok());
    }
}
