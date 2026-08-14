//! False-alarm-rate (FAR) calibration for anomaly score thresholds.
//!
//! Hosts score a **normal** (no-attack) history window, then pick a threshold
//! so that the fraction of scores exceeding it ≈ target FAR.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

extern crate alloc;

use alloc::vec::Vec;

use crate::input::AnalysisSeries;
use crate::stat_anomaly::{BaselineStats, StatAnomalyConfig};
use crate::stats::{hour_of_day_ns, robust_z_score, z_score};

/// One score used for FAR calibration (higher = more anomalous).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnomalyScoreSample {
    /// Anomaly score (e.g. absolute z-score).
    pub score: f32,
    /// When true, this sample is a known anomaly (for empirical FAR/FRR eval).
    pub is_anomaly: bool,
}

/// Result of calibrating a score threshold to a target false-alarm rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FarCalibrationReport {
    /// Requested FAR in `(0, 1]` (e.g. `0.01` = 1%).
    pub target_far: f32,
    /// Score threshold: flag when `score >= threshold`.
    pub threshold: f32,
    /// Empirical FAR on the calibration scores at `threshold`.
    pub empirical_far: f32,
    /// Number of **normal** scores used.
    pub n_normal: usize,
    /// Number of labeled anomaly scores (may be 0 for threshold-only calib).
    pub n_anomaly: usize,
    /// Empirical miss rate on labeled anomalies when provided (`None` if none).
    pub empirical_miss_rate: Option<f32>,
}

/// Calibrates a threshold so ≈ `target_far` of **normal** scores fire.
///
/// Uses the quantile of normal scores: threshold = percentile
/// `(1 - target_far)` of normal scores (higher scores more anomalous).
///
/// # Errors
///
/// Returns `None` when there are no finite normal scores or `target_far` invalid.
#[must_use]
pub fn calibrate_far_threshold(
    samples: &[AnomalyScoreSample],
    target_far: f32,
) -> Option<FarCalibrationReport> {
    if !(target_far > 0.0 && target_far <= 1.0) {
        return None;
    }
    let mut normals: Vec<f32> = samples
        .iter()
        .filter(|s| !s.is_anomaly && s.score.is_finite())
        .map(|s| s.score)
        .collect();
    if normals.is_empty() {
        return None;
    }
    normals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    // Threshold at (1 - FAR) quantile so fraction above ≈ FAR.
    let q = (1.0 - target_far).clamp(0.0, 1.0);
    let idx = ((q * (normals.len().saturating_sub(1) as f32)) + 0.5) as usize;
    let idx = idx.min(normals.len().saturating_sub(1));
    let threshold = normals[idx];

    let n_normal = normals.len();
    let false_alarms = normals.iter().filter(|s| **s >= threshold).count();
    let empirical_far = false_alarms as f32 / n_normal as f32;

    let anomalies: Vec<f32> = samples
        .iter()
        .filter(|s| s.is_anomaly && s.score.is_finite())
        .map(|s| s.score)
        .collect();
    let n_anomaly = anomalies.len();
    let empirical_miss_rate = if n_anomaly == 0 {
        None
    } else {
        let misses = anomalies.iter().filter(|s| **s < threshold).count();
        Some(misses as f32 / n_anomaly as f32)
    };

    Some(FarCalibrationReport {
        target_far,
        threshold,
        empirical_far,
        n_normal,
        n_anomaly,
        empirical_miss_rate,
    })
}

/// Maps a FAR calibration result into a statistical z-threshold config.
#[must_use]
pub fn apply_far_to_stat_config(
    mut config: StatAnomalyConfig,
    report: &FarCalibrationReport,
) -> StatAnomalyConfig {
    // z_threshold is absolute z; clamp to a sensible band.
    config.z_threshold = report.threshold.clamp(1.0, 12.0);
    config
}

/// Scores dwell/gap/hour absolute z-values on `series` vs `baseline` (for FAR).
///
/// All returned samples are marked `is_anomaly: false` (normal-window assumption).
#[must_use]
pub fn score_series_vs_baseline(
    series: &AnalysisSeries,
    baseline: &BaselineStats,
    config: StatAnomalyConfig,
) -> Vec<AnomalyScoreSample> {
    let mut out = Vec::new();

    if let (Some(mu), Some(sd)) = (baseline.dwell_mean, baseline.dwell_std) {
        for d in &series.durations {
            let v = d.duration_ns as f32;
            if let Some(z) = z_score(v, mu, sd) {
                out.push(AnomalyScoreSample {
                    score: z,
                    is_anomaly: false,
                });
            }
        }
    }
    if config.use_robust
        && let (Some(med), Some(mad_v)) = (baseline.dwell_median, baseline.dwell_mad)
    {
        for d in &series.durations {
            let v = d.duration_ns as f32;
            if let Some(z) = robust_z_score(v, med, mad_v) {
                out.push(AnomalyScoreSample {
                    score: z,
                    is_anomaly: false,
                });
            }
        }
    }
    if let (Some(mu), Some(sd)) = (baseline.gap_mean, baseline.gap_std) {
        let mut by_subj: Vec<(Option<sightloom_core::SubjectId>, Vec<i64>)> = Vec::new();
        for e in &series.timed {
            if let Some((_, t)) = by_subj.iter_mut().find(|(s, _)| *s == e.subject_id) {
                t.push(e.at_ns);
            } else {
                by_subj.push((e.subject_id, alloc::vec![e.at_ns]));
            }
        }
        for (_, mut times) in by_subj {
            times.sort_unstable();
            for w in times.windows(2) {
                let gap = (w[1] - w[0]) as f32;
                if let Some(z) = z_score(gap, mu, sd) {
                    out.push(AnomalyScoreSample {
                        score: z,
                        is_anomaly: false,
                    });
                }
            }
        }
    }
    if let (Some(mu), Some(sd)) = (baseline.hour_mean, baseline.hour_std) {
        for e in &series.timed {
            let hour = f32::from(hour_of_day_ns(e.at_ns));
            let d = (hour - mu).abs();
            let delta = d.min(24.0 - d);
            if let Some(z) = z_score(mu + delta, mu, sd.max(0.5)) {
                out.push(AnomalyScoreSample {
                    score: z,
                    is_anomaly: false,
                });
            }
        }
    }
    out
}

/// End-to-end: baseline fit on history → score history → FAR threshold.
#[must_use]
pub fn calibrate_far_from_series(
    history: &AnalysisSeries,
    config: StatAnomalyConfig,
    target_far: f32,
) -> Option<FarCalibrationReport> {
    let baseline = crate::stat_anomaly::build_baseline(history, config);
    let scores = score_series_vs_baseline(history, &baseline, config);
    calibrate_far_threshold(&scores, target_far)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::DurationSample;
    use sightloom_core::SubjectId;

    #[test]
    fn far_threshold_orders_normals() {
        let mut samples = Vec::new();
        for i in 0..100 {
            samples.push(AnomalyScoreSample {
                score: i as f32 * 0.05,
                is_anomaly: false,
            });
        }
        // High scores are anomalies in this test set.
        for i in 0..10 {
            samples.push(AnomalyScoreSample {
                score: 10.0 + i as f32,
                is_anomaly: true,
            });
        }
        let report = calibrate_far_threshold(&samples, 0.05).unwrap();
        assert!(report.threshold > 0.0);
        assert!(report.empirical_far <= 0.15);
        assert!(report.empirical_miss_rate.is_some());
    }

    #[test]
    fn series_far_smoke() {
        let mut hist = AnalysisSeries::default();
        for i in 0..30 {
            hist.durations.push(DurationSample {
                subject_id: Some(SubjectId(1)),
                source_id: None,
                zone_id: None,
                duration_ns: 1_000_000_000 + i * 10_000,
                at_ns: i * 86_400_000_000_000,
                event_id: None,
            });
        }
        let report = calibrate_far_from_series(&hist, StatAnomalyConfig::default(), 0.1);
        assert!(report.is_some());
    }
}
