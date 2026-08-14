//! Offline quality / evaluation report helpers (not published MOT/ReID leaderboards).
//!
//! Hosts fill labeled ground truth and compare against `SightLoom` outputs to
//! produce portable metrics. Numbers stay with the host evidence pack.
#![allow(clippy::cast_precision_loss)]

use sightloom_index::RedactionInterval;
use sightloom_tracking::BaselineMotMetrics;

/// Tracking quality summary wrapping baseline CLEAR metrics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrackingQualityReport {
    /// CLEAR-style metrics from synthetic or host-built frames.
    pub metrics: BaselineMotMetrics,
    /// Free-form host note (e.g. dataset name) — keep empty in library tests.
    pub dataset_tag: u32,
}

impl TrackingQualityReport {
    /// Wraps metrics with a host tag.
    #[must_use]
    pub const fn new(metrics: BaselineMotMetrics, dataset_tag: u32) -> Self {
        Self {
            metrics,
            dataset_tag,
        }
    }

    /// True when MOTA is at least `min_mota` and ID switches ≤ `max_switches`.
    #[must_use]
    pub fn passes_smoke(self, min_mota: f32, max_switches: u32) -> bool {
        self.metrics.mota >= min_mota && self.metrics.id_switches <= max_switches
    }
}

/// One pixel-level redaction evaluation sample (host supplies pixel counts).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RedactionPixelSample {
    /// Interval id under test (`0` if none).
    pub interval_id: u64,
    /// Target subject pixels that **should** be redacted.
    pub target_pixels: u64,
    /// Target pixels still visible after host render (leakage).
    pub target_visible_pixels: u64,
    /// Non-target pixels that were redacted (over-blur).
    pub collateral_redacted_pixels: u64,
    /// Non-target pixels total in frame/region.
    pub non_target_pixels: u64,
}

/// Aggregated redaction quality (host-render evaluation; `SightLoom` stores intervals only).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RedactionQualityReport {
    /// Samples aggregated.
    pub samples: u32,
    /// Mean fraction of target pixels still visible (`0` = perfect).
    pub mean_target_leakage: f32,
    /// Mean fraction of non-target area redacted (`0` = no over-blur).
    pub mean_collateral_ratio: f32,
    /// Intervals evaluated that had zero host samples (coverage gap).
    pub intervals_without_pixels: u32,
}

/// Builds a redaction quality report from host pixel samples.
#[must_use]
pub fn evaluate_redaction_pixels(samples: &[RedactionPixelSample]) -> RedactionQualityReport {
    if samples.is_empty() {
        return RedactionQualityReport::default();
    }
    let mut leak_sum = 0.0_f32;
    let mut coll_sum = 0.0_f32;
    let mut n = 0_u32;
    for s in samples {
        if s.target_pixels > 0 {
            leak_sum += s.target_visible_pixels as f32 / s.target_pixels as f32;
            n = n.saturating_add(1);
        }
        if s.non_target_pixels > 0 {
            coll_sum += s.collateral_redacted_pixels as f32 / s.non_target_pixels as f32;
        }
    }
    let denom = n.max(1) as f32;
    RedactionQualityReport {
        samples: u32::try_from(samples.len()).unwrap_or(u32::MAX),
        mean_target_leakage: leak_sum / denom,
        mean_collateral_ratio: coll_sum / (samples.len() as f32),
        intervals_without_pixels: 0,
    }
}

/// Counts planned redaction intervals missing a matching pixel sample id.
#[must_use]
pub fn redaction_coverage_gap(
    intervals: &[RedactionInterval],
    samples: &[RedactionPixelSample],
) -> u32 {
    let mut missing = 0_u32;
    for iv in intervals {
        if !samples.iter().any(|s| s.interval_id == iv.interval_id.0) {
            missing = missing.saturating_add(1);
        }
    }
    missing
}

/// Re-id quality snapshot from a calibration report (EER + recommended band).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReidQualityReport {
    /// Equal-error rate.
    pub eer: f32,
    /// Threshold at EER.
    pub eer_threshold: f32,
    /// Recommended accept.
    pub accept_threshold: f32,
    /// Recommended reject.
    pub reject_threshold: f32,
    /// Genuine pair count.
    pub genuine_count: u32,
    /// Impostor pair count.
    pub impostor_count: u32,
}

impl ReidQualityReport {
    /// Builds from a calibration report.
    #[must_use]
    pub fn from_calibration(report: &sightloom_reid::CalibrationReport) -> Self {
        Self {
            eer: report.eer,
            eer_threshold: report.eer_threshold,
            accept_threshold: report.recommended_accept,
            reject_threshold: report.recommended_reject,
            genuine_count: report.genuine_count,
            impostor_count: report.impostor_count,
        }
    }

    /// Smoke gate: EER under threshold with enough pairs.
    #[must_use]
    pub fn passes_smoke(self, max_eer: f32, min_pairs: u32) -> bool {
        self.eer <= max_eer && self.genuine_count >= min_pairs && self.impostor_count >= min_pairs
    }
}
