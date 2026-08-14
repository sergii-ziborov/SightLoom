//! Lightweight One-Class SVM (RBF) anomaly backend (pure Rust, no libsvm).
//!
//! Schölkopf-style dual with **equal dual weights** over training points and an
//! RBF kernel. Not a full SMO / libsvm parity port — a portable classical
//! novelty detector behind [`crate::AnomalyDetector`].

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp
)]

extern crate alloc;

use crate::anomaly::{AnomalyEvent, AnomalyReason, Severity};
use crate::anomaly_backend::AnomalyDetector;
use crate::input::AnalysisSeries;
use crate::stats::hour_of_day_ns;
use alloc::{vec, vec::Vec};
use sightloom_core::{AnomalyId, MediaTime, SubjectId};

/// One-Class SVM configuration (RBF novelty detector).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OcSvmConfig {
    /// Expected outlier fraction in `(0, 1]` (sets the decision offset quantile).
    pub nu: f32,
    /// RBF kernel width: `k(x,y) = exp(-gamma * ||x-y||^2)`.
    ///
    /// `0` → auto `1 / n_features` after standardization.
    pub gamma: f32,
    /// Scores above this (normalized distance past the margin) flag anomalies.
    pub score_threshold: f32,
    /// Cap on stored support vectors (subsample with stride when larger).
    pub max_support_vectors: usize,
}

impl Default for OcSvmConfig {
    fn default() -> Self {
        Self {
            nu: 0.1,
            gamma: 0.0,
            score_threshold: 0.15,
            max_support_vectors: 256,
        }
    }
}

/// Fitted RBF One-Class model on dense feature rows.
#[derive(Clone, Debug, Default)]
pub struct OcSvmDetector {
    /// Config.
    pub config: OcSvmConfig,
    /// Standardized support vectors.
    support: Vec<Vec<f32>>,
    /// Dual weight per support vector (sum ≈ 1).
    alpha: Vec<f32>,
    /// Feature means used for standardization.
    mean: Vec<f32>,
    /// Feature stddevs (clamped).
    std: Vec<f32>,
    /// Decision offset `rho` (decision = sum alpha k - rho; negative → outlier).
    rho: f32,
    /// Effective gamma after fit.
    gamma: f32,
    fitted: bool,
}

impl OcSvmDetector {
    /// Creates an unfitted detector.
    #[must_use]
    pub const fn new(config: OcSvmConfig) -> Self {
        Self {
            config,
            support: Vec::new(),
            alpha: Vec::new(),
            mean: Vec::new(),
            std: Vec::new(),
            rho: 0.0,
            gamma: 0.0,
            fitted: false,
        }
    }

    /// Fits on an explicit feature matrix (`rows × features`).
    pub fn fit_matrix(&mut self, rows: &[Vec<f32>]) {
        self.support.clear();
        self.alpha.clear();
        self.mean.clear();
        self.std.clear();
        self.fitted = false;
        self.rho = 0.0;
        self.gamma = 0.0;

        if rows.is_empty() {
            return;
        }
        let n_features = rows[0].len();
        if n_features == 0 {
            return;
        }
        // Require consistent width.
        if rows.iter().any(|r| r.len() != n_features) {
            return;
        }

        // Standardization stats.
        let mut mean = vec![0.0_f32; n_features];
        for row in rows {
            for (j, v) in row.iter().enumerate() {
                mean[j] += *v;
            }
        }
        let n = rows.len() as f32;
        for m in &mut mean {
            *m /= n;
        }
        let mut std = vec![0.0_f32; n_features];
        for row in rows {
            for (j, v) in row.iter().enumerate() {
                let d = *v - mean[j];
                std[j] += d * d;
            }
        }
        for s in &mut std {
            *s = sqrt_approx(*s / n).max(1e-6);
        }

        let mut scaled: Vec<Vec<f32>> = rows
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, v)| (*v - mean[j]) / std[j])
                    .collect()
            })
            .collect();

        // Subsample if too many rows.
        let max_sv = self.config.max_support_vectors.max(2);
        if scaled.len() > max_sv {
            let stride = scaled.len().div_ceil(max_sv).max(1);
            scaled = scaled.into_iter().step_by(stride).collect();
        }

        let m = scaled.len();
        if m < 2 {
            return;
        }

        // Equal dual weights (all points treated as support vectors).
        let alpha_i = 1.0 / m as f32;
        let alpha = vec![alpha_i; m];

        let gamma = if self.config.gamma > 0.0 {
            self.config.gamma
        } else {
            // Median pairwise distance heuristic → gamma = 1 / (2 * median^2).
            auto_gamma(&scaled)
        };

        // Decision values on training set: f(x_i) = sum_j alpha_j k(x_i, x_j)
        let mut fvals = Vec::with_capacity(m);
        for i in 0..m {
            let mut s = 0.0_f32;
            for j in 0..m {
                s += alpha[j] * rbf(&scaled[i], &scaled[j], gamma);
            }
            fvals.push(s);
        }
        // rho = nu-quantile of f so roughly nu of training points fall below margin.
        let mut sorted = fvals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let nu = self.config.nu.clamp(0.01, 1.0);
        let q_idx = ((nu * (m as f32 - 1.0)) as usize).min(m - 1);
        let rho = sorted[q_idx];

        self.support = scaled;
        self.alpha = alpha;
        self.mean = mean;
        self.std = std;
        self.rho = rho;
        self.gamma = gamma;
        self.fitted = true;
    }

    /// Raw decision value: `sum alpha k(x, sv) - rho` (negative → more out-of-class).
    #[must_use]
    pub fn decision_function(&self, row: &[f32]) -> f32 {
        if !self.fitted || row.len() != self.mean.len() {
            return 0.0;
        }
        let scaled: Vec<f32> = row
            .iter()
            .enumerate()
            .map(|(j, v)| (*v - self.mean[j]) / self.std[j])
            .collect();
        let mut s = 0.0_f32;
        for (sv, a) in self.support.iter().zip(self.alpha.iter()) {
            s += *a * rbf(&scaled, sv, self.gamma);
        }
        s - self.rho
    }

    /// Anomaly score in roughly `[0, 1]` (higher = more anomalous).
    ///
    /// Maps negative decision values into a soft score via `1 - sigmoid(decision)`.
    #[must_use]
    pub fn score_row(&self, row: &[f32]) -> f32 {
        if !self.fitted {
            return 0.0;
        }
        let d = self.decision_function(row);
        // When d < 0 (outlier side), score rises toward 1.
        let sig = sigmoid(d);
        (1.0 - sig).clamp(0.0, 1.0)
    }

    /// Whether the model has been fitted.
    #[must_use]
    pub const fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl AnomalyDetector for OcSvmDetector {
    type Error = &'static str;

    fn fit(&mut self, history: &AnalysisSeries) -> Result<(), Self::Error> {
        let rows = features_from_series(history);
        if rows.len() < 4 {
            return Err("need at least 4 feature rows");
        }
        self.fit_matrix(&rows);
        if !self.fitted {
            return Err("ocsvm fit failed");
        }
        Ok(())
    }

    fn detect(
        &mut self,
        live: &AnalysisSeries,
        next_id: &mut u64,
    ) -> Result<Vec<AnomalyEvent>, Self::Error> {
        if !self.fitted {
            let _ = self.fit(live);
        }
        if !self.fitted {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for (subject, row, at_ns) in labeled_rows(live) {
            let score = self.score_row(&row);
            if score < self.config.score_threshold {
                continue;
            }
            let severity = if score > 0.7 {
                Severity::High
            } else if score > 0.4 {
                Severity::Medium
            } else {
                Severity::Low
            };
            let id = AnomalyId(*next_id);
            *next_id = next_id.saturating_add(1);
            let at = MediaTime::new(at_ns, 1_000_000_000).unwrap_or_default();
            out.push(AnomalyEvent {
                anomaly_id: id,
                score,
                severity,
                reasons: vec![AnomalyReason::Custom(101)], // ocsvm tag
                evidence: Vec::new(),
                subject_id: subject,
                source_id: None,
                at,
            });
        }
        Ok(out)
    }
}

fn rbf(a: &[f32], b: &[f32], gamma: f32) -> f32 {
    let mut dist2 = 0.0_f32;
    let n = a.len().min(b.len());
    for i in 0..n {
        let d = a[i] - b[i];
        dist2 += d * d;
    }
    exp_neg(gamma * dist2)
}

fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    let mut dist2 = 0.0_f32;
    let n = a.len().min(b.len());
    for i in 0..n {
        let d = a[i] - b[i];
        dist2 += d * d;
    }
    dist2
}

fn auto_gamma(rows: &[Vec<f32>]) -> f32 {
    // Sample up to ~64^2 / 2 pairwise distances for median.
    let n = rows.len();
    if n < 2 {
        return 1.0;
    }
    let step = (n / 32).max(1);
    let mut dists = Vec::new();
    let mut i = 0;
    while i < n {
        let mut j = i + step;
        while j < n {
            let d2 = sq_dist(&rows[i], &rows[j]);
            if d2.is_finite() && d2 > 0.0 {
                dists.push(sqrt_approx(d2));
            }
            j += step;
        }
        i += step;
    }
    if dists.is_empty() {
        return 0.5;
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let med = dists[dists.len() / 2].max(1e-3);
    // gamma = 1 / (2 * sigma^2) with sigma = median distance
    1.0 / (2.0 * med * med)
}

fn exp_neg(x: f32) -> f32 {
    // e^{-x} via Taylor / clamp (no_std, no libm).
    if !x.is_finite() {
        return 0.0;
    }
    let y = -x.clamp(0.0, 40.0);
    let mut term = 1.0_f32;
    let mut sum = 1.0_f32;
    for k in 1..20 {
        term *= y / k as f32;
        sum += term;
    }
    sum.clamp(0.0, 1.0)
}

fn sigmoid(x: f32) -> f32 {
    // 1 / (1 + e^{-x})
    let e = exp_neg(if x >= 0.0 { x } else { -x });
    if x >= 0.0 {
        1.0 / (1.0 + e)
    } else {
        e / (1.0 + e)
    }
}

fn sqrt_approx(x: f32) -> f32 {
    if x <= 0.0 || !x.is_finite() {
        return 0.0;
    }
    // Newton from bit-level guess.
    let mut y = f32::from_bits((x.to_bits() >> 1) + 0x1fbb_4000);
    for _ in 0..4 {
        y = 0.5 * (y + x / y);
    }
    y
}

fn features_from_series(series: &AnalysisSeries) -> Vec<Vec<f32>> {
    labeled_rows(series)
        .into_iter()
        .map(|(_, row, _)| row)
        .collect()
}

/// Converts nanoseconds to seconds as `f32` (avoids mantissa collapse at 1e9 ns).
fn ns_to_sec(ns: i64) -> f32 {
    (ns as f64 / 1_000_000_000.0) as f32
}

fn labeled_rows(series: &AnalysisSeries) -> Vec<(Option<SubjectId>, Vec<f32>, i64)> {
    let mut out = Vec::new();
    for d in &series.durations {
        out.push((
            d.subject_id,
            vec![ns_to_sec(d.duration_ns), 0.0, 0.0],
            d.at_ns,
        ));
    }
    let mut by_subj: Vec<(Option<SubjectId>, Vec<i64>)> = Vec::new();
    for e in &series.timed {
        if let Some((_, t)) = by_subj.iter_mut().find(|(s, _)| *s == e.subject_id) {
            t.push(e.at_ns);
        } else {
            by_subj.push((e.subject_id, vec![e.at_ns]));
        }
    }
    for (subject, mut times) in by_subj {
        times.sort_unstable();
        for w in times.windows(2) {
            let gap = ns_to_sec(w[1] - w[0]);
            let hour = f32::from(hour_of_day_ns(w[1]));
            out.push((subject, vec![0.0, gap, hour], w[1]));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::DurationSample;

    #[test]
    fn ocsvm_flags_far_outlier() {
        let mut hist = AnalysisSeries::default();
        // Dwells roughly 0.5s .. 2.5s (spread so f32 features stay informative).
        for i in 0..40 {
            let sec = 0.5 + (i as f32) * 0.05;
            hist.durations.push(DurationSample {
                subject_id: Some(SubjectId(1)),
                zone_id: None,
                duration_ns: (sec * 1_000_000_000.0) as i64,
                at_ns: i * 86_400_000_000_000,
                event_id: None,
            });
        }
        let mut det = OcSvmDetector::new(OcSvmConfig {
            nu: 0.1,
            gamma: 0.0,
            score_threshold: 0.1,
            max_support_vectors: 64,
        });
        det.fit(&hist).unwrap();
        assert!(det.is_fitted());

        // Features are duration_seconds (see labeled_rows).
        let inlier = det.score_row(&[1.5_f32, 0.0, 0.0]);
        let outlier = det.score_row(&[80.0_f32, 0.0, 0.0]);
        assert!(
            outlier > inlier,
            "outlier={outlier} inlier={inlier} decision_out={} decision_in={}",
            det.decision_function(&[80.0, 0.0, 0.0]),
            det.decision_function(&[1.5, 0.0, 0.0]),
        );

        let mut live = AnalysisSeries::default();
        live.durations.push(DurationSample {
            subject_id: Some(SubjectId(1)),
            zone_id: None,
            duration_ns: 80_000_000_000,
            at_ns: 50 * 86_400_000_000_000,
            event_id: None,
        });
        let mut next = 1_u64;
        let events = det.detect(&live, &mut next).unwrap();
        assert!(
            !events.is_empty() || outlier >= 0.1,
            "expected flag or high score, events={}, score={outlier}",
            events.len()
        );
    }

    #[test]
    fn decision_negative_for_distant_point() {
        let rows: Vec<Vec<f32>> = (0..20)
            .map(|i| vec![i as f32 * 0.01, 0.0, 0.0])
            .collect();
        let mut det = OcSvmDetector::new(OcSvmConfig::default());
        det.fit_matrix(&rows);
        assert!(det.is_fitted());
        let d_near = det.decision_function(&[0.05, 0.0, 0.0]);
        let d_far = det.decision_function(&[100.0, 0.0, 0.0]);
        assert!(d_far < d_near, "d_far={d_far} d_near={d_near}");
    }
}
