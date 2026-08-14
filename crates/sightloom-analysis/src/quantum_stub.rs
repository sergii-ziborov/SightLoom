//! Host-side **quantum anomaly** adapter contract (no quantum SDK in-tree).
//!
//! Real quantum / hybrid solvers live in host binaries. This module only:
//! - documents the intended plug-in point ([`AnomalyDetector`])
//! - provides a deterministic **classical stub** that mimics a quantum-shaped
//!   score so hosts can wire session paths before a real backend exists.
//!
//! Custom reason code: [`AnomalyReason::Custom`]`(200)`.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

extern crate alloc;

use crate::anomaly::{AnomalyEvent, AnomalyReason, Severity};
use crate::anomaly_backend::AnomalyDetector;
use crate::input::AnalysisSeries;
use crate::stats::{hour_of_day_ns, mean, stddev, z_score};
use alloc::vec;
use alloc::vec::Vec;
use sightloom_core::{AnomalyId, MediaTime};

/// Configuration for the classical quantum-shaped stub.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantumStubConfig {
    /// Score threshold in `[0, 1]` (higher = more anomalous).
    pub score_threshold: f32,
    /// Deterministic seed mixed into the pseudo-amplitude.
    pub seed: u64,
}

impl Default for QuantumStubConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.75,
            seed: 0x00C0_FFEE_0001,
        }
    }
}

/// Classical stub implementing [`AnomalyDetector`] with a quantum-shaped API.
///
/// **Not** a real quantum algorithm. Hosts replace this with a true backend
/// that still implements [`AnomalyDetector`].
#[derive(Clone, Debug, Default)]
pub struct QuantumStubDetector {
    /// Config.
    pub config: QuantumStubConfig,
    /// Fitted dwell mean.
    dwell_mean: Option<f32>,
    /// Fitted dwell std.
    dwell_std: Option<f32>,
    fitted: bool,
}

impl QuantumStubDetector {
    /// Creates an unfitted stub.
    #[must_use]
    pub const fn new(config: QuantumStubConfig) -> Self {
        Self {
            config,
            dwell_mean: None,
            dwell_std: None,
            fitted: false,
        }
    }

    /// Pseudo quantum score in `[0, 1]` from a dwell nanoseconds value.
    #[must_use]
    pub fn score_dwell(&self, duration_ns: i64) -> f32 {
        let v = duration_ns as f32;
        let z = match (self.dwell_mean, self.dwell_std) {
            (Some(mu), Some(sd)) => z_score(v, mu, sd).unwrap_or(0.0),
            _ => 0.0,
        };
        // Map |z| through a smooth amplitude with seed phase.
        let phase = (self.config.seed as f32 * 1e-9).sin().abs();
        let amp = 1.0 - (-0.35 * z).exp().clamp(0.0, 1.0);
        (0.55 * amp + 0.45 * phase * amp.min(1.0)).clamp(0.0, 1.0)
    }
}

impl AnomalyDetector for QuantumStubDetector {
    type Error = &'static str;

    fn fit(&mut self, history: &AnalysisSeries) -> Result<(), Self::Error> {
        let dwells: Vec<f32> = history
            .durations
            .iter()
            .map(|d| d.duration_ns as f32)
            .filter(|v| *v > 0.0 && v.is_finite())
            .collect();
        if dwells.len() < 4 {
            return Err("need at least 4 dwell samples");
        }
        self.dwell_mean = mean(&dwells);
        self.dwell_std = stddev(&dwells);
        self.fitted = self.dwell_mean.is_some() && self.dwell_std.is_some();
        if !self.fitted {
            return Err("fit failed");
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
        for d in &live.durations {
            let score = self.score_dwell(d.duration_ns);
            if score < self.config.score_threshold {
                continue;
            }
            let id = AnomalyId(*next_id);
            *next_id = next_id.saturating_add(1);
            let at = MediaTime::new(d.at_ns, 1_000_000_000).unwrap_or_default();
            // Mild use of hour so the stub is not pure dwell.
            let _hour = hour_of_day_ns(d.at_ns);
            out.push(AnomalyEvent {
                anomaly_id: id,
                score,
                severity: if score > 0.9 {
                    Severity::High
                } else {
                    Severity::Medium
                },
                reasons: vec![AnomalyReason::Custom(200)],
                evidence: Vec::new(),
                subject_id: d.subject_id,
                source_id: d.source_id,
                at,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::DurationSample;
    use sightloom_core::SubjectId;

    #[test]
    fn stub_fits_and_scores() {
        let mut hist = AnalysisSeries::default();
        for i in 0..20 {
            hist.durations.push(DurationSample {
                subject_id: Some(SubjectId(1)),
                source_id: None,
                zone_id: None,
                duration_ns: 1_000_000_000 + i * 1_000,
                at_ns: i * 86_400_000_000_000,
                event_id: None,
            });
        }
        let mut det = QuantumStubDetector::new(QuantumStubConfig::default());
        det.fit(&hist).unwrap();
        let mut live = AnalysisSeries::default();
        live.durations.push(DurationSample {
            subject_id: Some(SubjectId(1)),
            source_id: None,
            zone_id: None,
            duration_ns: 50_000_000_000,
            at_ns: 30 * 86_400_000_000_000,
            event_id: None,
        });
        let mut next = 1_u64;
        let events = det.detect(&live, &mut next).unwrap();
        let s = det.score_dwell(50_000_000_000);
        assert!((0.0..=1.0).contains(&s));
        let _ = events;
    }
}
