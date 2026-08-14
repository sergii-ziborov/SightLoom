//! Anomaly FAR + scoped-baseline evidence section.

use crate::error::HostError;
use core::fmt::Write as _;
use serde::{Deserialize, Serialize};
use sightloom::analysis::{
    AnalysisSeries, DurationSample, ScopedBaselineStore, StatAnomalyConfig, TimedSubjectEvent,
    build_baseline, calibrate_far_from_series, score_series_vs_baseline,
};
use sightloom::core::{SourceId, SubjectId};
use std::fs;
use std::path::Path;

/// Anomaly FAR / scoped-baseline evidence.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnomalyEvidence {
    /// Markdown report.
    pub markdown: String,
    /// Target FAR used.
    pub target_far: f32,
    /// Calibrated z-threshold.
    pub threshold: f32,
    /// Empirical FAR on normal scores.
    pub empirical_far: f32,
    /// Subject-scoped baseline count on synthetic history.
    pub subject_scopes: usize,
    /// Source-scoped baseline count.
    pub source_scopes: usize,
    /// Smoke pass.
    pub smoke_pass: bool,
}

/// Builds synthetic anomaly FAR + scope evidence.
///
/// # Errors
///
/// Calibration failure when history is degenerate.
pub fn build_synthetic_anomaly_evidence() -> Result<AnomalyEvidence, HostError> {
    let mut hist = AnalysisSeries::default();
    // Subject 1 on camera 1 — stable dwells.
    for i in 0..25 {
        hist.durations.push(DurationSample {
            subject_id: Some(SubjectId(1)),
            source_id: Some(SourceId(1)),
            zone_id: None,
            duration_ns: 1_000_000_000 + i * 5_000,
            at_ns: i * 3_600_000_000_000,
            event_id: None,
        });
        hist.timed.push(TimedSubjectEvent {
            subject_id: Some(SubjectId(1)),
            source_id: Some(SourceId(1)),
            at_ns: i * 3_600_000_000_000,
            event_id: None,
            kind_tag: 0,
        });
    }
    // Subject 2 on camera 2.
    for i in 0..20 {
        hist.timed.push(TimedSubjectEvent {
            subject_id: Some(SubjectId(2)),
            source_id: Some(SourceId(2)),
            at_ns: i * 7_200_000_000_000,
            event_id: None,
            kind_tag: 0,
        });
        hist.durations.push(DurationSample {
            subject_id: Some(SubjectId(2)),
            source_id: Some(SourceId(2)),
            zone_id: None,
            duration_ns: 2_000_000_000 + i * 8_000,
            at_ns: i * 7_200_000_000_000,
            event_id: None,
        });
    }

    let config = StatAnomalyConfig::default();
    let target_far = 0.05_f32;
    let far = calibrate_far_from_series(&hist, config, target_far)
        .ok_or_else(|| HostError::Runtime("FAR calibration failed".into()))?;
    let store = ScopedBaselineStore::from_series(&hist, config);
    let baseline = build_baseline(&hist, config);
    let scores = score_series_vs_baseline(&hist, &baseline, config);

    let smoke_pass = far.empirical_far <= target_far * 3.0
        && far.n_normal >= 20
        && store.subject_count() >= 1
        && store.source_count() >= 1;

    let mut md = String::from(
        "# Anomaly FAR + scoped baselines (synthetic)\n\n\
         > Scores are absolute z-values on a normal history window. Hosts replace \
         history with production windows and re-run `calibrate_far_from_series`.\n\n",
    );
    let _ = writeln!(md, "| Metric | Value |");
    let _ = writeln!(md, "| --- | ---: |");
    let _ = writeln!(md, "| Target FAR | {:.3} |", far.target_far);
    let _ = writeln!(md, "| Calibrated z-threshold | {:.3} |", far.threshold);
    let _ = writeln!(md, "| Empirical FAR | {:.3} |", far.empirical_far);
    let _ = writeln!(md, "| Normal scores | {} |", far.n_normal);
    let _ = writeln!(md, "| Score samples total | {} |", scores.len());
    let _ = writeln!(md, "| Subject scopes | {} |", store.subject_count());
    let _ = writeln!(md, "| Source scopes | {} |", store.source_count());
    let _ = writeln!(
        md,
        "| Smoke | {} |",
        if smoke_pass { "PASS" } else { "FAIL" }
    );
    let _ = writeln!(md, "\n## Host recipe\n");
    let _ = writeln!(
        md,
        "1. Collect a quiet history window into `AnalysisSeries`.\n\
         2. `calibrate_far_from_series(..., target_far)` → threshold.\n\
         3. `apply_anomaly_far` / `apply_far_to_stat_config` on the session.\n\
         4. `freeze_scoped_anomaly_baselines` + `detect_and_store_anomalies_scoped`.\n\
         5. Report empirical FAR on a hold-out normal day."
    );

    Ok(AnomalyEvidence {
        markdown: md,
        target_far: far.target_far,
        threshold: far.threshold,
        empirical_far: far.empirical_far,
        subject_scopes: store.subject_count(),
        source_scopes: store.source_count(),
        smoke_pass,
    })
}

pub(crate) fn write_anomaly_section(
    dir: &Path,
    anomaly: &AnomalyEvidence,
) -> Result<(), HostError> {
    let d = dir.join("anomaly");
    fs::create_dir_all(&d).map_err(|e| HostError::Io(e.to_string()))?;
    fs::write(d.join("far.md"), &anomaly.markdown).map_err(|e| HostError::Io(e.to_string()))?;
    Ok(())
}
