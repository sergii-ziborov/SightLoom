//! Re-id ROC / EER evidence section.

use crate::error::HostError;
use core::fmt::Write as _;
use serde::{Deserialize, Serialize};
use sightloom::ReidQualityReport;
use sightloom::reid::{LabeledScore, compute_roc};
use std::fs;
use std::path::Path;

/// Re-id calibration evidence (synthetic labeled scores by default).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReidEvidence {
    /// Markdown report.
    pub markdown: String,
    /// CSV of labeled scores: `score,genuine`.
    pub scores_csv: String,
    /// EER from calibration.
    pub eer: f32,
    /// Smoke gate pass.
    pub smoke_pass: bool,
    /// Genuine / impostor counts.
    pub genuine_count: u32,
    /// Impostor count.
    pub impostor_count: u32,
}

/// Builds synthetic well-separated genuine/impostor scores + ROC report.
///
/// # Errors
///
/// Calibration failures (should not fail for default synthetic set).
pub fn build_synthetic_reid_evidence() -> Result<ReidEvidence, HostError> {
    let mut scores = Vec::new();
    for i in 0..40 {
        scores.push(LabeledScore {
            score: 0.90 - (i as f32) * 0.002,
            genuine: true,
        });
        scores.push(LabeledScore {
            score: 0.20 + (i as f32) * 0.003,
            genuine: false,
        });
    }
    let report = compute_roc(&scores, 48).map_err(|e| HostError::Runtime(format!("roc: {e:?}")))?;
    let quality = ReidQualityReport::from_calibration(&report);
    let smoke_pass = quality.passes_smoke(0.15, 20);

    let mut csv = String::from("score,genuine\n");
    for s in &scores {
        let _ = writeln!(csv, "{:.6},{}", s.score, u8::from(s.genuine));
    }

    let mut md = String::from(
        "# Re-id ROC / EER evidence (synthetic)\n\n\
         > Synthetic well-separated pairs. Replace `scores.csv` with host gallery pairs \
         for real false-match / false-non-match rates.\n\n",
    );
    let _ = writeln!(md, "| Metric | Value |");
    let _ = writeln!(md, "| --- | ---: |");
    let _ = writeln!(md, "| Genuine pairs | {} |", report.genuine_count);
    let _ = writeln!(md, "| Impostor pairs | {} |", report.impostor_count);
    let _ = writeln!(md, "| EER | {:.4} |", report.eer);
    let _ = writeln!(md, "| EER threshold | {:.4} |", report.eer_threshold);
    let _ = writeln!(
        md,
        "| Recommended accept | {:.4} |",
        report.recommended_accept
    );
    let _ = writeln!(
        md,
        "| Recommended reject | {:.4} |",
        report.recommended_reject
    );
    let _ = writeln!(
        md,
        "| Smoke (EER≤0.15, n≥20) | {} |",
        if smoke_pass { "PASS" } else { "FAIL" }
    );
    let _ = writeln!(md, "\n## Host next steps\n");
    let _ = writeln!(
        md,
        "1. Export cosine pairs from your real gallery (same/different subject).\n\
         2. Feed `LabeledScore` into `compute_roc`.\n\
         3. Apply thresholds via `apply_identity_calibration`.\n\
         4. Report cross-camera / clothing / lighting slices separately."
    );

    Ok(ReidEvidence {
        markdown: md,
        scores_csv: csv,
        eer: report.eer,
        smoke_pass,
        genuine_count: report.genuine_count,
        impostor_count: report.impostor_count,
    })
}

pub(crate) fn write_reid_section(dir: &Path, reid: &ReidEvidence) -> Result<(), HostError> {
    let d = dir.join("reid");
    fs::create_dir_all(&d).map_err(|e| HostError::Io(e.to_string()))?;
    fs::write(d.join("roc.md"), &reid.markdown).map_err(|e| HostError::Io(e.to_string()))?;
    fs::write(d.join("scores.csv"), &reid.scores_csv).map_err(|e| HostError::Io(e.to_string()))?;
    Ok(())
}
