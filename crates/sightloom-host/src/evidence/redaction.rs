//! Redaction pixel evidence section.

use crate::error::HostError;
use core::fmt::Write as _;
use serde::{Deserialize, Serialize};
use sightloom::{RedactionPixelSample, RedactionQualityReport, evaluate_redaction_pixels};
use std::fs;
use std::path::Path;

/// Redaction quality evidence (host supplies real pixel counts for production).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RedactionEvidence {
    /// Markdown report.
    pub markdown: String,
    /// JSON array of samples.
    pub samples_json: String,
    /// Aggregated report.
    pub mean_target_leakage: f32,
    /// Mean collateral redaction ratio.
    pub mean_collateral_ratio: f32,
    /// Smoke pass (low leakage + low collateral).
    pub smoke_pass: bool,
}

/// Synthetic pixel samples: mostly clean redaction with one mild leak.
#[must_use]
pub fn build_synthetic_redaction_evidence() -> RedactionEvidence {
    let samples = vec![
        RedactionPixelSample {
            interval_id: 1,
            target_pixels: 10_000,
            target_visible_pixels: 50,
            collateral_redacted_pixels: 200,
            non_target_pixels: 100_000,
        },
        RedactionPixelSample {
            interval_id: 2,
            target_pixels: 8_000,
            target_visible_pixels: 0,
            collateral_redacted_pixels: 100,
            non_target_pixels: 90_000,
        },
        RedactionPixelSample {
            interval_id: 3,
            target_pixels: 12_000,
            target_visible_pixels: 120,
            collateral_redacted_pixels: 400,
            non_target_pixels: 110_000,
        },
    ];
    let report = evaluate_redaction_pixels(&samples);
    let smoke_pass = report.mean_target_leakage <= 0.05 && report.mean_collateral_ratio <= 0.02;

    let samples_json =
        serde_json::to_string_pretty(&samples_dto(&samples)).unwrap_or_else(|_| "[]".into());

    let mut md = String::from(
        "# Redaction quality evidence (synthetic host pixels)\n\n\
         > `SightLoom` stores redaction **intervals** only. Pixel leakage is measured \
         by the **host render** after blur. These samples are synthetic placeholders.\n\n",
    );
    let _ = writeln!(md, "| Metric | Value |");
    let _ = writeln!(md, "| --- | ---: |");
    let _ = writeln!(md, "| Samples | {} |", report.samples);
    let _ = writeln!(
        md,
        "| Mean target leakage | {:.4} |",
        report.mean_target_leakage
    );
    let _ = writeln!(
        md,
        "| Mean collateral ratio | {:.4} |",
        report.mean_collateral_ratio
    );
    let _ = writeln!(
        md,
        "| Smoke (leak≤5%, collateral≤2%) | {} |",
        if smoke_pass { "PASS" } else { "FAIL" }
    );
    let _ = writeln!(md, "\n## Host measurement recipe\n");
    let _ = writeln!(
        md,
        "1. Export redaction intervals from `IndexSession`.\n\
         2. Render blur / box overlays in the host.\n\
         3. Count target vs non-target pixels (mask or bbox proxy).\n\
         4. Fill `RedactionPixelSample` and call `evaluate_redaction_pixels`.\n\
         5. Track uncertain-identity holds separately (hold ≠ redacted)."
    );

    RedactionEvidence {
        markdown: md,
        samples_json,
        mean_target_leakage: report.mean_target_leakage,
        mean_collateral_ratio: report.mean_collateral_ratio,
        smoke_pass,
    }
}

#[derive(Serialize)]
struct SampleDto {
    interval_id: u64,
    target_pixels: u64,
    target_visible_pixels: u64,
    collateral_redacted_pixels: u64,
    non_target_pixels: u64,
}

fn samples_dto(samples: &[RedactionPixelSample]) -> Vec<SampleDto> {
    samples
        .iter()
        .map(|s| SampleDto {
            interval_id: s.interval_id,
            target_pixels: s.target_pixels,
            target_visible_pixels: s.target_visible_pixels,
            collateral_redacted_pixels: s.collateral_redacted_pixels,
            non_target_pixels: s.non_target_pixels,
        })
        .collect()
}

pub(crate) fn write_redaction_section(
    dir: &Path,
    redaction: &RedactionEvidence,
) -> Result<(), HostError> {
    let d = dir.join("redaction");
    fs::create_dir_all(&d).map_err(|e| HostError::Io(e.to_string()))?;
    fs::write(d.join("report.md"), &redaction.markdown)
        .map_err(|e| HostError::Io(e.to_string()))?;
    fs::write(d.join("samples.json"), &redaction.samples_json)
        .map_err(|e| HostError::Io(e.to_string()))?;
    let _ = RedactionQualityReport::default();
    Ok(())
}
