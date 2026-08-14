//! Full evidence pack build + filesystem layout.

use super::mot::{MotEvidence, build_synthetic_mot_evidence, write_mot_section};
use super::redaction::{
    RedactionEvidence, build_synthetic_redaction_evidence, write_redaction_section,
};
use super::reid::{ReidEvidence, build_synthetic_reid_evidence, write_reid_section};
use crate::error::HostError;
use serde::{Deserialize, Serialize};
use sightloom::tracking::ByteTrackConfig;
use std::fs;
use std::path::{Path, PathBuf};

/// Complete host evidence pack (synthetic defaults + extension points).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvidencePack {
    /// Pack name / tag.
    pub name: String,
    /// Schema version for host tools.
    pub schema_version: u32,
    /// MOT section.
    pub mot: MotEvidence,
    /// Re-id section.
    pub reid: ReidEvidence,
    /// Redaction section.
    pub redaction: RedactionEvidence,
}

impl EvidencePack {
    /// True when all synthetic smoke gates pass.
    #[must_use]
    pub fn all_smoke_pass(&self) -> bool {
        self.mot.all_smoke_pass && self.reid.smoke_pass && self.redaction.smoke_pass
    }

    /// Markdown summary for the pack root.
    #[must_use]
    pub fn summary_markdown(&self) -> String {
        use core::fmt::Write as _;
        let mut out = String::from("# SightLoom evidence pack\n\n");
        let _ = writeln!(out, "- **name**: `{}`", self.name);
        let _ = writeln!(out, "- **schema**: {}", self.schema_version);
        let _ = writeln!(
            out,
            "- **overall smoke**: {}",
            if self.all_smoke_pass() {
                "PASS"
            } else {
                "FAIL"
            }
        );
        let _ = writeln!(out, "\n## Sections\n");
        let _ = writeln!(out, "| Section | Smoke | Notes |");
        let _ = writeln!(out, "| --- | --- | --- |");
        let _ = writeln!(
            out,
            "| MOT | {} | Synthetic CLEAR + MOTChallenge export; not MOT17 |",
            pass(self.mot.all_smoke_pass)
        );
        let _ = writeln!(
            out,
            "| Re-id | {} | Synthetic ROC/EER; replace scores.csv for real gallery |",
            pass(self.reid.smoke_pass)
        );
        let _ = writeln!(
            out,
            "| Redaction | {} | Synthetic host pixel counts; measure after render |",
            pass(self.redaction.smoke_pass)
        );
        let _ = writeln!(
            out,
            "\n## Layout\n\n\
             - `SUMMARY.md` — this file\n\
             - `manifest.json` — machine-readable pack\n\
             - `mot/` — suite.md, MOTChallenge gt/hyp, TRACK_EVAL.md\n\
             - `reid/` — roc.md, scores.csv\n\
             - `redaction/` — report.md, samples.json\n"
        );
        let _ = writeln!(
            out,
            "## Honest boundary\n\n\
             This pack proves the **evaluation harness**. Published leaderboard \
             scores require host datasets + external evaluators (TrackEval, etc.).\n"
        );
        out
    }
}

fn pass(ok: bool) -> &'static str {
    if ok { "PASS" } else { "FAIL" }
}

/// Paths written by [`write_evidence_pack`].
#[derive(Clone, Debug)]
pub struct EvidencePackPaths {
    /// Root directory.
    pub root: PathBuf,
    /// Summary markdown.
    pub summary: PathBuf,
    /// Manifest JSON.
    pub manifest: PathBuf,
}

/// Builds the default synthetic evidence pack.
///
/// # Errors
///
/// Tracker / calibration failures.
pub fn build_synthetic_evidence_pack(
    name: impl Into<String>,
    track_config: &ByteTrackConfig,
) -> Result<EvidencePack, HostError> {
    Ok(EvidencePack {
        name: name.into(),
        schema_version: 1,
        mot: build_synthetic_mot_evidence(track_config)?,
        reid: build_synthetic_reid_evidence()?,
        redaction: build_synthetic_redaction_evidence(),
    })
}

/// Writes the pack under `dir` (created if needed).
///
/// # Errors
///
/// I/O / serde failures.
pub fn write_evidence_pack(
    pack: &EvidencePack,
    dir: impl AsRef<Path>,
) -> Result<EvidencePackPaths, HostError> {
    let root = dir.as_ref();
    fs::create_dir_all(root).map_err(|e| HostError::Io(e.to_string()))?;

    let summary = root.join("SUMMARY.md");
    fs::write(&summary, pack.summary_markdown()).map_err(|e| HostError::Io(e.to_string()))?;

    let manifest = root.join("manifest.json");
    let json = serde_json::to_string_pretty(pack)
        .map_err(|e| HostError::Runtime(format!("manifest: {e}")))?;
    fs::write(&manifest, json).map_err(|e| HostError::Io(e.to_string()))?;

    write_mot_section(root, &pack.mot)?;
    write_reid_section(root, &pack.reid)?;
    write_redaction_section(root, &pack.redaction)?;

    Ok(EvidencePackPaths {
        root: root.to_path_buf(),
        summary,
        manifest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn synthetic_pack_writes_and_passes_smoke() {
        let pack = build_synthetic_evidence_pack("test", &ByteTrackConfig::default()).unwrap();
        assert!(pack.all_smoke_pass(), "reid eer={}", pack.reid.eer);
        let dir = tempdir().unwrap();
        let paths = write_evidence_pack(&pack, dir.path()).unwrap();
        assert!(paths.summary.is_file());
        assert!(paths.manifest.is_file());
        assert!(dir.path().join("mot/suite.md").is_file());
        assert!(dir.path().join("mot/parallel_walk_gt.txt").is_file());
        assert!(dir.path().join("reid/scores.csv").is_file());
        assert!(dir.path().join("redaction/samples.json").is_file());
        assert!(dir.path().join("mot/TRACK_EVAL.md").is_file());
    }
}
