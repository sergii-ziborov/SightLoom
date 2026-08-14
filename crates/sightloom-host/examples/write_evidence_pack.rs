//! Step-3: write a synthetic MOT / re-id / redaction evidence pack to disk.
//!
//! ```bash
//! cargo run -p sightloom-host --example write_evidence_pack
//! cargo run -p sightloom-host --example write_evidence_pack -- ./my-evidence
//! ```

use sightloom::tracking::ByteTrackConfig;
use sightloom_host::{build_synthetic_evidence_pack, write_evidence_pack};
use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = env::args()
        .nth(1)
        .map_or_else(|| PathBuf::from("evidence-out"), PathBuf::from);

    println!("sightloom-host step 3 — synthetic evidence pack");
    println!("output: {}", out.display());

    let pack = build_synthetic_evidence_pack("synthetic-default", &ByteTrackConfig::default())?;
    let paths = write_evidence_pack(&pack, &out)?;

    println!(
        "smoke: {}",
        if pack.all_smoke_pass() {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!("wrote {}", paths.summary.display());
    println!("wrote {}", paths.manifest.display());
    println!("  mot/suite.md, mot/*_gt.txt, mot/*_hyp.txt, mot/TRACK_EVAL.md");
    println!("  reid/roc.md, reid/scores.csv");
    println!("  redaction/report.md, redaction/samples.json");
    println!("ok — harness ready; attach real TrackEval / gallery pairs for product claims");
    Ok(())
}
