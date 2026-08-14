//! Scaffold a host weights layout: cache dir + example manifest + README.
//!
//! ```bash
//! cargo run -p sightloom-host --example weights_cookbook -- ./host-models
//! ```
//!
//! Does **not** download real networks. Fill `uri` / drop ONNX files, then use
//! `ModelManifest::ensure_all` (filesystem or `HttpModelFetcher`).

use sightloom_host::{
    FilesystemFetcher, ModelManifest, resolve_manifest, write_cache_readme,
};
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let out = env::args()
        .nth(1)
        .map_or_else(|| PathBuf::from("./host-models"), PathBuf::from);

    if let Err(e) = run(&out) {
        eprintln!("weights_cookbook: {e}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

fn run(out: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(out)?;
    let cache = out.join(".sightloom-models");
    write_cache_readme(&cache)?;

    let mut manifest = ModelManifest::example_person_bundle();
    manifest.cache_dir = cache;
    let cache = &manifest.cache_dir;
    let manifest_path = out.join("models.manifest.json");
    manifest.save_path(&manifest_path)?;

    // Offline resolve: expects files only if present; report status per model.
    println!("sightloom-host weights cookbook");
    println!("  output:     {}", out.display());
    println!("  cache:      {}", cache.display());
    println!("  manifest:   {}", manifest_path.display());
    println!();
    println!("Next steps:");
    println!("  1. Edit models.manifest.json — set ModelSpec.uri and/or sha256");
    println!("  2. Place {{id}}.onnx under the cache dir, or enable feature download");
    println!("  3. cargo run -p sightloom-host --features onnx --example onnx_photo_search");
    println!("  4. Read crates/sightloom-host/COOKBOOK.md");
    println!();

    match resolve_manifest(&manifest, &mut FilesystemFetcher) {
        Ok(resolved) => {
            println!("resolved {} model(s):", resolved.len());
            for r in resolved {
                println!(
                    "  ✓ {} → {} (sha256 verified={})",
                    r.id,
                    r.path.display(),
                    r.verified
                );
            }
        }
        Err(e) => {
            println!("not all models on disk yet (expected for a fresh scaffold):");
            println!("  {e}");
            println!("  (exit 0 — scaffold succeeded; weights are host-supplied)");
        }
    }

    // Also emit a HostBundleConfig snapshot for hosts that prefer that shape.
    let bundle = manifest.to_bundle_config();
    let bundle_path = out.join("host_bundle.example.json");
    std::fs::write(&bundle_path, bundle.to_json_pretty()?)?;
    println!("  bundle cfg: {}", bundle_path.display());

    Ok(())
}
