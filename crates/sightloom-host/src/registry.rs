//! Local model cache / path resolution (download hooks without network in step 1).

use crate::config::ModelSpec;
use crate::error::HostError;
use std::fs;
use std::path::{Path, PathBuf};

/// Resolves a [`ModelSpec`] to a local file path.
pub trait ModelFetcher {
    /// Ensures weights exist on disk and returns the path.
    ///
    /// # Errors
    ///
    /// Missing files / download failures.
    fn ensure_local(&mut self, spec: &ModelSpec, cache_dir: &Path) -> Result<PathBuf, HostError>;
}

/// Only uses `local_path` or `cache_dir / id`; never hits the network.
#[derive(Clone, Debug, Default)]
pub struct FilesystemFetcher;

impl ModelFetcher for FilesystemFetcher {
    fn ensure_local(&mut self, spec: &ModelSpec, cache_dir: &Path) -> Result<PathBuf, HostError> {
        if let Some(p) = &spec.local_path {
            if p.is_file() {
                return Ok(p.clone());
            }
            return Err(HostError::ModelNotFound(format!(
                "local_path missing: {}",
                p.display()
            )));
        }
        let candidate = cache_dir.join(&spec.id).with_extension(
            spec.format
                .as_deref()
                .unwrap_or("onnx")
                .trim_start_matches('.'),
        );
        if candidate.is_file() {
            return Ok(candidate);
        }
        // Also try bare id file.
        let bare = cache_dir.join(&spec.id);
        if bare.is_file() {
            return Ok(bare);
        }
        Err(HostError::ModelNotFound(format!(
            "no weights for '{}' under {} (uri={:?}). Place an ONNX file or set local_path. Step 1 has no network download.",
            spec.id,
            cache_dir.display(),
            spec.uri
        )))
    }
}

/// Fetcher that records requested downloads but does not fetch (step 1 stub).
#[derive(Clone, Debug, Default)]
pub struct DeferredDownloadFetcher {
    /// URIs that would be downloaded when a real fetcher is plugged in.
    pub pending: Vec<String>,
}

impl ModelFetcher for DeferredDownloadFetcher {
    fn ensure_local(&mut self, spec: &ModelSpec, cache_dir: &Path) -> Result<PathBuf, HostError> {
        match FilesystemFetcher.ensure_local(spec, cache_dir) {
            Ok(p) => Ok(p),
            Err(e) => {
                if let Some(uri) = &spec.uri {
                    self.pending.push(uri.clone());
                }
                Err(e)
            }
        }
    }
}

/// Creates cache directory if needed.
///
/// # Errors
///
/// I/O failures.
pub fn ensure_cache_dir(path: &Path) -> Result<(), HostError> {
    fs::create_dir_all(path).map_err(|e| HostError::Io(e.to_string()))
}

/// Writes a placeholder marker file documenting where real weights go.
///
/// # Errors
///
/// I/O failures.
pub fn write_cache_readme(cache_dir: &Path) -> Result<(), HostError> {
    ensure_cache_dir(cache_dir)?;
    let text = "\
SightLoom host model cache
==========================

Place ONNX (or other runtime) weight files here, named by ModelSpec.id, e.g.:

  ref_person_detect.onnx
  ref_person_reid.onnx

Or set ModelSpec.local_path / uri in HostBundleConfig JSON.

`sightloom-host` step 1 does not download weights. A future fetcher may pull
ModelSpec.uri into this directory.
";
    fs::write(cache_dir.join("README.txt"), text).map_err(|e| HostError::Io(e.to_string()))
}
