//! Local model cache / path resolution (+ optional HTTP download).

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
        let path = resolve_filesystem(spec, cache_dir)?;
        crate::integrity::maybe_verify_sha256(path.as_path(), spec.sha256.as_deref())?;
        Ok(path)
    }
}

fn resolve_filesystem(spec: &ModelSpec, cache_dir: &Path) -> Result<PathBuf, HostError> {
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
        "no weights for '{}' under {} (uri={:?}). Place an ONNX file, set local_path, or enable feature `download` with uri.",
        spec.id,
        cache_dir.display(),
        spec.uri
    )))
}

/// Fetcher that records requested downloads but does not fetch (offline stub).
#[derive(Clone, Debug, Default)]
pub struct DeferredDownloadFetcher {
    /// URIs that would be downloaded when a real fetcher is plugged in.
    pub pending: Vec<String>,
}

impl ModelFetcher for DeferredDownloadFetcher {
    fn ensure_local(&mut self, spec: &ModelSpec, cache_dir: &Path) -> Result<PathBuf, HostError> {
        match resolve_filesystem(spec, cache_dir) {
            Ok(p) => {
                crate::integrity::maybe_verify_sha256(p.as_path(), spec.sha256.as_deref())?;
                Ok(p)
            }
            Err(e) => {
                if let Some(uri) = &spec.uri {
                    self.pending.push(uri.clone());
                }
                Err(e)
            }
        }
    }
}

/// HTTP(S) fetcher: local first, then `ModelSpec.uri` into the cache (feature `download`).
///
/// **Security:** only enable for trusted config; does not follow auth secrets
/// from the environment unless the host passes a custom agent later.
#[cfg(feature = "download")]
#[derive(Clone, Debug, Default)]
pub struct HttpModelFetcher {
    /// Optional User-Agent.
    pub user_agent: Option<String>,
}

#[cfg(feature = "download")]
impl ModelFetcher for HttpModelFetcher {
    fn ensure_local(&mut self, spec: &ModelSpec, cache_dir: &Path) -> Result<PathBuf, HostError> {
        if let Ok(p) = resolve_filesystem(spec, cache_dir) {
            crate::integrity::maybe_verify_sha256(p.as_path(), spec.sha256.as_deref())?;
            return Ok(p);
        }
        let Some(uri) = spec.uri.as_ref() else {
            return FilesystemFetcher.ensure_local(spec, cache_dir);
        };
        if !(uri.starts_with("https://") || uri.starts_with("http://")) {
            return Err(HostError::Download(format!(
                "unsupported uri scheme (need http/https): {uri}"
            )));
        }
        ensure_cache_dir(cache_dir)?;
        let dest = cache_dir.join(&spec.id).with_extension(
            spec.format
                .as_deref()
                .unwrap_or("onnx")
                .trim_start_matches('.'),
        );
        let agent = ureq::AgentBuilder::new()
            .user_agent(
                self.user_agent
                    .as_deref()
                    .unwrap_or("sightloom-host/0.1 (model fetch)"),
            )
            .build();
        let response = agent
            .get(uri)
            .call()
            .map_err(|e| HostError::Download(format!("GET {uri}: {e}")))?;
        if !(200..300).contains(&response.status()) {
            return Err(HostError::Download(format!(
                "GET {uri}: HTTP {}",
                response.status()
            )));
        }
        let mut reader = response.into_reader();
        let tmp = dest.with_extension("part");
        {
            use std::io::Write as _;
            let mut file = fs::File::create(&tmp).map_err(|e| HostError::Io(e.to_string()))?;
            std::io::copy(&mut reader, &mut file).map_err(|e| HostError::Io(e.to_string()))?;
            file.flush().map_err(|e| HostError::Io(e.to_string()))?;
        }
        fs::rename(&tmp, &dest).map_err(|e| HostError::Io(e.to_string()))?;
        crate::integrity::maybe_verify_sha256(dest.as_path(), spec.sha256.as_deref())?;
        Ok(dest)
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

Place ONNX (or other runtime) weight files here, named by ModelSpec.id:

  person_detect.onnx   — NCHW RGB f32 → YOLO / N×6 boxes
  person_reid.onnx     — NCHW RGB f32 → embedding (L2 in OnnxEmbedder)
  face_embed.onnx      — optional, same embed contract (typically 112×112)

Recommended host pair (not shipped, license is yours):
  YOLOv8n (person class) + OSNet x1.0 (512-d)

Or set ModelSpec.local_path / uri in models.manifest.json.

Features:
  onnx          — OnnxEmbedder / OnnxDetector (tract pure-Rust)
  download      — HttpModelFetcher pulls ModelSpec.uri (http/https)
  image-decode  — JPEG/PNG → RGB for encoded PhotoView

Optional ModelSpec.sha256 (hex) is verified after resolve/download.
See crates/sightloom-host/COOKBOOK.md and ModelManifest JSON.

Never commit weight files to the SightLoom repo.
";
    fs::write(cache_dir.join("README.txt"), text).map_err(|e| HostError::Io(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelSpec, ModelTask};

    fn embed_spec(id: &str) -> ModelSpec {
        ModelSpec::embedder(id, ModelTask::PersonReId, 64)
    }

    #[test]
    fn filesystem_fetcher_uses_local_path() {
        let dir = tempfile::tempdir().unwrap();
        let weights = dir.path().join("weights.onnx");
        fs::write(&weights, b"fake-onnx").unwrap();
        let mut spec = embed_spec("person");
        spec.local_path = Some(weights.clone());
        let path = FilesystemFetcher.ensure_local(&spec, dir.path()).unwrap();
        assert_eq!(path, weights);
    }

    #[test]
    fn deferred_records_pending_uri() {
        let dir = tempfile::tempdir().unwrap();
        let mut fetcher = DeferredDownloadFetcher::default();
        let mut spec = embed_spec("missing");
        spec.uri = Some("https://example.com/model.onnx".into());
        let err = fetcher.ensure_local(&spec, dir.path()).unwrap_err();
        assert!(matches!(err, HostError::ModelNotFound(_)));
        assert_eq!(fetcher.pending.len(), 1);
    }

    #[cfg(feature = "download")]
    #[test]
    fn http_fetcher_rejects_non_http_scheme() {
        let dir = tempfile::tempdir().unwrap();
        let mut fetcher = HttpModelFetcher::default();
        let mut spec = embed_spec("bad");
        spec.uri = Some("file:///tmp/x.onnx".into());
        let err = fetcher.ensure_local(&spec, dir.path()).unwrap_err();
        assert!(matches!(err, HostError::Download(_)));
    }

    #[cfg(feature = "download")]
    #[test]
    fn http_fetcher_prefers_existing_local_file() {
        let dir = tempfile::tempdir().unwrap();
        let weights = dir.path().join("cached.onnx");
        fs::write(&weights, b"local-wins").unwrap();
        let mut spec = embed_spec("cached");
        spec.local_path = Some(weights.clone());
        // Would fail if network were attempted.
        spec.uri = Some("https://127.0.0.1:1/unreachable.onnx".into());
        let path = HttpModelFetcher::default()
            .ensure_local(&spec, dir.path())
            .unwrap();
        assert_eq!(path, weights);
    }
}
