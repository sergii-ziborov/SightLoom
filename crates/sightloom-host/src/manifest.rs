//! Host **model manifest** — JSON inventory of weights (no weights in-repo).
//!
//! Hosts keep a `models.manifest.json` next to the cache dir, resolve paths
//! with a [`crate::ModelFetcher`], and load ONNX via [`crate::OnnxEmbedder`].

use crate::config::{HostBundleConfig, ModelSpec, ModelTask};
use crate::error::HostError;
use crate::registry::{ModelFetcher, ensure_cache_dir, write_cache_readme};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Versioned list of host models (download / local placement policy).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Schema version (currently `1`).
    #[serde(default = "default_manifest_version")]
    pub version: u32,
    /// Cache directory for resolved weights (relative or absolute).
    #[serde(default = "default_manifest_cache")]
    pub cache_dir: PathBuf,
    /// Models to materialize.
    #[serde(default)]
    pub models: Vec<ModelSpec>,
    /// Optional human notes (not machine-consumed).
    #[serde(default)]
    pub notes: Option<String>,
}

fn default_manifest_version() -> u32 {
    1
}

fn default_manifest_cache() -> PathBuf {
    PathBuf::from(".sightloom-models")
}

impl Default for ModelManifest {
    fn default() -> Self {
        Self {
            version: 1,
            cache_dir: default_manifest_cache(),
            models: Vec::new(),
            notes: None,
        }
    }
}

impl ModelManifest {
    /// Parses JSON text.
    ///
    /// # Errors
    ///
    /// Serde / schema errors.
    pub fn from_json(text: &str) -> Result<Self, HostError> {
        let m: Self = serde_json::from_str(text)
            .map_err(|e| HostError::Config(format!("manifest json: {e}")))?;
        if m.version == 0 {
            return Err(HostError::Config("manifest version must be >= 1".into()));
        }
        Ok(m)
    }

    /// Loads from a JSON file path.
    ///
    /// # Errors
    ///
    /// I/O or parse failures.
    pub fn load_path(path: &Path) -> Result<Self, HostError> {
        let text = fs::read_to_string(path).map_err(|e| HostError::Io(e.to_string()))?;
        Self::from_json(&text)
    }

    /// Pretty JSON.
    ///
    /// # Errors
    ///
    /// Serde errors.
    pub fn to_json_pretty(&self) -> Result<String, HostError> {
        serde_json::to_string_pretty(self).map_err(|e| HostError::Config(e.to_string()))
    }

    /// Writes JSON to `path` (parent dirs created).
    ///
    /// # Errors
    ///
    /// I/O failures.
    pub fn save_path(&self, path: &Path) -> Result<(), HostError> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent).map_err(|e| HostError::Io(e.to_string()))?;
        }
        fs::write(path, self.to_json_pretty()?).map_err(|e| HostError::Io(e.to_string()))
    }

    /// Example person re-id + person detect skeleton (uris empty — host fills in).
    #[must_use]
    pub fn example_person_bundle() -> Self {
        let mut person_reid = ModelSpec::embedder("person_reid", ModelTask::PersonReId, 512);
        person_reid.preprocess = crate::preprocess::PreprocessConfig::imagenet_like(128, 256);

        let mut person_detect = ModelSpec::detector("person_detect", ModelTask::PersonDetect);
        person_detect.preprocess = crate::preprocess::PreprocessConfig::imagenet_like(640, 640);

        Self {
            version: 1,
            cache_dir: PathBuf::from(".sightloom-models"),
            models: vec![person_detect, person_reid],
            notes: Some(
                "Fill ModelSpec.uri or place files as {id}.onnx under cache_dir. \
                 Optional sha256 is lowercase hex of the weight file."
                    .into(),
            ),
        }
    }

    /// Maps known tasks into a [`HostBundleConfig`] (first match wins per slot).
    #[must_use]
    pub fn to_bundle_config(&self) -> HostBundleConfig {
        let mut cfg = HostBundleConfig {
            person_detect: None,
            face_detect: None,
            person_reid: None,
            face_embed: None,
            segmentation: None,
            cache_dir: self.cache_dir.clone(),
            require_real_weights: true,
        };
        for spec in &self.models {
            match spec.task {
                ModelTask::PersonDetect if cfg.person_detect.is_none() => {
                    cfg.person_detect = Some(spec.clone());
                }
                ModelTask::FaceDetect if cfg.face_detect.is_none() => {
                    cfg.face_detect = Some(spec.clone());
                }
                ModelTask::PersonReId if cfg.person_reid.is_none() => {
                    cfg.person_reid = Some(spec.clone());
                }
                ModelTask::FaceEmbed if cfg.face_embed.is_none() => {
                    cfg.face_embed = Some(spec.clone());
                }
                ModelTask::Segmentation if cfg.segmentation.is_none() => {
                    cfg.segmentation = Some(spec.clone());
                }
                _ => {}
            }
        }
        cfg
    }

    /// Ensures every model is local via `fetcher` (download or filesystem).
    ///
    /// Returns resolved paths in manifest order. Verifies `sha256` when set.
    ///
    /// # Errors
    ///
    /// Fetch / integrity / I/O failures.
    pub fn ensure_all(&self, fetcher: &mut dyn ModelFetcher) -> Result<Vec<PathBuf>, HostError> {
        ensure_cache_dir(&self.cache_dir)?;
        let _ = write_cache_readme(&self.cache_dir);
        let mut paths = Vec::with_capacity(self.models.len());
        for spec in &self.models {
            let path = fetcher.ensure_local(spec, &self.cache_dir)?;
            crate::integrity::maybe_verify_sha256(path.as_path(), spec.sha256.as_deref())?;
            paths.push(path);
        }
        Ok(paths)
    }

    /// Spec by id.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&ModelSpec> {
        self.models.iter().find(|m| m.id == id)
    }
}

/// Result of materializing one model for logging / evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedModel {
    /// Spec id.
    pub id: String,
    /// Absolute or relative path on disk.
    pub path: PathBuf,
    /// Whether sha256 was checked.
    pub verified: bool,
}

/// Resolves all models and returns structured results.
///
/// # Errors
///
/// Same as [`ModelManifest::ensure_all`].
pub fn resolve_manifest(
    manifest: &ModelManifest,
    fetcher: &mut dyn ModelFetcher,
) -> Result<Vec<ResolvedModel>, HostError> {
    let paths = manifest.ensure_all(fetcher)?;
    let mut out = Vec::with_capacity(paths.len());
    for (spec, path) in manifest.models.iter().zip(paths) {
        out.push(ResolvedModel {
            id: spec.id.clone(),
            path,
            verified: spec.sha256.is_some(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::FilesystemFetcher;
    use std::io::Write;

    #[test]
    fn example_roundtrip_json() {
        let m = ModelManifest::example_person_bundle();
        let text = m.to_json_pretty().unwrap();
        let back = ModelManifest::from_json(&text).unwrap();
        assert_eq!(back.models.len(), 2);
        assert_eq!(back.version, 1);
        let bundle = back.to_bundle_config();
        assert!(bundle.person_reid.is_some());
        assert!(bundle.person_detect.is_some());
        assert!(bundle.require_real_weights);
    }

    #[test]
    fn ensure_all_local_and_sha256() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("cache");
        fs::create_dir_all(&cache).unwrap();
        let weights = cache.join("toy.onnx");
        {
            let mut f = fs::File::create(&weights).unwrap();
            f.write_all(b"fake-onnx-bytes").unwrap();
        }
        let digest = crate::integrity::file_sha256_hex(&weights).unwrap();

        let mut spec = ModelSpec::embedder("toy", ModelTask::PersonReId, 8);
        spec.sha256 = Some(digest);
        let manifest = ModelManifest {
            version: 1,
            cache_dir: cache.clone(),
            models: vec![spec],
            notes: None,
        };
        let paths = manifest.ensure_all(&mut FilesystemFetcher).unwrap();
        assert_eq!(paths.len(), 1);
        assert!(paths[0].is_file());
    }

    #[test]
    fn ensure_all_rejects_bad_digest() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("cache");
        fs::create_dir_all(&cache).unwrap();
        let weights = cache.join("toy.onnx");
        fs::write(&weights, b"x").unwrap();
        let mut spec = ModelSpec::embedder("toy", ModelTask::PersonReId, 8);
        spec.sha256 =
            Some("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".into());
        let manifest = ModelManifest {
            version: 1,
            cache_dir: cache,
            models: vec![spec],
            notes: None,
        };
        let err = manifest.ensure_all(&mut FilesystemFetcher).unwrap_err();
        assert!(matches!(err, HostError::Integrity(_)));
    }
}
