//! Model specs and host bundle configuration.

use crate::device::DevicePreference;
use crate::preprocess::PreprocessConfig;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// High-level model task in the host package.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelTask {
    /// Face detector.
    FaceDetect,
    /// Person / body detector.
    PersonDetect,
    /// Face embedding (recognition).
    FaceEmbed,
    /// Person re-identification embedding.
    PersonReId,
    /// Instance / semantic segmentation (masks for host → `VisionIndex`).
    Segmentation,
}

impl ModelTask {
    /// Stable string id.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FaceDetect => "face_detect",
            Self::PersonDetect => "person_detect",
            Self::FaceEmbed => "face_embed",
            Self::PersonReId => "person_reid",
            Self::Segmentation => "segmentation",
        }
    }
}

/// One model the host may load (weights **not** shipped in `SightLoom` core).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Stable id (`"person_yolo_v8n"`, `"osnet_x1_0"`, …).
    pub id: String,
    /// Task this model performs.
    pub task: ModelTask,
    /// Optional remote URI for download (https / s3 / …). Host fetcher only.
    #[serde(default)]
    pub uri: Option<String>,
    /// Local path when already on disk.
    #[serde(default)]
    pub local_path: Option<PathBuf>,
    /// Expected embedding length (`0` for detectors / segmenters).
    #[serde(default)]
    pub embedding_dim: usize,
    /// Preprocess knobs.
    #[serde(default)]
    pub preprocess: PreprocessConfig,
    /// Device preference.
    #[serde(default)]
    pub device: DevicePreference,
    /// Optional model format hint (`"onnx"`, `"torchscript"`, `"openvino"`, …).
    #[serde(default)]
    pub format: Option<String>,
}

impl ModelSpec {
    /// Detector-oriented constructor.
    #[must_use]
    pub fn detector(id: impl Into<String>, task: ModelTask) -> Self {
        Self {
            id: id.into(),
            task,
            uri: None,
            local_path: None,
            embedding_dim: 0,
            preprocess: PreprocessConfig::default(),
            device: DevicePreference::Auto,
            format: Some("onnx".into()),
        }
    }

    /// Embedding-oriented constructor.
    #[must_use]
    pub fn embedder(id: impl Into<String>, task: ModelTask, dim: usize) -> Self {
        Self {
            id: id.into(),
            task,
            uri: None,
            local_path: None,
            embedding_dim: dim,
            preprocess: PreprocessConfig::imagenet_like(256, 128),
            device: DevicePreference::Auto,
            format: Some("onnx".into()),
        }
    }
}

/// Full host bundle: which models participate in photo search / ingest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HostBundleConfig {
    /// Person detector (ingest path).
    #[serde(default)]
    pub person_detect: Option<ModelSpec>,
    /// Face detector (optional crop path).
    #[serde(default)]
    pub face_detect: Option<ModelSpec>,
    /// Person re-id embedder.
    #[serde(default)]
    pub person_reid: Option<ModelSpec>,
    /// Face embedder.
    #[serde(default)]
    pub face_embed: Option<ModelSpec>,
    /// Segmentation model (optional masks).
    #[serde(default)]
    pub segmentation: Option<ModelSpec>,
    /// Root directory for downloaded / cached weights.
    #[serde(default = "default_cache_dir")]
    pub cache_dir: PathBuf,
    /// When true, missing local models are an error (no silent fake).
    #[serde(default)]
    pub require_real_weights: bool,
}

fn default_cache_dir() -> PathBuf {
    PathBuf::from(".sightloom-models")
}

impl Default for HostBundleConfig {
    fn default() -> Self {
        Self {
            person_detect: Some(ModelSpec::detector(
                "ref_person_detect",
                ModelTask::PersonDetect,
            )),
            face_detect: Some(ModelSpec::detector(
                "ref_face_detect",
                ModelTask::FaceDetect,
            )),
            person_reid: Some(ModelSpec::embedder(
                "ref_person_reid",
                ModelTask::PersonReId,
                128,
            )),
            face_embed: Some(ModelSpec::embedder(
                "ref_face_embed",
                ModelTask::FaceEmbed,
                128,
            )),
            segmentation: None,
            cache_dir: default_cache_dir(),
            require_real_weights: false,
        }
    }
}

impl HostBundleConfig {
    /// Parses JSON host config.
    ///
    /// # Errors
    ///
    /// Serde errors.
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    /// Serializes to pretty JSON.
    ///
    /// # Errors
    ///
    /// Serde errors.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// All configured specs in a stable order.
    #[must_use]
    pub fn all_specs(&self) -> Vec<&ModelSpec> {
        [
            self.person_detect.as_ref(),
            self.face_detect.as_ref(),
            self.person_reid.as_ref(),
            self.face_embed.as_ref(),
            self.segmentation.as_ref(),
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}
