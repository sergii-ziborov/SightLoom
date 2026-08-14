//! Host model package for [`sightloom`](https://docs.rs/sightloom).
//!
//! # Product boundary
//!
//! `SightLoom` **does not** turn photos into embeddings. This crate is the
//! host side of the killer path:
//!
//! ```text
//! photo / frame
//!   → host detect / embed (reference or ONNX)
//!   → SightLoom IndexSession (rank / memory / tracks)
//! ```
//!
//! Weights, GPU runtimes, and model download policies live **here** (or in a
//! private host binary), never inside `sightloom-core`.
//!
//! # Features
//!
//! | Feature | What |
//! | --- | --- |
//! | `std` (default) | host package + reference models |
//! | `onnx` | tract ONNX backends ([`OnnxEmbedder`], [`OnnxDetector`]) |
//! | `download` | [`HttpModelFetcher`] for `ModelSpec.uri` |
//! | `image-decode` | JPEG/PNG → RGB for encoded photos |
//! | `full` | `onnx` + `download` + `image-decode` |
//!
//! # Steps
//!
//! 1. Config, preprocess, reference models, [`HostPipeline`]
//! 2. **ONNX** load from cache / `ModelSpec.local_path`
//! 3. **Evidence packs** — MOT / re-id / redaction / anomaly FAR
//! 4. FAR + scoped anomaly baselines
//! 5. **Download + image decode** + analysis day-of-week seasonality
//! 6. **Weights cookbook** — [`ModelManifest`], SHA-256 integrity, host docs
//! 7. **`TrackEval` bridge** — parse/export `MOTChallenge`, import host summaries

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

mod config;
mod decode;
mod device;
mod error;
pub mod evidence;
mod integrity;
mod manifest;
#[cfg(feature = "onnx")]
mod onnx_backend;
mod pipeline;
mod preprocess;
mod reference;
mod registry;

pub use config::{HostBundleConfig, ModelSpec, ModelTask};
pub use decode::{DecodedRgb, decode_encoded_rgb, decode_photo_rgb};
pub use device::DevicePreference;
pub use error::HostError;
pub use evidence::{
    AnomalyEvidence, EvidencePack, EvidencePackPaths, MotEvidence, RedactionEvidence, ReidEvidence,
    build_synthetic_anomaly_evidence, build_synthetic_evidence_pack, build_synthetic_mot_evidence,
    build_synthetic_redaction_evidence, build_synthetic_reid_evidence, write_evidence_pack,
};
pub use integrity::{file_sha256_hex, maybe_verify_sha256, verify_file_sha256};
pub use manifest::{ModelManifest, ResolvedModel, resolve_manifest};
#[cfg(feature = "onnx")]
pub use onnx_backend::{OnnxDetector, OnnxEmbedder, OnnxModel};
pub use pipeline::HostPipeline;
pub use preprocess::{
    PreprocessConfig, crop_rgb8, prepare_rgb8_nchw, resize_rgb8_nearest, rgb8_to_chw_f32,
};
pub use reference::{
    ReferenceEmbedder, ReferenceFaceDetector, ReferenceHostModels, ReferencePersonDetector,
    frame_to_rgb8,
};
#[cfg(feature = "download")]
pub use registry::HttpModelFetcher;
pub use registry::{
    DeferredDownloadFetcher, FilesystemFetcher, ModelFetcher, ensure_cache_dir, write_cache_readme,
};

/// Re-export facade types commonly used by host binaries.
pub use sightloom::{
    DetectorAdapter, EmbeddingTask, FrameView, IndexSession, PhotoEmbeddingAdapter, PhotoView,
    PixelFormat, TrackEmbeddingAdapter,
};
