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
//! | `onnx` | ONNX Runtime backends ([`OnnxEmbedder`], [`OnnxDetector`]) |
//!
//! # Steps
//!
//! 1. Config, preprocess, reference models, [`HostPipeline`]
//! 2. **ONNX** load from `.sightloom-models/` / `ModelSpec.local_path` (`--features onnx`)
//! 3. **Evidence packs** — [`evidence`] MOT / re-id ROC / redaction reports
//! 4. Network download of `ModelSpec.uri` (later)

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

mod config;
mod device;
mod error;
pub mod evidence;
#[cfg(feature = "onnx")]
mod onnx_backend;
mod pipeline;
mod preprocess;
mod reference;
mod registry;

pub use config::{HostBundleConfig, ModelSpec, ModelTask};
pub use device::DevicePreference;
pub use error::HostError;
pub use evidence::{
    EvidencePack, EvidencePackPaths, MotEvidence, RedactionEvidence, ReidEvidence,
    build_synthetic_evidence_pack, build_synthetic_mot_evidence,
    build_synthetic_redaction_evidence, build_synthetic_reid_evidence, write_evidence_pack,
};
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
pub use registry::{
    DeferredDownloadFetcher, FilesystemFetcher, ModelFetcher, ensure_cache_dir, write_cache_readme,
};

/// Re-export facade types commonly used by host binaries.
pub use sightloom::{
    DetectorAdapter, EmbeddingTask, FrameView, IndexSession, PhotoEmbeddingAdapter, PhotoView,
    PixelFormat, TrackEmbeddingAdapter,
};
