//! Host model package for [`sightloom`](https://docs.rs/sightloom).
//!
//! # Product boundary
//!
//! `SightLoom` **does not** turn photos into embeddings. This crate is the
//! step-1 host side of the killer path:
//!
//! ```text
//! photo / frame
//!   → host detect / embed (reference or future ONNX)
//!   → SightLoom IndexSession (rank / memory / tracks)
//! ```
//!
//! Weights, GPU runtimes, and model download policies live **here** (or in a
//! private host binary), never inside `sightloom-core`.
//!
//! # Step 1 (this release)
//!
//! - [`HostBundleConfig`] / [`ModelSpec`] / [`DevicePreference`]
//! - pure-Rust [`preprocess`]
//! - filesystem [`ModelFetcher`] (no network download yet)
//! - deterministic [`ReferenceHostModels`] implementing `SightLoom` adapters
//! - [`HostPipeline`]: enroll photo, search photo, ingest frame
//!
//! # Later steps
//!
//! - real ONNX Runtime backends (`onnx` feature reserved)
//! - HTTP/S3 weight fetchers
//! - evidence packs (MOT / ROC / redaction) using exports from `SightLoom`

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

mod config;
mod device;
mod error;
mod pipeline;
mod preprocess;
mod reference;
mod registry;

pub use config::{HostBundleConfig, ModelSpec, ModelTask};
pub use device::DevicePreference;
pub use error::HostError;
pub use pipeline::HostPipeline;
pub use preprocess::{
    PreprocessConfig, crop_rgb8, prepare_rgb8_nchw, resize_rgb8_nearest, rgb8_to_chw_f32,
};
pub use reference::{
    ReferenceEmbedder, ReferenceFaceDetector, ReferenceHostModels, ReferencePersonDetector,
};
pub use registry::{
    DeferredDownloadFetcher, FilesystemFetcher, ModelFetcher, ensure_cache_dir, write_cache_readme,
};

/// Re-export facade types commonly used by host binaries.
pub use sightloom::{
    DetectorAdapter, EmbeddingTask, FrameView, IndexSession, PhotoEmbeddingAdapter, PhotoView,
    PixelFormat, TrackEmbeddingAdapter,
};
