//! Host detector and embedding adapter contracts.
//!
//! `SightLoom` does not ship model runtimes or turn photos into vectors by
//! itself. Hosts implement:
//! - [`DetectorAdapter`] — frames → detections
//! - [`PhotoEmbeddingAdapter`] — photo/crop bytes → embedding vector
//!
//! so the product path can be `photo → host model → SightLoom ranking`.

use sightloom_core::{Detection, FrameStamp};
use sightloom_reid::SubjectModality;

/// Pixel layout of a host frame buffer (no decode ownership).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelFormat {
    /// 8-bit grayscale.
    Gray8,
    /// Packed RGB.
    Rgb8,
    /// Packed BGR.
    Bgr8,
    /// Packed RGBA.
    Rgba8,
    /// Host-defined layout (width/height/stride still apply).
    Custom(u32),
}

/// Borrowed view of one host frame (pixels stay with the host).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameView<'a> {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row stride in bytes.
    pub stride: usize,
    /// Pixel format.
    pub format: PixelFormat,
    /// Contiguous or strided pixel bytes.
    pub data: &'a [u8],
}

impl<'a> FrameView<'a> {
    /// Creates a frame view.
    #[must_use]
    pub const fn new(
        width: u32,
        height: u32,
        stride: usize,
        format: PixelFormat,
        data: &'a [u8],
    ) -> Self {
        Self {
            width,
            height,
            stride,
            format,
            data,
        }
    }
}

/// Host-implemented detector: one frame stamp + pixel view → detections.
///
/// Implementations must not require `SightLoom` to own model weights or a runtime.
pub trait DetectorAdapter {
    /// Detector-specific error type.
    type Error: core::fmt::Debug;

    /// Runs detection on one host frame.
    ///
    /// # Errors
    ///
    /// Returns adapter-defined errors (model load, OOM, invalid buffer, …).
    fn detect(
        &mut self,
        stamp: FrameStamp,
        frame: &FrameView<'_>,
    ) -> Result<Vec<Detection>, Self::Error>;
}

/// Host track re-id embedder for continuous per-frame track vectors.
///
/// After association, the host embeds each track crop; `SightLoom` stores handles
/// via [`crate::IndexSession::note_track_embedding`].
pub trait TrackEmbeddingAdapter {
    /// Adapter error.
    type Error: core::fmt::Debug;

    /// Embeds one tracked box in the current host frame.
    ///
    /// # Errors
    ///
    /// Host model failures.
    fn embed_track(
        &mut self,
        stamp: FrameStamp,
        frame: &FrameView<'_>,
        track_key: sightloom_core::TrackKey,
        bbox: sightloom_core::Rect,
    ) -> Result<Vec<f32>, Self::Error>;
}

/// Kind of embedding a host model produces.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddingTask {
    /// Face recognition embedding.
    Face,
    /// Full-body / person re-ID embedding.
    PersonReId,
    /// Generic appearance embedding.
    GenericAppearance,
}

impl EmbeddingTask {
    /// Maps to a gallery modality.
    #[must_use]
    pub const fn to_modality(self) -> SubjectModality {
        match self {
            Self::Face => SubjectModality::Face,
            Self::PersonReId => SubjectModality::PersonAppearance,
            Self::GenericAppearance => SubjectModality::GenericObject,
        }
    }
}

/// Borrowed photo / crop buffer for embedding (host owns decode).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhotoView<'a> {
    /// Optional pixel layout when raw raster is provided.
    pub frame: Option<FrameView<'a>>,
    /// Encoded bytes (JPEG/PNG/…) when the host model wants compressed input.
    pub encoded: Option<&'a [u8]>,
}

impl<'a> PhotoView<'a> {
    /// Raw frame only.
    #[must_use]
    pub const fn from_frame(frame: FrameView<'a>) -> Self {
        Self {
            frame: Some(frame),
            encoded: None,
        }
    }

    /// Encoded image bytes only.
    #[must_use]
    pub const fn from_encoded(encoded: &'a [u8]) -> Self {
        Self {
            frame: None,
            encoded: Some(encoded),
        }
    }
}

/// Host-implemented photo → embedding vector adapter.
///
/// Completes the killer path **without** shipping weights in `SightLoom`:
/// `photo bytes → this trait → vector → gallery search`.
pub trait PhotoEmbeddingAdapter {
    /// Adapter error type.
    type Error: core::fmt::Debug;

    /// Preferred embedding task (face vs person re-id).
    fn task(&self) -> EmbeddingTask;

    /// Produces a dense embedding for one photo/crop.
    ///
    /// # Errors
    ///
    /// Model / preprocess failures.
    fn embed_photo(&mut self, photo: &PhotoView<'_>) -> Result<Vec<f32>, Self::Error>;
}
