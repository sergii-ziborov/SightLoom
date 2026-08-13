//! Host detector adapter contract (frames → detections).
//!
//! `SightLoom` does not ship model runtimes. Hosts implement [`DetectorAdapter`]
//! around their ONNX/Torch/custom stack and feed results into
//! [`crate::IndexSession::detect_and_ingest`].

use sightloom_core::{Detection, FrameStamp};

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
