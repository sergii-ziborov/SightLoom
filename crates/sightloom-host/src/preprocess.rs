//! Pure-Rust preprocess helpers (no image codec dependency).
//!
//! Hosts still own decode (JPEG/PNG → RGB). This module resizes / normalizes
//! dense RGB8 planes into CHW `f32` tensors for future ONNX adapters.

use crate::error::HostError;
use serde::{Deserialize, Serialize};

/// ImageNet-style or custom normalize + resize plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreprocessConfig {
    /// Output width.
    pub width: u32,
    /// Output height.
    pub height: u32,
    /// Per-channel mean (RGB).
    pub mean: [f32; 3],
    /// Per-channel std (RGB).
    pub std: [f32; 3],
    /// Scale pixel values by `1/255` before mean/std.
    #[serde(default = "default_true")]
    pub scale_1_255: bool,
}

fn default_true() -> bool {
    true
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self::imagenet_like(640, 640)
    }
}

impl PreprocessConfig {
    /// Common re-id / detector input size with `ImageNet` mean/std.
    #[must_use]
    pub fn imagenet_like(width: u32, height: u32) -> Self {
        Self {
            width: width.max(1),
            height: height.max(1),
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            scale_1_255: true,
        }
    }
}

/// Nearest-neighbor resize of packed RGB8 → packed RGB8 at `out_w × out_h`.
///
/// # Errors
///
/// Buffer size mismatches.
pub fn resize_rgb8_nearest(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    out_w: u32,
    out_h: u32,
) -> Result<Vec<u8>, HostError> {
    let src_w = src_w.max(1) as usize;
    let src_h = src_h.max(1) as usize;
    let out_w = out_w.max(1) as usize;
    let out_h = out_h.max(1) as usize;
    let need = src_w
        .checked_mul(src_h)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| HostError::Preprocess("src size overflow".into()))?;
    if src.len() < need {
        return Err(HostError::Preprocess(format!(
            "rgb buffer too small: have {} need {need}",
            src.len()
        )));
    }
    let mut out = vec![0_u8; out_w * out_h * 3];
    for y in 0..out_h {
        let sy = y * src_h / out_h;
        for x in 0..out_w {
            let sx = x * src_w / out_w;
            let si = (sy * src_w + sx) * 3;
            let di = (y * out_w + x) * 3;
            out[di] = src[si];
            out[di + 1] = src[si + 1];
            out[di + 2] = src[si + 2];
        }
    }
    Ok(out)
}

/// RGB8 packed → CHW `f32` with mean/std (and optional /255).
///
/// Layout: `[R plane | G plane | B plane]`, length `3 * W * H`.
///
/// # Errors
///
/// Size mismatch.
pub fn rgb8_to_chw_f32(
    rgb: &[u8],
    width: u32,
    height: u32,
    cfg: &PreprocessConfig,
) -> Result<Vec<f32>, HostError> {
    let w = width as usize;
    let h = height as usize;
    let n = w
        .checked_mul(h)
        .ok_or_else(|| HostError::Preprocess("plane overflow".into()))?;
    let need = n
        .checked_mul(3)
        .ok_or_else(|| HostError::Preprocess("rgb overflow".into()))?;
    if rgb.len() < need {
        return Err(HostError::Preprocess("rgb buffer too small".into()));
    }
    let mut out = vec![0.0_f32; need];
    for i in 0..n {
        let base = i * 3;
        for c in 0..3 {
            let mut v = f32::from(rgb[base + c]);
            if cfg.scale_1_255 {
                v /= 255.0;
            }
            v = (v - cfg.mean[c]) / cfg.std[c].max(1e-6);
            out[c * n + i] = v;
        }
    }
    Ok(out)
}

/// Resize then normalize in one shot.
///
/// # Errors
///
/// Preprocess failures.
pub fn prepare_rgb8_nchw(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    cfg: &PreprocessConfig,
) -> Result<Vec<f32>, HostError> {
    let resized = resize_rgb8_nearest(src, src_w, src_h, cfg.width, cfg.height)?;
    rgb8_to_chw_f32(&resized, cfg.width, cfg.height, cfg)
}

/// Crops an axis-aligned box from packed RGB8 (clamped to image).
///
/// # Errors
///
/// Empty crop / buffer issues.
pub fn crop_rgb8(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
) -> Result<(Vec<u8>, u32, u32), HostError> {
    let width = src_w as usize;
    let height = src_h as usize;
    let need = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| HostError::Preprocess("crop size overflow".into()))?;
    if src.len() < need {
        return Err(HostError::Preprocess(
            "rgb buffer too small for crop".into(),
        ));
    }
    let left_i = (left as usize).min(width.saturating_sub(1));
    let top_i = (top as usize).min(height.saturating_sub(1));
    let right_i = (right as usize).clamp(left_i + 1, width);
    let bottom_i = (bottom as usize).clamp(top_i + 1, height);
    let crop_w = right_i - left_i;
    let crop_h = bottom_i - top_i;
    let mut out = vec![0_u8; crop_w * crop_h * 3];
    for y in 0..crop_h {
        for x in 0..crop_w {
            let si = ((top_i + y) * width + (left_i + x)) * 3;
            let di = (y * crop_w + x) * 3;
            out[di..di + 3].copy_from_slice(&src[si..si + 3]);
        }
    }
    Ok((out, crop_w as u32, crop_h as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_and_chw_shapes() {
        let src = vec![10_u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]; // 2x2 RGB
        let cfg = PreprocessConfig::imagenet_like(4, 4);
        let t = prepare_rgb8_nchw(&src, 2, 2, &cfg).unwrap();
        assert_eq!(t.len(), 4 * 4 * 3);
    }

    #[test]
    fn crop_center() {
        // 3x1 RGB row
        let src = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let (crop, w, h) = crop_rgb8(&src, 3, 1, 1, 0, 2, 1).unwrap();
        assert_eq!((w, h), (1, 1));
        assert_eq!(crop, vec![4, 5, 6]);
    }
}
