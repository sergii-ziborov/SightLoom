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
    /// Keep aspect ratio and pad (YOLO letterbox) instead of stretching.
    #[serde(default)]
    pub letterbox: bool,
    /// Letterbox pad value in `0..=255` (Ultralytics uses 114).
    #[serde(default = "default_letterbox_pad")]
    pub letterbox_pad: u8,
}

fn default_true() -> bool {
    true
}

fn default_letterbox_pad() -> u8 {
    114
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self::imagenet_like(640, 640)
    }
}

impl PreprocessConfig {
    /// Common re-id / classifier input size with `ImageNet` mean/std.
    ///
    /// Stretches to `width × height` (no letterbox).
    #[must_use]
    pub fn imagenet_like(width: u32, height: u32) -> Self {
        Self {
            width: width.max(1),
            height: height.max(1),
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            scale_1_255: true,
            letterbox: false,
            letterbox_pad: 114,
        }
    }

    /// YOLO-style detector input: `/255`, zero mean, unit std, letterbox pad.
    #[must_use]
    pub fn yolo_detect(width: u32, height: u32) -> Self {
        Self {
            width: width.max(1),
            height: height.max(1),
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
            scale_1_255: true,
            letterbox: true,
            letterbox_pad: 114,
        }
    }
}

/// Maps network-space boxes back to the source frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Letterbox {
    /// `net_x = src_x * scale_x + pad_x`.
    pub scale_x: f32,
    /// `net_y = src_y * scale_y + pad_y`.
    pub scale_y: f32,
    /// Left pad in network pixels.
    pub pad_x: f32,
    /// Top pad in network pixels.
    pub pad_y: f32,
    /// Network width.
    pub net_w: u32,
    /// Network height.
    pub net_h: u32,
    /// Source width.
    pub src_w: u32,
    /// Source height.
    pub src_h: u32,
}

impl Letterbox {
    /// Stretch (no pad) from source to network size.
    #[must_use]
    pub fn stretch(src_w: u32, src_h: u32, net_w: u32, net_h: u32) -> Self {
        let src_w = src_w.max(1);
        let src_h = src_h.max(1);
        let net_w = net_w.max(1);
        let net_h = net_h.max(1);
        Self {
            scale_x: net_w as f32 / src_w as f32,
            scale_y: net_h as f32 / src_h as f32,
            pad_x: 0.0,
            pad_y: 0.0,
            net_w,
            net_h,
            src_w,
            src_h,
        }
    }

    /// Maps a network-space point onto the source frame (clamped).
    #[must_use]
    pub fn to_source(self, x: f32, y: f32) -> (f32, f32) {
        let sx = ((x - self.pad_x) / self.scale_x.max(1e-6)).clamp(0.0, self.src_w as f32);
        let sy = ((y - self.pad_y) / self.scale_y.max(1e-6)).clamp(0.0, self.src_h as f32);
        (sx, sy)
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

/// Letterbox packed RGB8 into `net_w × net_h` with a constant pad.
///
/// # Errors
///
/// Buffer size mismatches.
pub fn letterbox_rgb8(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    net_w: u32,
    net_h: u32,
    pad: u8,
) -> Result<(Vec<u8>, Letterbox), HostError> {
    let src_w = src_w.max(1);
    let src_h = src_h.max(1);
    let net_w = net_w.max(1);
    let net_h = net_h.max(1);
    let scale = (net_w as f32 / src_w as f32).min(net_h as f32 / src_h as f32);
    let inner_w = ((src_w as f32 * scale).round() as u32).clamp(1, net_w);
    let inner_h = ((src_h as f32 * scale).round() as u32).clamp(1, net_h);
    let pad_x = (net_w - inner_w) / 2;
    let pad_y = (net_h - inner_h) / 2;
    let resized = resize_rgb8_nearest(src, src_w, src_h, inner_w, inner_h)?;
    let mut out = vec![pad; (net_w as usize) * (net_h as usize) * 3];
    let dest_w = net_w as usize;
    let copy_w = inner_w as usize;
    let copy_h = inner_h as usize;
    let ox = pad_x as usize;
    let oy = pad_y as usize;
    for y in 0..copy_h {
        let src_row = y * copy_w * 3;
        let dst_row = ((oy + y) * dest_w + ox) * 3;
        out[dst_row..dst_row + copy_w * 3].copy_from_slice(&resized[src_row..src_row + copy_w * 3]);
    }
    Ok((
        out,
        Letterbox {
            scale_x: inner_w as f32 / src_w as f32,
            scale_y: inner_h as f32 / src_h as f32,
            pad_x: pad_x as f32,
            pad_y: pad_y as f32,
            net_w,
            net_h,
            src_w,
            src_h,
        },
    ))
}

/// Resize (or letterbox) then normalize. Returns the mapping used for boxes.
///
/// # Errors
///
/// Preprocess failures.
pub fn prepare_rgb8_nchw_with_meta(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    cfg: &PreprocessConfig,
) -> Result<(Vec<f32>, Letterbox), HostError> {
    let (plane, meta) = if cfg.letterbox {
        letterbox_rgb8(src, src_w, src_h, cfg.width, cfg.height, cfg.letterbox_pad)?
    } else {
        let resized = resize_rgb8_nearest(src, src_w, src_h, cfg.width, cfg.height)?;
        (
            resized,
            Letterbox::stretch(src_w, src_h, cfg.width, cfg.height),
        )
    };
    let tensor = rgb8_to_chw_f32(&plane, cfg.width, cfg.height, cfg)?;
    Ok((tensor, meta))
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
    Ok(prepare_rgb8_nchw_with_meta(src, src_w, src_h, cfg)?.0)
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

/// Canonical upright crop for re-id / face embed (no keypoints required).
///
/// Expands the box to `target_w:target_h` around its centre, clamps to the
/// frame, crops, then nearest-resizes. This is the host **pose-lite** path
/// (Fluendo-style alignment without a 5-point mesh).
///
/// # Errors
///
/// Empty box / buffer issues.
pub fn align_crop_rgb8(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
    target_w: u32,
    target_h: u32,
) -> Result<(Vec<u8>, u32, u32), HostError> {
    let tw = target_w.max(1);
    let th = target_h.max(1);
    let bw = (right - left).max(1.0);
    let bh = (bottom - top).max(1.0);
    let cx = (left + right) * 0.5;
    let cy = (top + bottom) * 0.5;
    let want = tw as f32 / th as f32;
    let have = bw / bh;
    let (aw, ah) = if have > want {
        (bw, bw / want)
    } else {
        (bh * want, bh)
    };
    let l = (cx - aw * 0.5).floor().max(0.0);
    let t = (cy - ah * 0.5).floor().max(0.0);
    let r = (cx + aw * 0.5).ceil();
    let b = (cy + ah * 0.5).ceil();
    let (crop, cw, ch) = crop_rgb8(
        src,
        src_w,
        src_h,
        l as u32,
        t as u32,
        r.max(l + 1.0) as u32,
        b.max(t + 1.0) as u32,
    )?;
    let resized = resize_rgb8_nearest(&crop, cw, ch, tw, th)?;
    Ok((resized, tw, th))
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
    fn align_crop_is_canonical_size() {
        let src = vec![80_u8; 40 * 80 * 3];
        let (crop, w, h) = align_crop_rgb8(&src, 40, 80, 5.0, 10.0, 20.0, 50.0, 128, 256).unwrap();
        assert_eq!((w, h), (128, 256));
        assert_eq!(crop.len(), 128 * 256 * 3);
    }

    #[test]
    fn crop_center() {
        // 3x1 RGB row
        let src = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let (crop, w, h) = crop_rgb8(&src, 3, 1, 1, 0, 2, 1).unwrap();
        assert_eq!((w, h), (1, 1));
        assert_eq!(crop, vec![4, 5, 6]);
    }

    #[test]
    fn letterbox_keeps_aspect_and_inverts() {
        let src = vec![200_u8; 40 * 20 * 3];
        let (out, meta) = letterbox_rgb8(&src, 40, 20, 64, 64, 114).unwrap();
        assert_eq!(out.len(), 64 * 64 * 3);
        assert!((meta.scale_x - meta.scale_y).abs() < 1e-5);
        assert!(meta.pad_y > meta.pad_x);
        let (x, y) = meta.to_source(meta.pad_x, meta.pad_y);
        assert!(x < 1.0 && y < 1.0);
        let (x2, y2) = meta.to_source(
            meta.pad_x + 40.0 * meta.scale_x,
            meta.pad_y + 20.0 * meta.scale_y,
        );
        assert!((x2 - 40.0).abs() < 1.5);
        assert!((y2 - 20.0).abs() < 1.5);
    }

    #[test]
    fn yolo_detect_preset_letterboxes() {
        let src = vec![10_u8; 8 * 4 * 3];
        let cfg = PreprocessConfig::yolo_detect(16, 16);
        let (t, meta) = prepare_rgb8_nchw_with_meta(&src, 8, 4, &cfg).unwrap();
        assert_eq!(t.len(), 16 * 16 * 3);
        assert!(cfg.letterbox);
        assert!(meta.pad_y >= meta.pad_x);
    }
}
