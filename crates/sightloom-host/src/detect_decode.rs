//! YOLO-like detector-head decode + NMS (no model runtime).
//!
//! Accepts the common export layouts:
//! - `N×6` already-decoded `x1,y1,x2,y2,score,class`
//! - YOLOv8/v11 `[1, 4+C, N]` (transposed, no objectness)
//! - `YOLOv5` `[1, N, 5+C]` (objectness × class)
//! - the same shapes without a leading batch dim of 1

use crate::error::HostError;
use crate::preprocess::Letterbox;
use sightloom::core::{ClassId, Detection, NmsConfig, NmsMode, OverlapMetric, Rect, nms_in_place};

/// One decoded box in **network** pixel space (`x1,y1,x2,y2`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RawBox {
    /// Left.
    pub x1: f32,
    /// Top.
    pub y1: f32,
    /// Right.
    pub x2: f32,
    /// Bottom.
    pub y2: f32,
    /// Confidence in `0..=1` (or model units, already thresholded).
    pub score: f32,
    /// Class index when the head provides one.
    pub class: Option<u16>,
}

/// Thresholds for decode + NMS.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectorDecodeConfig {
    /// Minimum confidence to keep a raw box.
    pub conf_thresh: f32,
    /// `IoU` threshold for class-aware hard NMS.
    pub nms_thresh: f32,
    /// Maximum detections after NMS.
    pub max_det: usize,
    /// Class used when the head has no class channel.
    pub default_class: u16,
}

impl Default for DetectorDecodeConfig {
    fn default() -> Self {
        Self {
            conf_thresh: 0.25,
            nms_thresh: 0.45,
            max_det: 300,
            default_class: 0,
        }
    }
}

/// Decodes a flattened detector tensor into network-space boxes.
///
/// `shape` is the ONNX output shape (may be empty — then layout is inferred
/// from `raw.len()` the same way as the original Nx6 / stride probe).
///
/// # Errors
///
/// Unrecognized layout, or a tensor that is empty after a required parse.
pub fn decode_detector_output(
    raw: &[f32],
    shape: &[usize],
    net_w: f32,
    net_h: f32,
    conf_thresh: f32,
) -> Result<Vec<RawBox>, HostError> {
    if raw.is_empty() {
        return Ok(Vec::new());
    }

    let squeezed = squeeze_ones(shape);
    if squeezed.len() >= 2 {
        let rows = squeezed[squeezed.len() - 2];
        let cols = squeezed[squeezed.len() - 1];
        if rows.checked_mul(cols) == Some(raw.len())
            && let Some(out) = decode_matrix(raw, rows, cols, net_w, net_h, conf_thresh)
        {
            // Trust an explicit tensor shape even when nothing is above thresh.
            return Ok(out);
        }
    }

    if let Some(out) = decode_flat_xyxy6(raw, conf_thresh)
        && !out.is_empty()
    {
        return Ok(out);
    }

    let mut recognized_empty = false;
    let mut best: Option<Vec<RawBox>> = None;
    for stride in [6_usize, 7, 84, 85, 5] {
        if !raw.len().is_multiple_of(stride) {
            continue;
        }
        let n = raw.len() / stride;
        if n == 0 || n > 50_000 {
            continue;
        }
        for (rows, cols) in [(n, stride), (stride, n)] {
            if let Some(out) = decode_matrix(raw, rows, cols, net_w, net_h, conf_thresh) {
                if out.is_empty() {
                    recognized_empty = true;
                    continue;
                }
                if best.as_ref().is_none_or(|b| out.len() > b.len()) {
                    best = Some(out);
                }
            }
        }
    }
    if let Some(out) = best {
        return Ok(out);
    }
    if recognized_empty {
        return Ok(Vec::new());
    }

    Err(HostError::Runtime(format!(
        "unrecognized detector output length {} shape {shape:?} (expected Nx6 or YOLO-like)",
        raw.len()
    )))
}

/// Maps network-space boxes to source-frame [`Detection`]s and applies NMS.
///
/// # Errors
///
/// NMS configuration errors (should not happen with [`DetectorDecodeConfig::default`]).
pub fn detections_from_raw_boxes(
    boxes: &[RawBox],
    letterbox: Letterbox,
    cfg: DetectorDecodeConfig,
) -> Result<Vec<Detection>, HostError> {
    let src_w = letterbox.src_w as f32;
    let src_h = letterbox.src_h as f32;
    let mut dets = Vec::with_capacity(boxes.len());
    for b in boxes {
        let (left, top) = letterbox.to_source(b.x1, b.y1);
        let (right, bottom) = letterbox.to_source(b.x2, b.y2);
        let left = left.clamp(0.0, src_w);
        let top = top.clamp(0.0, src_h);
        let right = right.clamp(0.0, src_w).max(left + 1.0);
        let bottom = bottom.clamp(0.0, src_h).max(top + 1.0);
        let Ok(rect) = Rect::new(left, top, right, bottom) else {
            continue;
        };
        let class = ClassId(b.class.unwrap_or(cfg.default_class));
        if let Ok(det) = Detection::new(rect, b.score, Some(class), None) {
            dets.push(det);
        }
    }
    apply_class_aware_nms(&mut dets, cfg.nms_thresh, cfg.max_det)?;
    Ok(dets)
}

fn apply_class_aware_nms(
    dets: &mut Vec<Detection>,
    nms_thresh: f32,
    max_det: usize,
) -> Result<(), HostError> {
    if dets.len() <= 1 {
        if dets.len() > max_det {
            dets.truncate(max_det);
        }
        return Ok(());
    }
    let mut order = vec![0_usize; dets.len()];
    let mut suppressed = vec![false; dets.len()];
    let kept = nms_in_place(
        dets,
        &mut order,
        &mut suppressed,
        NmsConfig {
            threshold: nms_thresh.clamp(0.0, 1.0),
            mode: NmsMode::ClassAware,
            metric: OverlapMetric::IoU,
        },
    )
    .map_err(|e| HostError::Runtime(format!("nms: {e}")))?;
    dets.truncate(kept);
    dets.sort_by(|a, b| {
        b.score()
            .partial_cmp(&a.score())
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    if dets.len() > max_det {
        dets.truncate(max_det);
    }
    Ok(())
}

fn squeeze_ones(shape: &[usize]) -> Vec<usize> {
    let squeezed: Vec<usize> = shape.iter().copied().filter(|&d| d != 1).collect();
    if squeezed.is_empty() && !shape.is_empty() {
        vec![1]
    } else {
        squeezed
    }
}

fn channel_preference(d: usize) -> i32 {
    match d {
        84 | 85 => 100,
        6 | 7 => 80,
        5 => 70,
        80 | 81 | 90 | 91 => 60,
        4 => 40,
        8..=256 => 20,
        _ => 0,
    }
}

fn decode_matrix(
    raw: &[f32],
    rows: usize,
    cols: usize,
    net_w: f32,
    net_h: f32,
    conf_thresh: f32,
) -> Option<Vec<RawBox>> {
    if rows == 0 || cols == 0 {
        return None;
    }
    let row_pref = channel_preference(rows);
    let col_pref = channel_preference(cols);
    let (n, c, transposed) = if row_pref > col_pref {
        (cols, rows, true)
    } else if col_pref > row_pref {
        (rows, cols, false)
    } else if rows <= 256 && cols > rows {
        (cols, rows, true)
    } else {
        (rows, cols, false)
    };
    if c < 4 || n == 0 || n > 50_000 {
        return None;
    }
    let at = |channel: usize, index: usize| -> f32 {
        if transposed {
            raw[channel * n + index]
        } else {
            raw[index * c + channel]
        }
    };
    let mut out = Vec::new();
    for i in 0..n {
        if let Some(b) = decode_one(c, |ch| at(ch, i), net_w, net_h, conf_thresh) {
            out.push(b);
        }
    }
    Some(out)
}

fn decode_one(
    channels: usize,
    at: impl Fn(usize) -> f32,
    net_w: f32,
    net_h: f32,
    conf_thresh: f32,
) -> Option<RawBox> {
    if channels == 6 {
        let x1 = at(0);
        let y1 = at(1);
        let x2 = at(2);
        let y2 = at(3);
        let score = at(4);
        // Prefer already-decoded xyxy when the box is ordered. Do not reject
        // the row here: YOLOv8 2-class heads also have C=6 (cxcywh + 2 cls).
        if x2 > x1 && y2 > y1 && score.is_finite() && score >= conf_thresh {
            return Some(scale_xyxy(
                x1,
                y1,
                x2,
                y2,
                score,
                Some(at(5) as u16),
                net_w,
                net_h,
            ));
        }
    }

    let cx = at(0);
    let cy = at(1);
    let w = at(2);
    let h = at(3);
    if ![cx, cy, w, h].iter().all(|v| v.is_finite()) {
        return None;
    }

    let (class, score) = class_and_score(channels, &at);
    if !score.is_finite() || score < conf_thresh {
        return None;
    }
    Some(cxcywh_to_box(cx, cy, w, h, score, class, net_w, net_h))
}

fn class_and_score(channels: usize, at: &impl Fn(usize) -> f32) -> (Option<u16>, f32) {
    if channels == 4 {
        return (None, 1.0);
    }
    if channels == 5 {
        return (None, at(4));
    }
    // Prefer the modern no-objectness head (YOLOv8: 4 + C) when that class
    // count looks plausible. YOLOv5 COCO (`C=85`) still takes objectness.
    let yolo8_classes = channels.saturating_sub(4);
    let yolo5_classes = channels.saturating_sub(5);
    let use_objectness =
        is_typical_class_count(yolo5_classes) && !is_typical_class_count(yolo8_classes);
    if use_objectness {
        let obj = at(4);
        let (cls, cls_s) = best_class(channels, 5, at);
        (Some(cls), obj * cls_s)
    } else {
        let (cls, cls_s) = best_class(channels, 4, at);
        (Some(cls), cls_s)
    }
}

fn is_typical_class_count(n: usize) -> bool {
    (1..=10).contains(&n) || matches!(n, 20 | 80 | 90 | 91)
}

fn best_class(channels: usize, start: usize, at: &impl Fn(usize) -> f32) -> (u16, f32) {
    let mut best_i = 0_u16;
    let mut best_s = f32::NEG_INFINITY;
    for j in start..channels {
        let s = at(j);
        if s > best_s {
            best_s = s;
            best_i = (j - start) as u16;
        }
    }
    if best_s.is_finite() {
        (best_i, best_s)
    } else {
        (0, 0.0)
    }
}

#[allow(clippy::too_many_arguments)]
fn cxcywh_to_box(
    mut cx: f32,
    mut cy: f32,
    mut w: f32,
    mut h: f32,
    score: f32,
    class: Option<u16>,
    net_w: f32,
    net_h: f32,
) -> RawBox {
    if cx <= 1.5 && cy <= 1.5 && w <= 1.5 && h <= 1.5 {
        cx *= net_w;
        cy *= net_h;
        w *= net_w;
        h *= net_h;
    }
    RawBox {
        x1: cx - w * 0.5,
        y1: cy - h * 0.5,
        x2: cx + w * 0.5,
        y2: cy + h * 0.5,
        score,
        class,
    }
}

#[allow(clippy::too_many_arguments)]
fn scale_xyxy(
    mut x1: f32,
    mut y1: f32,
    mut x2: f32,
    mut y2: f32,
    score: f32,
    class: Option<u16>,
    net_w: f32,
    net_h: f32,
) -> RawBox {
    if x1 >= 0.0 && y1 >= 0.0 && x2 <= 1.5 && y2 <= 1.5 {
        x1 *= net_w;
        y1 *= net_h;
        x2 *= net_w;
        y2 *= net_h;
    }
    RawBox {
        x1,
        y1,
        x2,
        y2,
        score,
        class,
    }
}

fn decode_flat_xyxy6(raw: &[f32], conf_thresh: f32) -> Option<Vec<RawBox>> {
    if !raw.len().is_multiple_of(6) || raw.len() < 6 {
        return None;
    }
    let mut out = Vec::new();
    for chunk in raw.chunks_exact(6) {
        let score = chunk[4];
        if score < conf_thresh {
            continue;
        }
        if chunk[2] <= chunk[0] || chunk[3] <= chunk[1] {
            return None;
        }
        out.push(RawBox {
            x1: chunk[0],
            y1: chunk[1],
            x2: chunk[2],
            y2: chunk[3],
            score,
            class: Some(chunk[5] as u16),
        });
    }
    if out.is_empty() { None } else { Some(out) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_shaped_head_is_ok() {
        let raw = vec![0.0_f32; 6 * 4];
        let boxes = decode_detector_output(&raw, &[1, 6, 4], 640.0, 640.0, 0.25).unwrap();
        assert!(boxes.is_empty());
    }

    #[test]
    fn parse_flat_boxes() {
        let raw = [
            10.0_f32, 10.0, 50.0, 80.0, 0.9, 0.0, 1.0, 1.0, 2.0, 2.0, 0.05, 1.0,
        ];
        let boxes = decode_detector_output(&raw, &[], 640.0, 640.0, 0.25).unwrap();
        assert_eq!(boxes.len(), 1);
        assert!((boxes[0].score - 0.9).abs() < 1e-5);
        assert_eq!(boxes[0].class, Some(0));
    }

    #[test]
    fn parse_yolov8_transposed() {
        // shape [1, 6, 2] = 4 box + 2 classes, two predictions, channels-first
        let n = 2_usize;
        let c = 6_usize;
        let mut raw = vec![0.0_f32; c * n];
        // pred 0: cx=320, cy=320, w=40, h=80, cls0=0.1, cls1=0.9
        raw[0] = 320.0;
        raw[n] = 320.0;
        raw[2 * n] = 40.0;
        raw[3 * n] = 80.0;
        raw[4 * n] = 0.1;
        raw[5 * n] = 0.9;
        // pred 1: low score
        raw[1] = 10.0;
        raw[n + 1] = 10.0;
        raw[2 * n + 1] = 4.0;
        raw[3 * n + 1] = 4.0;
        raw[4 * n + 1] = 0.01;
        raw[5 * n + 1] = 0.02;
        let boxes = decode_detector_output(&raw, &[1, 6, 2], 640.0, 640.0, 0.25).unwrap();
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].class, Some(1));
        assert!((boxes[0].score - 0.9).abs() < 1e-5);
        assert!((boxes[0].x1 - 300.0).abs() < 1e-3);
        assert!((boxes[0].y2 - 360.0).abs() < 1e-3);
    }

    #[test]
    fn parse_yolov5_rows() {
        // [1, 2, 85] = two YOLOv5-COCO rows (obj × class).
        let c = 85_usize;
        let mut raw = vec![0.0_f32; 2 * c];
        raw[0] = 100.0;
        raw[1] = 50.0;
        raw[2] = 20.0;
        raw[3] = 10.0;
        raw[4] = 0.8;
        raw[5] = 0.5; // score 0.40
        raw[c] = 200.0;
        raw[c + 1] = 80.0;
        raw[c + 2] = 10.0;
        raw[c + 3] = 10.0;
        raw[c + 4] = 0.9;
        raw[c + 5] = 0.9; // score 0.81
        let boxes = decode_detector_output(&raw, &[1, 2, 85], 640.0, 640.0, 0.5).unwrap();
        assert_eq!(boxes.len(), 1);
        assert!((boxes[0].score - 0.81).abs() < 1e-5);
        assert_eq!(boxes[0].class, Some(0));
    }

    #[test]
    fn nms_drops_overlap_keeps_class_split() {
        let meta = Letterbox::stretch(640, 640, 640, 640);
        let boxes = [
            RawBox {
                x1: 10.0,
                y1: 10.0,
                x2: 50.0,
                y2: 50.0,
                score: 0.9,
                class: Some(0),
            },
            RawBox {
                x1: 12.0,
                y1: 12.0,
                x2: 52.0,
                y2: 52.0,
                score: 0.8,
                class: Some(0),
            },
            RawBox {
                x1: 12.0,
                y1: 12.0,
                x2: 52.0,
                y2: 52.0,
                score: 0.85,
                class: Some(1),
            },
        ];
        let dets =
            detections_from_raw_boxes(&boxes, meta, DetectorDecodeConfig::default()).unwrap();
        assert_eq!(dets.len(), 2);
        assert_eq!(dets[0].class_id(), Some(ClassId(0)));
        assert_eq!(dets[1].class_id(), Some(ClassId(1)));
    }

    #[test]
    fn letterbox_maps_boxes_back() {
        let src = vec![0_u8; 40 * 20 * 3];
        let (_, meta) = crate::preprocess::letterbox_rgb8(&src, 40, 20, 64, 64, 114).unwrap();
        let boxes = [RawBox {
            x1: meta.pad_x,
            y1: meta.pad_y,
            x2: meta.pad_x + 40.0 * meta.scale_x,
            y2: meta.pad_y + 20.0 * meta.scale_y,
            score: 0.99,
            class: Some(0),
        }];
        let dets =
            detections_from_raw_boxes(&boxes, meta, DetectorDecodeConfig::default()).unwrap();
        assert_eq!(dets.len(), 1);
        let b = dets[0].bbox();
        assert!(b.left() < 2.0);
        assert!(b.top() < 2.0);
        assert!((b.right() - 40.0).abs() < 2.0);
        assert!((b.bottom() - 20.0).abs() < 2.0);
    }
}
