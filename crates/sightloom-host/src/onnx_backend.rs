//! ONNX backends via pure-Rust [`tract_onnx`] (feature `onnx`).
//!
//! Loads weights from [`crate::ModelSpec`] via [`crate::FilesystemFetcher`].
//! Does **not** download models; place `.onnx` under the cache dir or set
//! `local_path`.
//!
//! Uses **tract** instead of Microsoft ONNX Runtime so hosts work on
//! `windows-gnu` and other targets without ORT prebuilts. Hosts that prefer
//! ORT can still implement adapters themselves.

use crate::config::ModelSpec;
use crate::error::HostError;
use crate::preprocess::{PreprocessConfig, crop_rgb8, prepare_rgb8_nchw};
use crate::reference::frame_to_rgb8;
use crate::registry::{FilesystemFetcher, ModelFetcher, ensure_cache_dir};
use sightloom::core::{ClassId, Detection, FrameStamp, Rect, TrackKey};
use sightloom::{
    DetectorAdapter, EmbeddingTask, FrameView, PhotoEmbeddingAdapter, PhotoView,
    TrackEmbeddingAdapter,
};
use std::path::{Path, PathBuf};
use tract_onnx::prelude::*;

type TractModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// Shared loaded ONNX model + path metadata.
pub struct OnnxModel {
    /// Spec used to load.
    pub spec: ModelSpec,
    /// Resolved weight path.
    pub path: PathBuf,
    plan: TractModel,
}

impl OnnxModel {
    /// Loads and optimizes an ONNX model from disk.
    ///
    /// # Errors
    ///
    /// Missing file / unsupported ops / I/O.
    pub fn load(spec: ModelSpec, cache_dir: &Path) -> Result<Self, HostError> {
        ensure_cache_dir(cache_dir)?;
        let mut fetcher = FilesystemFetcher;
        let path = fetcher.ensure_local(&spec, cache_dir)?;
        let plan = load_plan(&path)?;
        Ok(Self { spec, path, plan })
    }

    /// Runs NCHW `f32` batch as the first model input; returns first output flat.
    ///
    /// # Errors
    ///
    /// Shape / runtime failures.
    pub fn run_nchw_f32(
        &self,
        nchw: &[f32],
        n: usize,
        c: usize,
        h: usize,
        w: usize,
    ) -> Result<Vec<f32>, HostError> {
        let expected = n.saturating_mul(c).saturating_mul(h).saturating_mul(w);
        if nchw.len() < expected {
            return Err(HostError::Runtime(format!(
                "nchw too short: have {} need {expected}",
                nchw.len()
            )));
        }
        let tensor = Tensor::from_shape(&[n, c, h, w], &nchw[..expected])
            .map_err(|e| HostError::Runtime(format!("tensor: {e}")))?;
        let result = self
            .plan
            .run(tvec!(tensor.into()))
            .map_err(|e| HostError::Runtime(format!("run: {e}")))?;
        let first = result
            .into_iter()
            .next()
            .ok_or_else(|| HostError::Runtime("onnx model produced no outputs".into()))?;
        let view = first
            .to_array_view::<f32>()
            .map_err(|e| HostError::Runtime(format!("output f32: {e}")))?;
        Ok(view.iter().copied().collect())
    }
}

fn load_plan(path: &Path) -> Result<TractModel, HostError> {
    tract_onnx::onnx()
        .model_for_path(path)
        .map_err(|e| HostError::Runtime(format!("parse {}: {e}", path.display())))?
        .into_optimized()
        .map_err(|e| HostError::Runtime(format!("optimize {}: {e}", path.display())))?
        .into_runnable()
        .map_err(|e| HostError::Runtime(format!("runnable {}: {e}", path.display())))
}

/// ONNX embedding model (face / person re-id).
///
/// Expects NCHW RGB float input and a 1-D (or `[1, D]`) embedding output.
pub struct OnnxEmbedder {
    model: OnnxModel,
    task: EmbeddingTask,
    dim: usize,
    preprocess: PreprocessConfig,
}

impl OnnxEmbedder {
    /// Loads an embedding model.
    ///
    /// # Errors
    ///
    /// Path / parse / optimize errors.
    pub fn load(spec: ModelSpec, cache_dir: &Path, task: EmbeddingTask) -> Result<Self, HostError> {
        let preprocess = spec.preprocess.clone();
        let dim = spec.embedding_dim.max(1);
        let model = OnnxModel::load(spec, cache_dir)?;
        Ok(Self {
            model,
            task,
            dim,
            preprocess,
        })
    }

    /// Resolved weights path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.model.path
    }

    fn embed_rgb8(&self, rgb: &[u8], width: u32, height: u32) -> Result<Vec<f32>, HostError> {
        let nchw = prepare_rgb8_nchw(rgb, width, height, &self.preprocess)?;
        let h = self.preprocess.height as usize;
        let w = self.preprocess.width as usize;
        let mut out = self.model.run_nchw_f32(&nchw, 1, 3, h, w)?;
        if out.is_empty() {
            return Err(HostError::Runtime("empty embedding output".into()));
        }
        let norm = out.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for x in &mut out {
            *x /= norm;
        }
        if out.len() > self.dim {
            out.truncate(self.dim);
        } else if out.len() < self.dim {
            out.resize(self.dim, 0.0);
        }
        Ok(out)
    }
}

impl PhotoEmbeddingAdapter for OnnxEmbedder {
    type Error = HostError;

    fn task(&self) -> EmbeddingTask {
        self.task
    }

    fn embed_photo(&mut self, photo: &PhotoView<'_>) -> Result<Vec<f32>, Self::Error> {
        if let Some(frame) = photo.frame {
            let rgb = frame_to_rgb8(&frame)?;
            return self.embed_rgb8(&rgb, frame.width, frame.height);
        }
        Err(HostError::Runtime(
            "OnnxEmbedder requires PhotoView::frame (decoded RGB); decode JPEG/PNG in the host first"
                .into(),
        ))
    }
}

impl TrackEmbeddingAdapter for OnnxEmbedder {
    type Error = HostError;

    fn embed_track(
        &mut self,
        _stamp: FrameStamp,
        frame: &FrameView<'_>,
        _track_key: TrackKey,
        bbox: Rect,
    ) -> Result<Vec<f32>, Self::Error> {
        let rgb = frame_to_rgb8(frame)?;
        let left = bbox.left().max(0.0) as u32;
        let top = bbox.top().max(0.0) as u32;
        let right = bbox.right().max(0.0).ceil() as u32;
        let bottom = bbox.bottom().max(0.0).ceil() as u32;
        let (crop, w, h) = crop_rgb8(&rgb, frame.width, frame.height, left, top, right, bottom)?;
        self.embed_rgb8(&crop, w, h)
    }
}

/// ONNX detector with YOLO-like / flat-box postprocess.
pub struct OnnxDetector {
    model: OnnxModel,
    preprocess: PreprocessConfig,
    /// Min confidence.
    pub conf_thresh: f32,
    /// Class id when model has no class head.
    pub default_class: u16,
}

impl OnnxDetector {
    /// Loads a detector model.
    ///
    /// # Errors
    ///
    /// Path / parse errors.
    pub fn load(spec: ModelSpec, cache_dir: &Path) -> Result<Self, HostError> {
        let preprocess = spec.preprocess.clone();
        let model = OnnxModel::load(spec, cache_dir)?;
        Ok(Self {
            model,
            preprocess,
            conf_thresh: 0.25,
            default_class: 0,
        })
    }

    /// Resolved weights path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.model.path
    }

    fn detect_rgb(&self, rgb: &[u8], src_w: u32, src_h: u32) -> Result<Vec<Detection>, HostError> {
        let nchw = prepare_rgb8_nchw(rgb, src_w, src_h, &self.preprocess)?;
        let net_h = self.preprocess.height as usize;
        let net_w = self.preprocess.width as usize;
        let raw = self.model.run_nchw_f32(&nchw, 1, 3, net_h, net_w)?;
        let boxes = parse_detector_output(&raw, net_w as f32, net_h as f32, self.conf_thresh)?;
        let sx = src_w as f32 / net_w as f32;
        let sy = src_h as f32 / net_h as f32;
        let mut out = Vec::new();
        for b in boxes {
            let left = (b.x1 * sx).clamp(0.0, src_w as f32);
            let top = (b.y1 * sy).clamp(0.0, src_h as f32);
            let right = (b.x2 * sx).clamp(0.0, src_w as f32).max(left + 1.0);
            let bottom = (b.y2 * sy).clamp(0.0, src_h as f32).max(top + 1.0);
            let Ok(rect) = Rect::new(left, top, right, bottom) else {
                continue;
            };
            let class = ClassId(b.class.unwrap_or(self.default_class));
            if let Ok(det) = Detection::new(rect, b.score, Some(class), None) {
                out.push(det);
            }
        }
        out.sort_by(|a, b| {
            b.score()
                .partial_cmp(&a.score())
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        Ok(out)
    }
}

impl DetectorAdapter for OnnxDetector {
    type Error = HostError;

    fn detect(
        &mut self,
        _stamp: FrameStamp,
        frame: &FrameView<'_>,
    ) -> Result<Vec<Detection>, Self::Error> {
        let rgb = frame_to_rgb8(frame)?;
        self.detect_rgb(&rgb, frame.width, frame.height)
    }
}

#[derive(Clone, Copy, Debug)]
struct RawBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    score: f32,
    class: Option<u16>,
}

fn parse_detector_output(
    raw: &[f32],
    net_w: f32,
    net_h: f32,
    conf_thresh: f32,
) -> Result<Vec<RawBox>, HostError> {
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    if raw.len().is_multiple_of(6) && raw.len() >= 6 {
        let mut out = Vec::new();
        for chunk in raw.chunks_exact(6) {
            let score = chunk[4];
            if score < conf_thresh {
                continue;
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
        if !out.is_empty() {
            return Ok(out);
        }
    }
    for stride in [6_usize, 7, 84, 85] {
        if !raw.len().is_multiple_of(stride) {
            continue;
        }
        let n = raw.len() / stride;
        if n == 0 || n > 50_000 {
            continue;
        }
        let mut out = Vec::new();
        for i in 0..n {
            let row = &raw[i * stride..(i + 1) * stride];
            let (cx, cy, w, h, obj) = (row[0], row[1], row[2], row[3], row[4]);
            let (cls, cls_score) = if stride > 5 {
                let mut best_i = 0_usize;
                let mut best_s = row[5];
                for (j, &s) in row.iter().enumerate().skip(5) {
                    if s > best_s {
                        best_s = s;
                        best_i = j - 5;
                    }
                }
                (best_i as u16, best_s)
            } else {
                (0_u16, 1.0)
            };
            let score = obj * cls_score;
            if score < conf_thresh {
                continue;
            }
            let (cx, cy, w, h) = if cx <= 1.5 && cy <= 1.5 && w <= 1.5 && h <= 1.5 {
                (cx * net_w, cy * net_h, w * net_w, h * net_h)
            } else {
                (cx, cy, w, h)
            };
            out.push(RawBox {
                x1: cx - w * 0.5,
                y1: cy - h * 0.5,
                x2: cx + w * 0.5,
                y2: cy + h * 0.5,
                score,
                class: Some(cls),
            });
        }
        if !out.is_empty() {
            return Ok(out);
        }
    }
    Err(HostError::Runtime(format!(
        "unrecognized detector output length {} (expected Nx6 or YOLO-like strides)",
        raw.len()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_flat_boxes() {
        let raw = [
            10.0_f32, 10.0, 50.0, 80.0, 0.9, 0.0, 1.0, 1.0, 2.0, 2.0, 0.05, 1.0,
        ];
        let boxes = parse_detector_output(&raw, 640.0, 640.0, 0.25).unwrap();
        assert_eq!(boxes.len(), 1);
        assert!((boxes[0].score - 0.9).abs() < 1e-5);
    }
}
