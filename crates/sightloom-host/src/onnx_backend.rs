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
use crate::detect_decode::{
    DetectorDecodeConfig, decode_detector_output, detections_from_raw_boxes,
};
use crate::error::HostError;
use crate::preprocess::{
    PreprocessConfig, crop_rgb8, prepare_rgb8_nchw, prepare_rgb8_nchw_with_meta,
};
use crate::reference::frame_to_rgb8;
use crate::registry::{FilesystemFetcher, ModelFetcher, ensure_cache_dir};
use sightloom::core::{Detection, FrameStamp, Rect, TrackKey};
use sightloom::{
    DetectorAdapter, EmbeddingTask, FrameView, PhotoEmbeddingAdapter, PhotoView,
    TrackEmbeddingAdapter,
};
use std::path::{Path, PathBuf};
use tract_onnx::prelude::*;

type TractModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;
type ShapedTensor = (Vec<usize>, Vec<f32>);

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
        Ok(self
            .run_nchw_f32_outputs(nchw, n, c, h, w)?
            .into_iter()
            .next()
            .ok_or_else(|| HostError::Runtime("onnx model produced no outputs".into()))?
            .1)
    }

    /// Runs NCHW `f32` input and returns every output as `(shape, values)`.
    ///
    /// # Errors
    ///
    /// Shape / runtime failures.
    pub fn run_nchw_f32_outputs(
        &self,
        nchw: &[f32],
        n: usize,
        c: usize,
        h: usize,
        w: usize,
    ) -> Result<Vec<ShapedTensor>, HostError> {
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
        let mut outs = Vec::with_capacity(result.len());
        for tensor in result {
            let view = tensor
                .to_array_view::<f32>()
                .map_err(|e| HostError::Runtime(format!("output f32: {e}")))?;
            outs.push((view.shape().to_vec(), view.iter().copied().collect()));
        }
        if outs.is_empty() {
            return Err(HostError::Runtime("onnx model produced no outputs".into()));
        }
        Ok(outs)
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
        let decoded = crate::decode::decode_photo_rgb(photo)?;
        self.embed_rgb8(&decoded.rgb, decoded.width, decoded.height)
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
///
/// Expects NCHW RGB `f32` input. Postprocess understands:
/// - `N×6` (`x1,y1,x2,y2,score,class`)
/// - YOLOv8/v11 `[1, 4+C, N]` (no objectness)
/// - `YOLOv5` `[1, N, 5+C]` (objectness × class)
///
/// Use [`PreprocessConfig::yolo_detect`] for Ultralytics-style `/255` + letterbox.
pub struct OnnxDetector {
    model: OnnxModel,
    preprocess: PreprocessConfig,
    /// Min confidence.
    pub conf_thresh: f32,
    /// Class id when the model has no class head.
    pub default_class: u16,
    /// `IoU` threshold for class-aware hard NMS.
    pub nms_thresh: f32,
    /// Maximum detections kept after NMS.
    pub max_det: usize,
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
        let decode = DetectorDecodeConfig::default();
        Ok(Self {
            model,
            preprocess,
            conf_thresh: decode.conf_thresh,
            default_class: decode.default_class,
            nms_thresh: decode.nms_thresh,
            max_det: decode.max_det,
        })
    }

    /// Resolved weights path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.model.path
    }

    fn decode_cfg(&self) -> DetectorDecodeConfig {
        DetectorDecodeConfig {
            conf_thresh: self.conf_thresh,
            nms_thresh: self.nms_thresh,
            max_det: self.max_det.max(1),
            default_class: self.default_class,
        }
    }

    /// Runs detection on a packed RGB8 plane.
    ///
    /// # Errors
    ///
    /// Preprocess / runtime / unrecognized head layout.
    pub fn detect_rgb8(
        &self,
        rgb: &[u8],
        src_w: u32,
        src_h: u32,
    ) -> Result<Vec<Detection>, HostError> {
        let (nchw, meta) = prepare_rgb8_nchw_with_meta(rgb, src_w, src_h, &self.preprocess)?;
        let net_h = self.preprocess.height as usize;
        let net_w = self.preprocess.width as usize;
        let outputs = self.model.run_nchw_f32_outputs(&nchw, 1, 3, net_h, net_w)?;
        let decode = self.decode_cfg();
        let mut last_err = None;
        for (shape, raw) in outputs {
            match decode_detector_output(
                &raw,
                &shape,
                net_w as f32,
                net_h as f32,
                decode.conf_thresh,
            ) {
                Ok(boxes) => {
                    return detections_from_raw_boxes(&boxes, meta, decode);
                }
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or_else(|| HostError::Runtime("no decoder matched".into())))
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
        self.detect_rgb8(&rgb, frame.width, frame.height)
    }
}
