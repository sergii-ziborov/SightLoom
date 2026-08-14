//! Deterministic reference models (no weights) implementing `SightLoom` adapters.
//!
//! These prove the **end-to-end wiring** (`photo → embed → rank`) without
//! claiming real re-id accuracy. Replace with ONNX backends in a later step.

use crate::config::{HostBundleConfig, ModelTask};
use crate::error::HostError;
use crate::preprocess::{PreprocessConfig, crop_rgb8, prepare_rgb8_nchw};
use sightloom::core::{ClassId, Detection, FrameStamp, Rect, TrackKey};
use sightloom::{
    DetectorAdapter, EmbeddingTask, FrameView, PhotoEmbeddingAdapter, PhotoView, PixelFormat,
    TrackEmbeddingAdapter,
};

const REF_DIM: usize = 128;

/// Deterministic person detector: one full-frame box when any non-zero pixel exists.
#[derive(Clone, Debug, Default)]
pub struct ReferencePersonDetector {
    /// Score for the synthetic box.
    pub score: f32,
}

impl DetectorAdapter for ReferencePersonDetector {
    type Error = HostError;

    fn detect(
        &mut self,
        _stamp: FrameStamp,
        frame: &FrameView<'_>,
    ) -> Result<Vec<Detection>, Self::Error> {
        if frame.width == 0 || frame.height == 0 {
            return Ok(Vec::new());
        }
        let has_signal = frame.data.iter().any(|&b| b > 0);
        if !has_signal {
            return Ok(Vec::new());
        }
        // Inset box so multi-frame motion is possible when host shifts content.
        let margin_x = (frame.width / 10).max(1);
        let margin_y = (frame.height / 10).max(1);
        let bbox = Rect::new(
            margin_x as f32,
            margin_y as f32,
            (frame.width - margin_x) as f32,
            (frame.height - margin_y) as f32,
        )
        .map_err(|_| HostError::Runtime("invalid det box".into()))?;
        let det = Detection::new(bbox, self.score.max(0.5), Some(ClassId(0)), None)
            .map_err(|_| HostError::Runtime("invalid detection".into()))?;
        Ok(vec![det])
    }
}

/// Face detector stub: smaller centered box when pixels present.
#[derive(Clone, Debug, Default)]
pub struct ReferenceFaceDetector;

impl DetectorAdapter for ReferenceFaceDetector {
    type Error = HostError;

    fn detect(
        &mut self,
        _stamp: FrameStamp,
        frame: &FrameView<'_>,
    ) -> Result<Vec<Detection>, Self::Error> {
        if frame.width < 8 || frame.height < 8 {
            return Ok(Vec::new());
        }
        if !frame.data.iter().any(|&b| b > 0) {
            return Ok(Vec::new());
        }
        let cx = frame.width as f32 * 0.5;
        let cy = frame.height as f32 * 0.35;
        let hw = frame.width as f32 * 0.15;
        let hh = frame.height as f32 * 0.18;
        let bbox = Rect::new(cx - hw, cy - hh, cx + hw, cy + hh)
            .map_err(|_| HostError::Runtime("invalid face box".into()))?;
        let det = Detection::new(bbox, 0.85, Some(ClassId(1)), None)
            .map_err(|_| HostError::Runtime("invalid face det".into()))?;
        Ok(vec![det])
    }
}

/// Deterministic embedding from RGB (or encoded bytes fingerprint).
#[derive(Clone, Debug)]
pub struct ReferenceEmbedder {
    /// Task label.
    pub task: EmbeddingTask,
    /// Output dimension.
    pub dim: usize,
    /// Preprocess for raw frames.
    pub preprocess: PreprocessConfig,
}

impl Default for ReferenceEmbedder {
    fn default() -> Self {
        Self::person_reid()
    }
}

impl ReferenceEmbedder {
    /// Person re-id style embedder.
    #[must_use]
    pub fn person_reid() -> Self {
        Self {
            task: EmbeddingTask::PersonReId,
            dim: REF_DIM,
            preprocess: PreprocessConfig::imagenet_like(128, 256),
        }
    }

    /// Face embedder.
    #[must_use]
    pub fn face() -> Self {
        Self {
            task: EmbeddingTask::Face,
            dim: REF_DIM,
            preprocess: PreprocessConfig::imagenet_like(112, 112),
        }
    }

    fn embed_bytes(&self, bytes: &[u8]) -> Vec<f32> {
        let mut v = vec![0.0_f32; self.dim.max(8)];
        let n = v.len();
        for (i, chunk) in bytes.chunks(4).enumerate() {
            let mut acc = (i as u32).wrapping_mul(0x9E37_79B9);
            for &b in chunk {
                acc = acc.wrapping_mul(16_777_619) ^ u32::from(b);
            }
            let idx = (i * 3) % n;
            let f = (acc as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            v[idx] += f;
            v[(idx + 1) % n] += f * 0.5;
        }
        // L2 normalize
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for x in &mut v {
            *x /= norm;
        }
        v.truncate(self.dim.max(1));
        if v.len() < self.dim {
            v.resize(self.dim, 0.0);
        }
        v
    }

    /// Embed packed RGB8 with known size.
    ///
    /// # Errors
    ///
    /// Preprocess failures.
    pub fn embed_rgb8(&self, rgb: &[u8], width: u32, height: u32) -> Result<Vec<f32>, HostError> {
        let tensor = prepare_rgb8_nchw(rgb, width, height, &self.preprocess)?;
        // Mix CHW stats into fingerprint so geometry matters.
        let mut bytes = Vec::with_capacity(64 + tensor.len().min(256));
        bytes.extend_from_slice(&width.to_le_bytes());
        bytes.extend_from_slice(&height.to_le_bytes());
        for (i, x) in tensor.iter().enumerate().take(256) {
            let q = (x.clamp(-2.0, 2.0) * 64.0) as i8;
            bytes.push(q as u8);
            if i % 17 == 0 {
                bytes.push((i as u8).wrapping_add(3));
            }
        }
        Ok(self.embed_bytes(&bytes))
    }
}

impl PhotoEmbeddingAdapter for ReferenceEmbedder {
    type Error = HostError;

    fn task(&self) -> EmbeddingTask {
        self.task
    }

    fn embed_photo(&mut self, photo: &PhotoView<'_>) -> Result<Vec<f32>, Self::Error> {
        if let Some(frame) = photo.frame {
            let rgb = frame_to_rgb8(&frame)?;
            return self.embed_rgb8(&rgb, frame.width, frame.height);
        }
        if let Some(enc) = photo.encoded {
            return Ok(self.embed_bytes(enc));
        }
        Err(HostError::Runtime(
            "PhotoView has neither frame nor encoded bytes".into(),
        ))
    }
}

impl TrackEmbeddingAdapter for ReferenceEmbedder {
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

/// Converts a [`FrameView`] into packed RGB8 (host preprocess helper).
///
/// # Errors
///
/// Unsupported format / short buffers.
pub fn frame_to_rgb8(frame: &FrameView<'_>) -> Result<Vec<u8>, HostError> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let n = w * h;
    match frame.format {
        PixelFormat::Rgb8 => {
            let need = n * 3;
            if frame.data.len() < need {
                return Err(HostError::Preprocess("rgb8 short".into()));
            }
            Ok(frame.data[..need].to_vec())
        }
        PixelFormat::Bgr8 => {
            let need = n * 3;
            if frame.data.len() < need {
                return Err(HostError::Preprocess("bgr8 short".into()));
            }
            let mut out = vec![0_u8; need];
            for i in 0..n {
                out[i * 3] = frame.data[i * 3 + 2];
                out[i * 3 + 1] = frame.data[i * 3 + 1];
                out[i * 3 + 2] = frame.data[i * 3];
            }
            Ok(out)
        }
        PixelFormat::Gray8 => {
            if frame.data.len() < n {
                return Err(HostError::Preprocess("gray short".into()));
            }
            let mut out = vec![0_u8; n * 3];
            for i in 0..n {
                let g = frame.data[i];
                out[i * 3] = g;
                out[i * 3 + 1] = g;
                out[i * 3 + 2] = g;
            }
            Ok(out)
        }
        PixelFormat::Rgba8 => {
            let need = n * 4;
            if frame.data.len() < need {
                return Err(HostError::Preprocess("rgba short".into()));
            }
            let mut out = vec![0_u8; n * 3];
            for i in 0..n {
                out[i * 3] = frame.data[i * 4];
                out[i * 3 + 1] = frame.data[i * 4 + 1];
                out[i * 3 + 2] = frame.data[i * 4 + 2];
            }
            Ok(out)
        }
        PixelFormat::Custom(_) => Err(HostError::Preprocess(
            "custom pixel format needs host conversion".into(),
        )),
    }
}

/// Bundled reference detectors + embedders for e2e demos.
#[derive(Clone, Debug)]
pub struct ReferenceHostModels {
    /// Bundle config (paths / tasks).
    pub config: HostBundleConfig,
    /// Person detector.
    pub person_detect: ReferencePersonDetector,
    /// Face detector.
    pub face_detect: ReferenceFaceDetector,
    /// Person re-id embedder.
    pub person_reid: ReferenceEmbedder,
    /// Face embedder.
    pub face_embed: ReferenceEmbedder,
}

impl ReferenceHostModels {
    /// Builds from config (dims taken from specs when set).
    #[must_use]
    pub fn from_config(config: HostBundleConfig) -> Self {
        let person_dim = config
            .person_reid
            .as_ref()
            .map_or(REF_DIM, |s| s.embedding_dim.max(8));
        let face_dim = config
            .face_embed
            .as_ref()
            .map_or(REF_DIM, |s| s.embedding_dim.max(8));
        let mut person_reid = ReferenceEmbedder::person_reid();
        person_reid.dim = person_dim;
        if let Some(spec) = &config.person_reid {
            person_reid.preprocess = spec.preprocess.clone();
        }
        let mut face_embed = ReferenceEmbedder::face();
        face_embed.dim = face_dim;
        if let Some(spec) = &config.face_embed {
            face_embed.preprocess = spec.preprocess.clone();
        }
        Self {
            config,
            person_detect: ReferencePersonDetector { score: 0.9 },
            face_detect: ReferenceFaceDetector,
            person_reid,
            face_embed,
        }
    }

    /// Default reference bundle.
    #[must_use]
    pub fn new() -> Self {
        Self::from_config(HostBundleConfig::default())
    }

    /// Lists model tasks declared in config (for diagnostics).
    #[must_use]
    pub fn declared_tasks(&self) -> Vec<ModelTask> {
        self.config
            .all_specs()
            .into_iter()
            .map(|s| s.task)
            .collect()
    }
}

impl Default for ReferenceHostModels {
    fn default() -> Self {
        Self::new()
    }
}
