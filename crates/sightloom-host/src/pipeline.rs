//! End-to-end host pipelines: photo→rank and detect→track→embed.

use crate::error::HostError;
use crate::reference::ReferenceHostModels;
use sightloom::core::{Detection, FrameStamp, SubjectId};
use sightloom::reid::{PhotoSearchHit, SubjectModality};
use sightloom::tracking::{ByteTrackConfig, TrackedDetection};
use sightloom::{DetectorAdapter, FrameView, IndexSession, PhotoEmbeddingAdapter, PhotoView};
use std::path::Path;
#[cfg(feature = "onnx")]
use std::path::PathBuf;

#[cfg(feature = "onnx")]
use crate::onnx_backend::{OnnxDetector, OnnxEmbedder};
#[cfg(feature = "onnx")]
use crate::{EmbeddingTask, ModelManifest, ModelSpec, ModelTask};

/// Optional ONNX backends that replace the reference detector / embedder.
#[cfg(feature = "onnx")]
struct OnnxSlot {
    person_detect: OnnxDetector,
    person_reid: OnnxEmbedder,
    face_embed: Option<OnnxEmbedder>,
}

/// Runs the killer path with **host** embedders + `SightLoom` ranking.
///
/// ```text
/// photo / frame
///   → reference fake  **or**  OnnxDetector / OnnxEmbedder
///   → IndexSession gallery rank / tracks
/// ```
///
/// Host-facing entry points:
/// - [`HostPipeline::enroll_photo`] — JPEG/PNG → [`SubjectId`]
/// - [`HostPipeline::search_photo`] — JPEG/PNG or raster → [`PhotoSearchHit`]
/// - [`HostPipeline::ingest_frame`] — RGB frame → [`TrackedDetection`]
/// - [`HostPipeline::save_package`] — persist `VisionIndex` + gallery
pub struct HostPipeline {
    /// Live session.
    pub session: IndexSession,
    /// Reference (no-weight) models. Used when no ONNX slot is loaded.
    pub models: ReferenceHostModels,
    #[cfg(feature = "onnx")]
    onnx: Option<OnnxSlot>,
}

impl HostPipeline {
    /// New pipeline with default tracker config and reference models.
    ///
    /// # Errors
    ///
    /// Tracker config errors.
    pub fn new(name: impl Into<String>) -> Result<Self, HostError> {
        Self::with_models(name, ReferenceHostModels::new())
    }

    /// New pipeline with custom models.
    ///
    /// # Errors
    ///
    /// Tracker config errors.
    pub fn with_models(
        name: impl Into<String>,
        models: ReferenceHostModels,
    ) -> Result<Self, HostError> {
        let session = IndexSession::new(name, ByteTrackConfig::default())
            .map_err(|e| HostError::Runtime(format!("session: {e}")))?;
        Ok(Self {
            session,
            models,
            #[cfg(feature = "onnx")]
            onnx: None,
        })
    }

    /// True when ONNX detector / embedder are driving this pipeline.
    #[must_use]
    pub fn uses_onnx(&self) -> bool {
        #[cfg(feature = "onnx")]
        {
            self.onnx.is_some()
        }
        #[cfg(not(feature = "onnx"))]
        {
            false
        }
    }

    /// Load `person_detect.onnx` + `person_reid.onnx` from `cache_dir`.
    ///
    /// Looks up `person_detect.onnx` then `yolov8n.onnx`. Optional
    /// `face_embed.onnx` is loaded when present.
    ///
    /// # Errors
    ///
    /// Missing weights / parse / optimize. Callers that want CI-friendly skip
    /// should treat [`HostError::ModelNotFound`] as exit 2.
    #[cfg(feature = "onnx")]
    pub fn from_onnx_cache(
        name: impl Into<String>,
        cache_dir: impl AsRef<Path>,
    ) -> Result<Self, HostError> {
        let cache = cache_dir.as_ref();
        let detect_path = first_existing(cache, &["person_detect.onnx", "yolov8n.onnx"])
            .ok_or_else(|| {
                HostError::ModelNotFound(format!(
                    "no person_detect.onnx / yolov8n.onnx under {}",
                    cache.display()
                ))
            })?;
        let reid_path = first_existing(cache, &["person_reid.onnx"]).ok_or_else(|| {
            HostError::ModelNotFound(format!("no person_reid.onnx under {}", cache.display()))
        })?;
        let mut detect_spec = ModelSpec::detector("person_detect", ModelTask::PersonDetect);
        detect_spec.local_path = Some(detect_path);
        let mut reid_spec = ModelSpec::embedder("person_reid", ModelTask::PersonReId, 512);
        reid_spec.local_path = Some(reid_path);
        reid_spec.preprocess = crate::PreprocessConfig::imagenet_like(128, 256);
        let detector = OnnxDetector::load(detect_spec, cache)?;
        let embedder = OnnxEmbedder::load(reid_spec, cache, EmbeddingTask::PersonReId)?;
        let face = first_existing(cache, &["face_embed.onnx"])
            .map(|path| {
                let mut spec = ModelSpec::embedder("face_embed", ModelTask::FaceEmbed, 512);
                spec.local_path = Some(path);
                spec.preprocess = crate::PreprocessConfig::imagenet_like(112, 112);
                OnnxEmbedder::load(spec, cache, EmbeddingTask::Face)
            })
            .transpose()?;
        let mut pipe = Self::new(name)?;
        pipe.onnx = Some(OnnxSlot {
            person_detect: detector,
            person_reid: embedder,
            face_embed: face,
        });
        Ok(pipe)
    }

    /// Load ONNX backends from a [`ModelManifest`] (resolved via `cache_dir`).
    ///
    /// `person_detect` and `person_reid` are required. `face_embed` is optional
    /// (`ModelNotFound` is ignored for that slot).
    ///
    /// # Errors
    ///
    /// Missing required weights / parse / optimize.
    #[cfg(feature = "onnx")]
    pub fn from_manifest(
        name: impl Into<String>,
        manifest: &ModelManifest,
    ) -> Result<Self, HostError> {
        let cache = manifest.cache_dir.as_path();
        let bundle = manifest.to_bundle_config();
        let detect_spec = bundle
            .person_detect
            .clone()
            .ok_or_else(|| HostError::Config("manifest has no person_detect model".into()))?;
        let reid_spec = bundle
            .person_reid
            .clone()
            .ok_or_else(|| HostError::Config("manifest has no person_reid model".into()))?;
        let detector = OnnxDetector::load(detect_spec, cache)?;
        let embedder = OnnxEmbedder::load(reid_spec, cache, EmbeddingTask::PersonReId)?;
        let face = match bundle.face_embed {
            Some(spec) => match OnnxEmbedder::load(spec, cache, EmbeddingTask::Face) {
                Ok(e) => Some(e),
                Err(HostError::ModelNotFound(_)) => None,
                Err(e) => return Err(e),
            },
            None => None,
        };
        let mut pipe = Self::new(name)?;
        pipe.onnx = Some(OnnxSlot {
            person_detect: detector,
            person_reid: embedder,
            face_embed: face,
        });
        Ok(pipe)
    }

    /// Install already-loaded ONNX models (replaces reference on detect/embed).
    ///
    /// # Errors
    ///
    /// Tracker config errors from [`HostPipeline::new`].
    #[cfg(feature = "onnx")]
    pub fn with_onnx(
        name: impl Into<String>,
        detector: OnnxDetector,
        embedder: OnnxEmbedder,
    ) -> Result<Self, HostError> {
        let mut pipe = Self::new(name)?;
        pipe.onnx = Some(OnnxSlot {
            person_detect: detector,
            person_reid: embedder,
            face_embed: None,
        });
        Ok(pipe)
    }

    /// JPEG/PNG enroll (person re-id). Requires feature `image-decode`.
    ///
    /// # Errors
    ///
    /// Empty buffer, missing `image-decode`, decode / embed / gallery failures.
    pub fn enroll_photo(&mut self, jpeg: &[u8]) -> Result<SubjectId, HostError> {
        require_encoded_photo(jpeg, "enroll_photo")?;
        self.enroll_photos(&[PhotoView::from_encoded(jpeg)], false)
    }

    /// Enrolls a subject from one or more photos (host embed → gallery).
    ///
    /// Encoded [`PhotoView`]s need feature `image-decode`.
    ///
    /// # Errors
    ///
    /// Embed / gallery failures.
    pub fn enroll_photos(
        &mut self,
        photos: &[PhotoView<'_>],
        face: bool,
    ) -> Result<SubjectId, HostError> {
        if photos.is_empty() {
            return Err(HostError::Runtime("enroll_photos: empty".into()));
        }
        let mut vectors = Vec::with_capacity(photos.len());
        for p in photos {
            vectors.push(self.embed_photo(p, face)?);
        }
        let modality = if face {
            SubjectModality::Face
        } else {
            SubjectModality::PersonAppearance
        };
        self.session
            .enroll_subject_photos(modality, &vectors)
            .map_err(|e| HostError::Runtime(format!("enroll: {e}")))
    }

    /// JPEG/PNG → multi-factor gallery search (person re-id).
    ///
    /// Requires feature `image-decode`. This is the Host `search_photo(jpeg, top_k)` path.
    ///
    /// # Errors
    ///
    /// Empty buffer, missing `image-decode`, embed / search failures.
    pub fn search_photo_jpeg(
        &mut self,
        jpeg: &[u8],
        top_k: usize,
    ) -> Result<Vec<PhotoSearchHit>, HostError> {
        require_encoded_photo(jpeg, "search_photo")?;
        self.search_photo(&PhotoView::from_encoded(jpeg), false, top_k)
    }

    /// Photo → embedding → multi-factor gallery search.
    ///
    /// Pass [`PhotoView::from_encoded`] for JPEG/PNG (feature `image-decode`).
    ///
    /// # Errors
    ///
    /// Embed / search failures.
    pub fn search_photo(
        &mut self,
        photo: &PhotoView<'_>,
        face: bool,
        top_k: usize,
    ) -> Result<Vec<PhotoSearchHit>, HostError> {
        #[cfg(feature = "onnx")]
        if let Some(onnx) = self.onnx.as_mut() {
            return if face {
                let adapter = onnx
                    .face_embed
                    .as_mut()
                    .ok_or_else(|| HostError::ModelNotFound("no face_embed.onnx loaded".into()))?;
                self.session
                    .search_photo_with_adapter(photo, adapter, top_k)
                    .map_err(|e| HostError::Runtime(format!("search: {e}")))
            } else {
                self.session
                    .search_photo_with_adapter(photo, &mut onnx.person_reid, top_k)
                    .map_err(|e| HostError::Runtime(format!("search: {e}")))
            };
        }
        if face {
            self.session
                .search_photo_with_adapter(photo, &mut self.models.face_embed, top_k)
                .map_err(|e| HostError::Runtime(format!("search: {e}")))
        } else {
            self.session
                .search_photo_with_adapter(photo, &mut self.models.person_reid, top_k)
                .map_err(|e| HostError::Runtime(format!("search: {e}")))
        }
    }

    /// Detect on one RGB frame (no tracking).
    ///
    /// # Errors
    ///
    /// Detector failures.
    pub fn detect_frame(
        &mut self,
        stamp: FrameStamp,
        frame: &FrameView<'_>,
    ) -> Result<Vec<Detection>, HostError> {
        #[cfg(feature = "onnx")]
        if let Some(onnx) = self.onnx.as_mut() {
            return onnx
                .person_detect
                .detect(stamp, frame)
                .map_err(|e| HostError::Runtime(format!("detect: {e}")));
        }
        self.models
            .person_detect
            .detect(stamp, frame)
            .map_err(|e| HostError::Runtime(format!("detect: {e}")))
    }

    /// Detect persons → track → embed each track on this frame.
    ///
    /// # Errors
    ///
    /// Detect / track / embed failures.
    pub fn ingest_frame(
        &mut self,
        stamp: FrameStamp,
        frame: &FrameView<'_>,
    ) -> Result<Vec<TrackedDetection>, HostError> {
        #[cfg(feature = "onnx")]
        if let Some(onnx) = self.onnx.as_mut() {
            return self
                .session
                .detect_ingest_and_embed_tracks(
                    stamp,
                    frame,
                    &mut onnx.person_detect,
                    &mut onnx.person_reid,
                )
                .map_err(|e| HostError::Runtime(format!("ingest: {e}")));
        }
        self.session
            .detect_ingest_and_embed_tracks(
                stamp,
                frame,
                &mut self.models.person_detect,
                &mut self.models.person_reid,
            )
            .map_err(|e| HostError::Runtime(format!("ingest: {e}")))
    }

    /// Detect + track only (no re-id embed). Use with `embed_every` skip.
    ///
    /// # Errors
    ///
    /// Detect / track failures.
    pub fn ingest_frame_track_only(
        &mut self,
        stamp: FrameStamp,
        frame: &FrameView<'_>,
    ) -> Result<Vec<TrackedDetection>, HostError> {
        #[cfg(feature = "onnx")]
        if let Some(onnx) = self.onnx.as_mut() {
            return self
                .session
                .detect_and_ingest(stamp, frame, &mut onnx.person_detect)
                .map_err(|e| HostError::Runtime(format!("track: {e}")));
        }
        self.session
            .detect_and_ingest(stamp, frame, &mut self.models.person_detect)
            .map_err(|e| HostError::Runtime(format!("track: {e}")))
    }

    /// Persist the live session package (`VisionIndex` + `gallery.json`).
    ///
    /// # Errors
    ///
    /// Package I/O failures.
    pub fn save_package(&self, dir: impl AsRef<Path>) -> Result<(), HostError> {
        self.session
            .save_package(dir)
            .map_err(|e| HostError::Runtime(format!("save_package: {e}")))
    }

    fn embed_photo(&mut self, photo: &PhotoView<'_>, face: bool) -> Result<Vec<f32>, HostError> {
        #[cfg(feature = "onnx")]
        if let Some(onnx) = self.onnx.as_mut() {
            return if face {
                onnx.face_embed
                    .as_mut()
                    .ok_or_else(|| HostError::ModelNotFound("no face_embed.onnx loaded".into()))?
                    .embed_photo(photo)
                    .map_err(|e| HostError::Runtime(format!("{e}")))
            } else {
                onnx.person_reid
                    .embed_photo(photo)
                    .map_err(|e| HostError::Runtime(format!("{e}")))
            };
        }
        if face {
            self.models
                .face_embed
                .embed_photo(photo)
                .map_err(|e| HostError::Runtime(format!("{e}")))
        } else {
            self.models
                .person_reid
                .embed_photo(photo)
                .map_err(|e| HostError::Runtime(format!("{e}")))
        }
    }

    /// Borrow session.
    #[must_use]
    pub fn session(&self) -> &IndexSession {
        &self.session
    }

    /// Mutable session.
    pub fn session_mut(&mut self) -> &mut IndexSession {
        &mut self.session
    }
}

fn require_encoded_photo(bytes: &[u8], what: &str) -> Result<(), HostError> {
    if bytes.is_empty() {
        return Err(HostError::Runtime(format!("{what}: empty")));
    }
    #[cfg(not(feature = "image-decode"))]
    {
        let _ = what;
        return Err(HostError::Preprocess(
            "JPEG/PNG photo path requires feature `image-decode`".into(),
        ));
    }
    #[cfg(feature = "image-decode")]
    {
        let _ = crate::decode::decode_encoded_rgb(bytes)?;
        Ok(())
    }
}

#[cfg(feature = "onnx")]
fn first_existing(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    names.iter().map(|n| dir.join(n)).find(|p| p.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sightloom::core::{MediaTime, SourceId};
    use sightloom::{FrameView, PhotoView, PixelFormat};

    fn solid_rgb(width: u32, height: u32, red: u8, green: u8, blue: u8) -> Vec<u8> {
        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for _ in 0..(width * height) {
            pixels.push(red);
            pixels.push(green);
            pixels.push(blue);
        }
        pixels
    }

    #[test]
    fn photo_enroll_and_search_ranks_same_subject() {
        let mut pipe = HostPipeline::new("host-test").unwrap();
        let a = solid_rgb(32, 64, 200, 40, 40);
        let b = solid_rgb(32, 64, 40, 40, 200);
        let fa = FrameView::new(32, 64, 32 * 3, PixelFormat::Rgb8, &a);
        let fb = FrameView::new(32, 64, 32 * 3, PixelFormat::Rgb8, &b);
        let sid = pipe
            .enroll_photos(&[PhotoView::from_frame(fa)], false)
            .unwrap();
        // Different person
        let _ = pipe
            .enroll_photos(&[PhotoView::from_frame(fb)], false)
            .unwrap();

        let hits = pipe
            .search_photo(&PhotoView::from_frame(fa), false, 3)
            .unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].subject_id, sid);
    }

    #[test]
    fn ingest_frame_tracks_and_embeds() {
        let mut pipe = HostPipeline::new("ingest").unwrap();
        let pixels = solid_rgb(64, 64, 10, 20, 30);
        let frame = FrameView::new(64, 64, 64 * 3, PixelFormat::Rgb8, &pixels);
        let stamp = FrameStamp::new(
            SourceId(1),
            0,
            MediaTime::new(0, 1_000_000_000).unwrap(),
            None,
        );
        let tracked = pipe.ingest_frame(stamp, &frame).unwrap();
        assert!(!tracked.is_empty());
    }

    #[test]
    fn detect_frame_reference_returns_box() {
        let mut pipe = HostPipeline::new("detect").unwrap();
        let pixels = solid_rgb(32, 32, 8, 8, 8);
        let frame = FrameView::new(32, 32, 32 * 3, PixelFormat::Rgb8, &pixels);
        let stamp = FrameStamp::new(
            SourceId(1),
            0,
            MediaTime::new(0, 1_000_000_000).unwrap(),
            None,
        );
        let dets = pipe.detect_frame(stamp, &frame).unwrap();
        assert!(!dets.is_empty());
    }

    #[cfg(feature = "image-decode")]
    fn solid_png(width: u32, height: u32, red: u8, green: u8, blue: u8) -> Vec<u8> {
        use image::{ImageBuffer, ImageFormat, Rgb};
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_pixel(width, height, Rgb([red, green, blue]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    #[cfg(feature = "image-decode")]
    #[test]
    fn enroll_and_search_jpeg_reference_accepts_same_rejects_other() {
        use sightloom::reid::MatchDecision;
        let mut pipe = HostPipeline::new("jpeg-ref").unwrap();
        pipe.session_mut()
            .set_resolve_config(sightloom::reid::ResolveConfig {
                accept_threshold: 0.75,
                reject_threshold: 0.25,
                require_same_modality: true,
                negative_reject_threshold: 0.95,
                ..sightloom::reid::ResolveConfig::default()
            })
            .unwrap();
        let alice = solid_png(32, 64, 220, 40, 40);
        let bob = solid_png(32, 64, 40, 40, 220);
        let sid = pipe.enroll_photo(&alice).unwrap();
        let _ = pipe.enroll_photo(&bob).unwrap();
        let hits = pipe.search_photo_jpeg(&alice, 3).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].subject_id, sid);
        assert_eq!(hits[0].decision, MatchDecision::Accept);
        let other = pipe.search_photo_jpeg(&bob, 3).unwrap();
        assert!(!other.is_empty());
        let alice_hit = other.iter().find(|h| h.subject_id == sid);
        if let Some(hit) = alice_hit {
            assert_ne!(hit.decision, MatchDecision::Accept);
        }
    }

    #[cfg(feature = "onnx")]
    fn onnx_cache_dir() -> PathBuf {
        if let Ok(p) = std::env::var("SIGHTLOOM_MODELS") {
            return PathBuf::from(p);
        }
        let here = PathBuf::from(".sightloom-models");
        if here.is_dir() {
            return here;
        }
        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.sightloom-models");
        if workspace.is_dir() { workspace } else { here }
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn onnx_weights_optional_skip_when_missing() {
        match HostPipeline::from_onnx_cache("skip", onnx_cache_dir()) {
            Ok(pipe) => assert!(pipe.uses_onnx()),
            Err(HostError::ModelNotFound(_)) => {
                eprintln!("skip: no ONNX weights under .sightloom-models/");
            }
            Err(e) => panic!("unexpected from_onnx_cache error: {e}"),
        }
    }

    #[cfg(feature = "full")]
    #[test]
    fn onnx_multi_person_detect_and_photo_rank() {
        use sightloom::reid::MatchDecision;
        let cache = onnx_cache_dir();
        let detect = first_existing(&cache, &["person_detect.onnx", "yolov8n.onnx"]);
        let scene = ["bus.jpg", "scene.jpg", "people.jpg"]
            .iter()
            .map(|n| cache.join(n))
            .find(|p| p.is_file());
        if let (Some(det), Some(img)) = (detect, scene) {
            let mut detect_spec =
                crate::ModelSpec::detector("person_detect", crate::ModelTask::PersonDetect);
            detect_spec.local_path = Some(det);
            let detector = crate::OnnxDetector::load(detect_spec, &cache)
                .expect("person_detect ONNX must load");
            let decoded = crate::decode::decode_encoded_rgb(&std::fs::read(&img).unwrap())
                .expect("scene image");
            let frame = FrameView::new(
                decoded.width,
                decoded.height,
                (decoded.width as usize) * 3,
                PixelFormat::Rgb8,
                &decoded.rgb,
            );
            let stamp = FrameStamp::new(
                SourceId(1),
                0,
                MediaTime::new(0, 1_000_000_000).unwrap(),
                None,
            );
            let mut detector = detector;
            let dets = detector.detect(stamp, &frame).expect("detect");
            assert!(
                dets.len() >= 2,
                "expected ≥2 people on {} (got {})",
                img.display(),
                dets.len()
            );
        } else {
            eprintln!(
                "skip detect: need person_detect.onnx + a scene JPEG under .sightloom-models/"
            );
        }

        let mut pipe = match HostPipeline::from_onnx_cache("onnx-rank", &cache) {
            Ok(p) => p,
            Err(HostError::ModelNotFound(_)) => {
                eprintln!("skip rank: no person_reid.onnx under .sightloom-models/");
                return;
            }
            Err(e) => panic!("onnx load: {e}"),
        };
        pipe.session_mut()
            .set_resolve_config(sightloom::reid::ResolveConfig {
                accept_threshold: 0.75,
                reject_threshold: 0.25,
                require_same_modality: true,
                negative_reject_threshold: 0.95,
                ..sightloom::reid::ResolveConfig::default()
            })
            .unwrap();
        let a_path = cache.join("person_a.jpg");
        let b_path = cache.join("person_b.jpg");
        if !(a_path.is_file() && b_path.is_file()) {
            eprintln!("skip rank: drop person_a.jpg + person_b.jpg next to weights");
            return;
        }
        let alice = std::fs::read(&a_path).unwrap();
        let bob = std::fs::read(&b_path).unwrap();
        let sid = pipe.enroll_photo(&alice).unwrap();
        let hits = pipe.search_photo_jpeg(&alice, 3).unwrap();
        assert_eq!(hits[0].subject_id, sid);
        assert_eq!(hits[0].decision, MatchDecision::Accept);
        let _ = pipe.enroll_photo(&bob).unwrap();
        let other = pipe.search_photo_jpeg(&bob, 3).unwrap();
        let alice_as_bob = other.iter().find(|h| h.subject_id == sid);
        if let Some(hit) = alice_as_bob {
            assert_ne!(hit.decision, MatchDecision::Accept);
        }
    }
}
