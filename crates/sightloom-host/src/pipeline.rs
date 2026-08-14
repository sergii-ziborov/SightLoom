//! End-to-end host pipelines: photo→rank and detect→track→embed.

use crate::error::HostError;
use crate::reference::ReferenceHostModels;
use sightloom::core::{FrameStamp, SubjectId};
use sightloom::reid::{PhotoSearchHit, SubjectModality};
use sightloom::tracking::{ByteTrackConfig, TrackedDetection};
use sightloom::{FrameView, IndexSession, PhotoEmbeddingAdapter, PhotoView};

/// Runs the killer path with **host** embedders + `SightLoom` ranking.
///
/// ```text
/// photo bytes/pixels
///   → ReferenceEmbedder (or future ONNX)
///   → IndexSession gallery rank
/// ```
pub struct HostPipeline {
    /// Live session.
    pub session: IndexSession,
    /// Host models (reference now; swap later).
    pub models: ReferenceHostModels,
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
        Ok(Self { session, models })
    }

    /// Enrolls a subject from one or more photos (host embed → gallery).
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
            let v = if face {
                self.models
                    .face_embed
                    .embed_photo(p)
                    .map_err(|e| HostError::Runtime(format!("{e}")))?
            } else {
                self.models
                    .person_reid
                    .embed_photo(p)
                    .map_err(|e| HostError::Runtime(format!("{e}")))?
            };
            vectors.push(v);
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

    /// Photo → embedding → multi-factor gallery search.
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
        self.session
            .detect_ingest_and_embed_tracks(
                stamp,
                frame,
                &mut self.models.person_detect,
                &mut self.models.person_reid,
            )
            .map_err(|e| HostError::Runtime(format!("ingest: {e}")))
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
}
