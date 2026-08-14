//! Host model stub: fake detector + photo embedder → SightLoom memory.
//!
//! Demonstrates the **honest** product boundary:
//! - host owns model weights / preprocessing (here: pure stubs)
//! - SightLoom owns tracks, subjects, ranking, VisionIndex
//!
//! ```bash
//! cargo run -p sightloom --example host_model_stub
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::doc_markdown
)]

use sightloom::{
    DetectorAdapter, EmbeddingTask, FrameView, IndexSession, PhotoEmbeddingAdapter, PhotoView,
    PixelFormat,
};
use sightloom_core::{Detection, FrameStamp, MediaTime, Rect, SourceId, generate_tiles};
use sightloom_index::SourceEntry;
use sightloom_tracking::ByteTrackConfig;

/// Stub person detector: one box walking right per frame.
struct StubPersonDetector {
    frame: u64,
}

impl DetectorAdapter for StubPersonDetector {
    type Error = &'static str;

    fn detect(
        &mut self,
        _stamp: FrameStamp,
        _frame: &FrameView<'_>,
    ) -> Result<Vec<Detection>, Self::Error> {
        let x = self.frame as f32 * 3.0;
        self.frame = self.frame.saturating_add(1);
        Ok(vec![
            Detection::new(
                Rect::new(40.0 + x, 40.0, 80.0 + x, 140.0).unwrap(),
                0.92,
                Some(sightloom_core::ClassId(0)),
                None,
            )
            .unwrap(),
        ])
    }
}

/// Stub re-id embedder: fixed prototype + tiny noise from byte length.
struct StubPersonEmbedder;

impl PhotoEmbeddingAdapter for StubPersonEmbedder {
    type Error = &'static str;

    fn task(&self) -> EmbeddingTask {
        EmbeddingTask::PersonReId
    }

    fn embed_photo(&mut self, photo: &PhotoView<'_>) -> Result<Vec<f32>, Self::Error> {
        let n = photo.encoded.map_or(8, <[u8]>::len) as f32;
        // Host would run ONNX/Torch here.
        Ok(vec![0.9, 0.1, n * 1e-4, 0.0])
    }
}

fn main() {
    let cfg = ByteTrackConfig {
        track_high_thresh: 0.5,
        track_activation_thresh: 0.5,
        track_low_thresh: 0.1,
        match_thresh: 0.3,
        max_time_lost: 30,
        class_aware: false,
    };
    let mut session = IndexSession::new("host-model-stub", cfg).expect("session");
    session.add_source(SourceEntry {
        source_id: 1,
        uri: "file://demo.mp4".into(),
        hash: None,
    });

    // --- Reference photo enrollment (host embeds, SightLoom stores) ---
    let mut embedder = StubPersonEmbedder;
    let ref_bytes = b"JPEG-BYTES-OF-PERSON-A";
    let ref_vec = embedder
        .embed_photo(&PhotoView::from_encoded(ref_bytes))
        .expect("embed");
    let subject = session
        .enroll_subject_photos(
            sightloom_reid::SubjectModality::PersonAppearance,
            &[ref_vec],
        )
        .expect("enroll");
    println!("enrolled subject={}", subject.0);

    // --- Tiled "4K" windows (host would crop + detect each tile) ---
    let tiles = generate_tiles(3840, 2160, 640, 64).expect("tiles");
    println!("4K tiling: {} tiles (tile=640 overlap=64)", tiles.len());

    // --- Fake video frames: detect → track → assign subject ---
    let mut detector = StubPersonDetector { frame: 0 };
    let blank = [0_u8; 16];
    for frame in 0..12_u64 {
        let stamp = FrameStamp::new(
            SourceId(1),
            frame,
            MediaTime::new(i64::try_from(frame).unwrap(), 30).unwrap(),
            None,
        );
        let view = FrameView::new(4, 4, 4, PixelFormat::Gray8, &blank);
        let tracked = session
            .detect_and_ingest(stamp, &view, &mut detector)
            .expect("detect");
        for item in &tracked {
            session.assign_subject(item.track_key, subject);
        }
    }

    // --- Query photo search via adapter ---
    let query_bytes = b"JPEG-BYTES-OF-PERSON-A-QUERY";
    let hits = session
        .search_photo_with_adapter(&PhotoView::from_encoded(query_bytes), &mut embedder, 3)
        .expect("search");
    println!("photo search hits={}", hits.len());
    for h in &hits {
        println!(
            "  subject={} score={:.3} decision={:?}",
            h.subject_id.0, h.score, h.decision
        );
    }

    let (apps, visits, profiles) = session.rebuild_memory_from_tracks();
    println!("memory appearances={apps} visits={visits} profiles={profiles}");
    let n = session.plan_redaction_subject(subject, 1);
    println!("redaction intervals planned={n}");
    println!("done (host owns models; SightLoom owns memory)");
}
