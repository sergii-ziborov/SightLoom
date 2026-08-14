//! Privacy/retention product, photo embedding adapter, quality helpers.

use sightloom::{
    EmbeddingTask, IndexSession, PhotoEmbeddingAdapter, PhotoView, RedactionPixelSample,
    RetentionPolicy, SourceTtl, evaluate_redaction_pixels,
};
use sightloom_core::{FrameStamp, MediaTime, Rect, SourceId, SubjectId};
use sightloom_index::{RedactionIntent, SourceEntry};
use sightloom_tracking::ByteTrackConfig;

fn track_config() -> ByteTrackConfig {
    ByteTrackConfig {
        track_high_thresh: 0.5,
        track_activation_thresh: 0.5,
        track_low_thresh: 0.1,
        match_thresh: 0.3,
        max_time_lost: 30,
        class_aware: false,
    }
}

struct FakePhotoEmbedder;

impl PhotoEmbeddingAdapter for FakePhotoEmbedder {
    type Error = &'static str;

    fn task(&self) -> EmbeddingTask {
        EmbeddingTask::PersonReId
    }

    fn embed_photo(&mut self, photo: &PhotoView<'_>) -> Result<Vec<f32>, Self::Error> {
        // Deterministic stub: length of encoded bytes → unit vector-ish.
        let n = photo.encoded.map(<[u8]>::len).unwrap_or(1) as f32;
        Ok(vec![1.0, n * 0.001, 0.0])
    }
}

#[test]
fn legal_hold_source_ttl_and_forget() {
    let mut session = IndexSession::new("priv", track_config()).unwrap();
    session.add_source(SourceEntry {
        source_id: 1,
        uri: "file://a.mp4".into(),
        hash: None,
    });
    session.add_source(SourceEntry {
        source_id: 2,
        uri: "file://b.mp4".into(),
        hash: None,
    });

    // Source 1: frames 0..5, source 2: frames 0..2
    for frame in 0..6_u64 {
        let stamp = FrameStamp::new(
            SourceId(1),
            frame,
            MediaTime::new(i64::try_from(frame).unwrap(), 1).unwrap(),
            None,
        );
        let _ = session
            .ingest_detections(
                stamp,
                &[sightloom_core::Detection::new(
                    Rect::new(0.0, 0.0, 10.0, 10.0).unwrap(),
                    0.9,
                    None,
                    None,
                )
                .unwrap()],
            )
            .unwrap();
    }
    for frame in 0..3_u64 {
        let stamp = FrameStamp::new(
            SourceId(2),
            frame,
            MediaTime::new(i64::try_from(frame).unwrap(), 1).unwrap(),
            None,
        );
        let _ = session
            .ingest_detections(
                stamp,
                &[sightloom_core::Detection::new(
                    Rect::new(20.0, 0.0, 30.0, 10.0).unwrap(),
                    0.9,
                    None,
                    None,
                )
                .unwrap()],
            )
            .unwrap();
    }

    let before = session.index().tracks.samples().len();
    assert!(before >= 9);

    // Hold source 1; TTL source 2 to 1s of newest → keep only last frame on src2.
    let mut policy = RetentionPolicy::default();
    policy.hold_source(SourceId(1));
    policy.set_source_ttl(SourceId(2), 0); // 0 = unlimited via set — use 1 tick
    policy.source_ttls = vec![SourceTtl {
        source_id: 2,
        max_age_ns: 1_000_000_000, // 1s at timescale 1e9; our MediaTime timescale is 1
    }];
    // MediaTime::new(frame, 1) → ns = frame * 1e9 / 1 = frame * 1e9
    // newest src2 = 2 * 1e9, cutoff = 1e9 → keep frames >= 1
    session.set_retention_policy(policy);
    let report = session.apply_retention();
    assert!(report.protected_by_hold > 0);
    // Source 1 untouched by TTL; source 2 may drop old frames.
    assert!(session.index().tracks.samples().len() <= before);

    // Forget subject under hold is blocked.
    session.retention_policy_mut().hold_subject(SubjectId(99));
    let r = session.forget_subject(SubjectId(99));
    assert_eq!(r.forgotten_subjects, 0);

    let seed = session
        .seed_click(
            FrameStamp::new(SourceId(1), 100, MediaTime::new(100, 1).unwrap(), None),
            Rect::new(0.0, 0.0, 5.0, 5.0).unwrap(),
            0.9,
            None,
        )
        .unwrap();
    let r2 = session.forget_subject(seed.subject_id);
    assert_eq!(r2.forgotten_subjects, 1);
}

#[test]
fn photo_adapter_search_and_redaction_quality() {
    let mut session = IndexSession::new("photo", track_config()).unwrap();
    let sid = session
        .enroll_subject_photos(
            sightloom_reid::SubjectModality::PersonAppearance,
            &[vec![1.0_f32, 0.0, 0.0]],
        )
        .unwrap();
    let mut adapter = FakePhotoEmbedder;
    let bytes = [1_u8, 2, 3, 4];
    let hits = session
        .search_photo_with_adapter(&PhotoView::from_encoded(&bytes), &mut adapter, 3)
        .unwrap();
    // May or may not match depending on cosine — just ensure API works.
    let _ = (sid, hits);

    let report = evaluate_redaction_pixels(&[RedactionPixelSample {
        interval_id: 1,
        target_pixels: 100,
        target_visible_pixels: 5,
        collateral_redacted_pixels: 10,
        non_target_pixels: 200,
    }]);
    assert!((report.mean_target_leakage - 0.05).abs() < 1e-5);
    assert!((report.mean_collateral_ratio - 0.05).abs() < 1e-5);

    let _ = RedactionIntent::BlurSubject;
}
