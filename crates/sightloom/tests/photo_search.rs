//! Reference-photo enrollment and search across enrolled subjects.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use sightloom::IndexSession;
use sightloom_core::{FrameStamp, MediaTime, Rect, SourceId};
use sightloom_index::SourceEntry;
use sightloom_reid::{MatchDecision, SubjectModality};
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

#[test]
fn enroll_three_photos_and_find_in_video_session() {
    let mut session = IndexSession::new("search", track_config()).unwrap();
    session.add_source(SourceEntry {
        source_id: 1,
        uri: "file://cam.mp4".into(),
        hash: None,
    });
    session
        .set_resolve_config(sightloom_reid::ResolveConfig {
            accept_threshold: 0.75,
            reject_threshold: 0.25,
            require_same_modality: true,
            negative_reject_threshold: 0.95,
            ..sightloom_reid::ResolveConfig::default()
        })
        .unwrap();

    // Three reference photos (host embeddings) for one person.
    let subject = session
        .enroll_subject_photos(
            SubjectModality::PersonAppearance,
            &[
                vec![1.0_f32, 0.0, 0.0],
                vec![0.98, 0.02, 0.0],
                vec![0.97, 0.03, 0.0],
            ],
        )
        .unwrap();

    // Video track for that person (seed + a few frames).
    let stamp0 = FrameStamp::new(SourceId(1), 0, MediaTime::new(0, 30).unwrap(), None);
    let seed = session
        .seed_click(
            stamp0,
            Rect::new(10.0, 10.0, 40.0, 80.0).unwrap(),
            0.9,
            Some(subject),
        )
        .unwrap();
    assert_eq!(seed.subject_id, subject);

    for frame in 1..4 {
        let stamp = FrameStamp::new(
            SourceId(1),
            frame,
            MediaTime::new(i64::try_from(frame).unwrap(), 30).unwrap(),
            None,
        );
        let tracked = session
            .ingest_detections(
                stamp,
                &[sightloom_core::Detection::new(
                    Rect::new(10.0 + frame as f32, 10.0, 40.0 + frame as f32, 80.0).unwrap(),
                    0.88,
                    None,
                    None,
                )
                .unwrap()],
            )
            .unwrap();
        if let Some(item) = tracked.first() {
            session.assign_subject(item.track_key, subject);
        }
    }

    // Query photo close to references → Accept + reel.
    let results = session
        .search_photo_with_reels(
            [0.99_f32, 0.01, 0.0],
            SubjectModality::PersonAppearance,
            5,
            1_000_000_000,
        )
        .unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].hit.subject_id, subject);
    assert_eq!(results[0].hit.decision, MatchDecision::Accept);
    let reel = results[0].reel.as_ref().expect("reel");
    assert!(!reel.is_empty());

    // Orthogonal photo should not Accept the same subject as best high score path
    // (may be Reject or Uncertain depending on thresholds).
    let other = session
        .search_by_photo([0.0_f32, 1.0, 0.0], SubjectModality::PersonAppearance, 3)
        .unwrap();
    if let Some(best) = other.first() {
        assert_ne!(best.decision, MatchDecision::Accept);
    }
}
