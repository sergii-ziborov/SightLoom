//! Auto appearances / visits from track samples.

use sightloom::IndexSession;
use sightloom_core::{FrameStamp, MediaTime, Rect, SourceId};
use sightloom_index::{MemoryBuildConfig, SourceEntry};
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
fn rebuild_appearances_and_visits_from_seeded_track() {
    let mut session = IndexSession::new("memory", track_config()).unwrap();
    session.add_source(SourceEntry {
        source_id: 1,
        uri: "file://a.mp4".into(),
        hash: None,
    });
    session.set_memory_build_config(MemoryBuildConfig {
        appearance_gap_ns: 1_000_000_000,
        visit_gap_ns: 60_000_000_000,
        require_subject: true,
    });

    let stamp0 = FrameStamp::new(SourceId(1), 0, MediaTime::new(0, 30).unwrap(), None);
    let seed = session
        .seed_click(stamp0, Rect::new(0.0, 0.0, 20.0, 40.0).unwrap(), 0.9, None)
        .unwrap();

    for frame in 1..=5 {
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
                    Rect::new(frame as f32, 0.0, 20.0 + frame as f32, 40.0).unwrap(),
                    0.9,
                    None,
                    None,
                )
                .unwrap()],
            )
            .unwrap();
        if let Some(item) = tracked.first() {
            session.assign_subject(item.track_key, seed.subject_id);
        }
    }

    let (n_app, n_vis) = session.rebuild_appearances_and_visits();
    assert!(n_app >= 1);
    assert_eq!(n_vis, 1);
    assert_eq!(session.index().appearances.len(), n_app);
    assert_eq!(session.index().visits.len(), n_vis);
    assert_eq!(
        session.index().appearances[0].subject_id,
        Some(seed.subject_id)
    );
    assert_eq!(session.index().visits[0].subject_id, Some(seed.subject_id));

    // Idempotent rebuild still works.
    let (n_app2, n_vis2) = session.rebuild_appearances_and_visits();
    assert_eq!(n_app2, n_app);
    assert_eq!(n_vis2, n_vis);
}
