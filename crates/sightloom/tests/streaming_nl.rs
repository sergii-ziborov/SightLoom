//! Streaming subject query + deterministic NL bridge.

use sightloom::IndexSession;
use sightloom_core::{FrameStamp, MediaTime, Rect, SourceId};
use sightloom_index::{SourceEntry, SubjectQuery};
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
fn stream_pages_and_nl_query() {
    let mut session = IndexSession::new("nl", track_config()).unwrap();
    session.add_source(SourceEntry {
        source_id: 1,
        uri: "file://a.mp4".into(),
        hash: None,
    });

    for frame in 0..4_u64 {
        let stamp = FrameStamp::new(
            SourceId(1),
            frame,
            MediaTime::new(i64::try_from(frame).unwrap(), 30).unwrap(),
            None,
        );
        let x = frame as f32 * 30.0;
        let tracked = session
            .ingest_detections(
                stamp,
                &[sightloom_core::Detection::new(
                    Rect::new(x, 0.0, x + 10.0, 20.0).unwrap(),
                    0.9,
                    None,
                    None,
                )
                .unwrap()],
            )
            .unwrap();
        // Assign distinct subjects for first three tracks when new.
        if let Some(item) = tracked.first() {
            let sid = session.register_subject(sightloom_reid::SubjectModality::PersonAppearance);
            session.assign_subject(item.track_key, sid);
        }
    }

    let mut stream = session.stream_subjects(SubjectQuery::new().seen_on(SourceId(1)), 2);
    let p1 = session.stream_next_page(&mut stream);
    assert_eq!(p1.len(), 2);
    let p2 = session.stream_next_page(&mut stream);
    assert!(!p2.is_empty());

    let (parsed, hits) = session
        .query_nl("seen on source 1 and min confidence 0.5")
        .unwrap();
    assert!(!hits.is_empty());
    assert!(parsed.warnings.is_empty() || parsed.warnings.iter().all(|w| w.contains("unknown")));
}
