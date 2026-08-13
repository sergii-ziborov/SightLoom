//! End-to-end materialization: detections to tracks to zone events to `VisionIndex` JSON.

use sightloom::IndexSession;
use sightloom_analysis::{AnchorPolicy, ZoneAnalytics, ZoneAnalyticsConfig};
use sightloom_core::{
    ClassId, Detection, FrameStamp, MediaTime, Point, Polygon, Rect, SourceId, TrackKey, ZoneId,
};
use sightloom_index::SourceEntry;
use sightloom_tracking::ByteTrackConfig;

fn det(left: f32, top: f32, right: f32, bottom: f32, score: f32) -> Detection {
    Detection::new(
        Rect::new(left, top, right, bottom).unwrap(),
        score,
        Some(ClassId(0)),
        None,
    )
    .unwrap()
}

#[test]
fn detections_to_serialized_vision_index() {
    let mut session = IndexSession::new(
        "lobby",
        ByteTrackConfig {
            track_high_thresh: 0.5,
            track_activation_thresh: 0.5,
            track_low_thresh: 0.1,
            match_thresh: 0.3,
            max_time_lost: 30,
            class_aware: false,
        },
    )
    .unwrap();
    session.add_source(SourceEntry {
        source_id: 1,
        uri: "file://lobby.mp4".into(),
        hash: None,
    });

    let points = [
        Point::new(0.0, 0.0).unwrap(),
        Point::new(100.0, 0.0).unwrap(),
        Point::new(100.0, 100.0).unwrap(),
        Point::new(0.0, 100.0).unwrap(),
    ];
    let polygon = Polygon::new(&points).unwrap();
    let config = ZoneAnalyticsConfig {
        anchor: AnchorPolicy::Center,
        enter_hysteresis: 1,
        exit_hysteresis: 1,
        missed_frame_tolerance: 0,
        class_filter: None,
        dwell_start_debounce_ns: 0,
    };
    let mut zone: ZoneAnalytics<'_, 8> = ZoneAnalytics::new(ZoneId(1), polygon, config).unwrap();

    let stamp0 = FrameStamp::new(SourceId(1), 0, MediaTime::new(0, 30).unwrap(), None);
    let tracked0 = session
        .ingest_detections(stamp0, &[det(40.0, 40.0, 60.0, 60.0, 0.9)])
        .unwrap();
    assert_eq!(tracked0.len(), 1);
    assert!(tracked0[0].detection.track_id().is_some());
    let events0 = session
        .ingest_zone_updates(stamp0, &mut zone, &tracked0)
        .unwrap();
    assert!(events0 >= 1);

    let stamp1 = FrameStamp::new(SourceId(1), 1, MediaTime::new(1, 30).unwrap(), None);
    let tracked1 = session
        .ingest_detections(stamp1, &[det(42.0, 42.0, 62.0, 62.0, 0.88)])
        .unwrap();
    assert_eq!(
        tracked1[0].track_key.local_track_id,
        tracked0[0].track_key.local_track_id
    );
    let _ = session
        .ingest_zone_updates(stamp1, &mut zone, &tracked1)
        .unwrap();

    let mask_ref = session.store_mask_bytes([1_u8, 1, 0, 1]);
    assert!(session.attach_mask_to_latest_track(tracked1[0].track_key, mask_ref));

    let json = session.materialize_json().unwrap();
    assert!(json.contains("\"name\": \"lobby\""));
    assert!(json.contains("\"tracks\""));
    assert!(json.contains("\"events\""));
    let snap = sightloom_index::VisionIndexSnapshot::from_json(&json).unwrap();
    assert!(!snap.tracks.is_empty());
    assert!(!snap.events.is_empty());
    assert_eq!(snap.header.name, "lobby");

    // Sample carries a global track uid.
    assert!(snap.tracks[0].track_uid.is_some());
    let _ = TrackKey::new(SourceId(1), tracked0[0].track_key.local_track_id);
}
