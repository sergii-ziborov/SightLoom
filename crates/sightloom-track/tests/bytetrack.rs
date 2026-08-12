//! `ByteTrack` integration tests.

use sightloom_core::{ClassId, Detection, Rect, TrackId};
use sightloom_track::{ByteTrackConfig, ByteTracker, TrackState};

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
fn assigns_stable_ids_across_frames() {
    let mut tracker = ByteTracker::new(ByteTrackConfig::default()).unwrap();

    let frame1 = [det(10.0, 10.0, 50.0, 80.0, 0.9)];
    let out1 = tracker.update(&frame1).unwrap();
    assert_eq!(out1.len(), 1);
    let id = out1[0].track_id().expect("id");

    let frame2 = [det(12.0, 12.0, 52.0, 82.0, 0.88)];
    let out2 = tracker.update(&frame2).unwrap();
    assert_eq!(out2.len(), 1);
    assert_eq!(out2[0].track_id(), Some(id));
}

#[test]
fn creates_two_tracks_for_disjoint_detections() {
    let mut tracker = ByteTracker::new(ByteTrackConfig::default()).unwrap();
    let frame = [
        det(0.0, 0.0, 20.0, 40.0, 0.9),
        det(100.0, 0.0, 120.0, 40.0, 0.91),
    ];
    let out = tracker.update(&frame).unwrap();
    assert_eq!(out.len(), 2);
    let ids: Vec<TrackId> = out.iter().filter_map(|d| d.track_id()).collect();
    assert_ne!(ids[0], ids[1]);
}

#[test]
fn low_confidence_can_keep_lost_track() {
    let config = ByteTrackConfig {
        track_high_thresh: 0.6,
        track_activation_thresh: 0.6,
        track_low_thresh: 0.1,
        match_thresh: 0.3,
        max_time_lost: 30,
        class_aware: false,
    };
    let mut tracker = ByteTracker::new(config).unwrap();
    let _ = tracker.update(&[det(10.0, 10.0, 50.0, 80.0, 0.9)]).unwrap();
    // High conf continues
    let _ = tracker
        .update(&[det(12.0, 12.0, 52.0, 82.0, 0.85)])
        .unwrap();
    // Only low conf detection near the track
    let out = tracker.update(&[det(14.0, 14.0, 54.0, 84.0, 0.3)]).unwrap();
    // Track should still be associated via second stage
    assert_eq!(out.len(), 1);
}

#[test]
fn removes_tracks_after_lost_buffer() {
    let config = ByteTrackConfig {
        track_high_thresh: 0.5,
        track_activation_thresh: 0.5,
        track_low_thresh: 0.1,
        match_thresh: 0.8,
        max_time_lost: 2,
        class_aware: false,
    };
    let mut tracker = ByteTracker::new(config).unwrap();
    let _ = tracker.update(&[det(10.0, 10.0, 50.0, 80.0, 0.9)]).unwrap();
    let _ = tracker.update(&[]).unwrap();
    let _ = tracker.update(&[]).unwrap();
    let _ = tracker.update(&[]).unwrap();
    assert!(
        tracker
            .tracks()
            .iter()
            .all(|t| t.state != TrackState::Tracked && t.state != TrackState::New)
    );
}
