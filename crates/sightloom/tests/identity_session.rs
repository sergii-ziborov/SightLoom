//! Session re-id wireup: embeddings assign `subject_id` onto `VisionIndex` tracks/events.

use sightloom::IndexSession;
use sightloom_analysis::{AnchorPolicy, ZoneAnalytics, ZoneAnalyticsConfig};
use sightloom_core::{
    ClassId, Detection, FrameStamp, MediaTime, Point, Polygon, Rect, SourceId, ZoneId,
};
use sightloom_index::SourceEntry;
use sightloom_reid::{MatchDecision, ReferenceSample, ResolveConfig, SubjectModality};
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
fn session_auto_assigns_subject_on_track_samples_and_zone_events() {
    let mut session = IndexSession::new("lobby", track_config()).unwrap();
    session.add_source(SourceEntry {
        source_id: 1,
        uri: "file://lobby.mp4".into(),
        hash: None,
    });
    session
        .set_resolve_config(ResolveConfig {
            accept_threshold: 0.80,
            reject_threshold: 0.30,
            require_same_modality: true,
            negative_reject_threshold: 0.90,
        })
        .unwrap();
    session.set_default_modality(SubjectModality::PersonAppearance);

    let subject = session.register_subject(SubjectModality::PersonAppearance);
    let pos = session
        .gallery_mut()
        .embeddings
        .insert([1.0_f32, 0.0, 0.0])
        .unwrap();
    session
        .add_subject_reference(
            subject,
            ReferenceSample {
                source_id: Some(SourceId(1)),
                track_id: None,
                at: None,
                embedding: Some(pos),
                evidence: None,
                is_positive: Some(true),
            },
        )
        .unwrap();

    let points = [
        Point::new(0.0, 0.0).unwrap(),
        Point::new(100.0, 0.0).unwrap(),
        Point::new(100.0, 100.0).unwrap(),
        Point::new(0.0, 100.0).unwrap(),
    ];
    let polygon = Polygon::new(&points).unwrap();
    let mut zone: ZoneAnalytics<'_, 8> = ZoneAnalytics::new(
        ZoneId(1),
        polygon,
        ZoneAnalyticsConfig {
            anchor: AnchorPolicy::Center,
            enter_hysteresis: 1,
            exit_hysteresis: 1,
            missed_frame_tolerance: 0,
            class_filter: None,
            dwell_start_debounce_ns: 0,
        },
    )
    .unwrap();

    let stamp0 = FrameStamp::new(SourceId(1), 0, MediaTime::new(0, 30).unwrap(), None);
    let tracked0 = session
        .ingest_detections(stamp0, &[det(40.0, 40.0, 60.0, 60.0, 0.9)])
        .unwrap();
    let track_id = tracked0[0].track_id().unwrap();

    session
        .note_track_embedding(track_id, [0.98_f32, 0.02, 0.0], stamp0.pts)
        .unwrap();
    let (fragment, matches) = session
        .resolve_track_identity(track_id, SourceId(1), None, stamp0.pts)
        .unwrap();
    assert_eq!(fragment.subject_id, Some(subject));
    assert_eq!(matches[0].decision, MatchDecision::Accept);
    assert_eq!(session.subject_for_track(track_id), Some(subject));

    // Latest track sample should be patched with subject_id.
    let last = session.index().tracks.samples().last().unwrap();
    assert_eq!(last.subject_id, Some(subject));

    let events = session
        .ingest_zone_updates(stamp0, &mut zone, &tracked0)
        .unwrap();
    assert!(events >= 1);
    let envelope = session.index().events.last().unwrap();
    assert_eq!(envelope.subject_id, Some(subject));

    // Next frame reuses the track→subject mapping without a new resolve.
    let stamp1 = FrameStamp::new(SourceId(1), 1, MediaTime::new(1, 30).unwrap(), None);
    let tracked1 = session
        .ingest_detections(stamp1, &[det(42.0, 42.0, 62.0, 62.0, 0.88)])
        .unwrap();
    assert_eq!(tracked1[0].track_id(), Some(track_id));
    let latest = session.index().tracks.samples().last().unwrap();
    assert_eq!(latest.subject_id, Some(subject));

    let json = session.materialize_json().unwrap();
    assert!(json.contains("\"subject_id\": 1") || json.contains("\"subject_id\":1"));
}

#[test]
fn resolve_pending_identities_handles_multiple_tracks() {
    let mut session = IndexSession::new("yard", track_config()).unwrap();
    session
        .set_resolve_config(ResolveConfig {
            accept_threshold: 0.75,
            reject_threshold: 0.25,
            require_same_modality: true,
            negative_reject_threshold: 0.9,
        })
        .unwrap();
    let subject = session.register_subject(SubjectModality::PersonAppearance);
    let pos = session
        .gallery_mut()
        .embeddings
        .insert([0.0_f32, 1.0])
        .unwrap();
    session
        .add_subject_reference(
            subject,
            ReferenceSample {
                source_id: None,
                track_id: None,
                at: None,
                embedding: Some(pos),
                evidence: None,
                is_positive: Some(true),
            },
        )
        .unwrap();

    let stamp = FrameStamp::new(SourceId(1), 0, MediaTime::new(0, 1).unwrap(), None);
    let tracked = session
        .ingest_detections(
            stamp,
            &[
                det(0.0, 0.0, 10.0, 20.0, 0.9),
                det(100.0, 0.0, 120.0, 20.0, 0.91),
            ],
        )
        .unwrap();
    assert_eq!(tracked.len(), 2);
    for detection in &tracked {
        let track_id = detection.track_id().unwrap();
        session
            .note_track_embedding(track_id, [0.01_f32, 0.99], stamp.pts)
            .unwrap();
    }
    let n = session.resolve_pending_identities(stamp, None).unwrap();
    assert_eq!(n, 2);
    for detection in &tracked {
        assert_eq!(
            session.subject_for_track(detection.track_id().unwrap()),
            Some(subject)
        );
    }
}
