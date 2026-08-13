//! Session re-id wireup: embeddings assign `subject_id` onto `VisionIndex` tracks/events.

use sightloom::IndexSession;
use sightloom_analysis::{AnchorPolicy, ZoneAnalytics, ZoneAnalyticsConfig};
use sightloom_core::{
    ClassId, Detection, FrameStamp, MediaTime, Point, Polygon, Rect, SourceId, TrackKey, ZoneId,
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
            ..ResolveConfig::default()
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
                quality: None,
                class_id: None,
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
    let key = tracked0[0].track_key;

    session
        .note_track_embedding(key, [0.98_f32, 0.02, 0.0], stamp0.pts)
        .unwrap();
    let (fragment, matches) = session
        .resolve_track_identity(key, None, stamp0.pts)
        .unwrap();
    assert_eq!(fragment.subject_id, Some(subject));
    assert_eq!(matches[0].decision, MatchDecision::Accept);
    assert_eq!(session.subject_for_track_key(key), Some(subject));

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
    assert_eq!(tracked1[0].track_key.local_track_id, key.local_track_id);
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
            ..ResolveConfig::default()
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
                quality: None,
                class_id: None,
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
    for item in &tracked {
        session
            .note_track_embedding(item.track_key, [0.01_f32, 0.99], stamp.pts)
            .unwrap();
    }
    let n = session.resolve_pending_identities(stamp.pts, None).unwrap();
    assert_eq!(n, 2);
    for item in &tracked {
        assert_eq!(session.subject_for_track_key(item.track_key), Some(subject));
    }
}

#[test]
fn multi_camera_local_track_ids_get_distinct_uids() {
    let mut session = IndexSession::new("campus", track_config()).unwrap();
    session.add_source(SourceEntry {
        source_id: 1,
        uri: "file://cam-a.mp4".into(),
        hash: None,
    });
    session.add_source(SourceEntry {
        source_id: 2,
        uri: "file://cam-b.mp4".into(),
        hash: None,
    });
    session
        .set_resolve_config(ResolveConfig {
            accept_threshold: 0.80,
            reject_threshold: 0.30,
            require_same_modality: true,
            negative_reject_threshold: 0.90,
            ..ResolveConfig::default()
        })
        .unwrap();

    let subject = session.register_subject(SubjectModality::PersonAppearance);
    let pos = session
        .gallery_mut()
        .embeddings
        .insert([1.0_f32, 0.0])
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
                quality: None,
                class_id: None,
            },
        )
        .unwrap();

    let stamp_a = FrameStamp::new(SourceId(1), 0, MediaTime::new(0, 30).unwrap(), None);
    let stamp_b = FrameStamp::new(SourceId(2), 0, MediaTime::new(0, 30).unwrap(), None);

    let cam_a = session
        .ingest_detections(stamp_a, &[det(10.0, 10.0, 30.0, 50.0, 0.9)])
        .unwrap();
    let cam_b = session
        .ingest_detections(stamp_b, &[det(200.0, 10.0, 230.0, 50.0, 0.91)])
        .unwrap();

    assert_eq!(cam_a.len(), 1);
    assert_eq!(cam_b.len(), 1);
    // Same local track id is allowed across cameras…
    assert_eq!(
        cam_a[0].track_key.local_track_id,
        cam_b[0].track_key.local_track_id
    );
    // …but global uids must differ.
    assert_ne!(cam_a[0].track_uid, cam_b[0].track_uid);
    assert_eq!(cam_a[0].track_key.local_track_id.0, 1);
    assert_eq!(cam_b[0].track_key.local_track_id.0, 1);

    // Observation queries do not mix sources.
    let key_a = cam_a[0].track_key;
    let key_b = cam_b[0].track_key;
    assert_eq!(session.index().tracks.for_track_key(key_a).len(), 1);
    assert_eq!(session.index().tracks.for_track_key(key_b).len(), 1);
    assert_eq!(
        session
            .index()
            .tracks
            .for_track_uid(cam_a[0].track_uid)
            .len(),
        1
    );
    assert_eq!(
        session
            .index()
            .tracks
            .for_track_uid(cam_b[0].track_uid)
            .len(),
        1
    );

    // Re-id may merge both into one SubjectId.
    session
        .note_track_embedding(key_a, [0.99_f32, 0.01], stamp_a.pts)
        .unwrap();
    session
        .note_track_embedding(key_b, [0.98_f32, 0.02], stamp_b.pts)
        .unwrap();
    let n = session
        .resolve_pending_identities(stamp_a.pts, None)
        .unwrap();
    assert_eq!(n, 2);
    assert_eq!(session.subject_for_track_key(key_a), Some(subject));
    assert_eq!(session.subject_for_track_key(key_b), Some(subject));
    // Keys remain distinct even when subjects match.
    assert_ne!(key_a, key_b);
    let _ = TrackKey::new(SourceId(1), cam_a[0].track_key.local_track_id);
}

#[test]
fn session_checkpoint_restores_runtime_and_continues_ingest() {
    let dir = tempfile::tempdir().unwrap();
    let mut session = IndexSession::new("edge", track_config()).unwrap();
    session.add_source(SourceEntry {
        source_id: 1,
        uri: "file://edge.mp4".into(),
        hash: None,
    });
    session.set_embedding_model_id("demo-embed-v1");
    let subject = session.register_subject(SubjectModality::PersonAppearance);
    let pos = session
        .gallery_mut()
        .embeddings
        .insert([1.0_f32, 0.0])
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
                quality: None,
                class_id: None,
            },
        )
        .unwrap();

    let stamp0 = FrameStamp::new(SourceId(1), 0, MediaTime::new(0, 30).unwrap(), None);
    let tracked0 = session
        .ingest_detections(stamp0, &[det(40.0, 40.0, 60.0, 60.0, 0.9)])
        .unwrap();
    let key = tracked0[0].track_key;
    let uid = tracked0[0].track_uid;
    session
        .note_track_embedding(key, [0.99_f32, 0.01], stamp0.pts)
        .unwrap();
    session
        .resolve_track_identity(key, None, stamp0.pts)
        .unwrap();
    assert_eq!(session.subject_for_track_key(key), Some(subject));

    session.save_checkpoint(dir.path()).unwrap();

    let mut restored = IndexSession::load_checkpoint(dir.path()).unwrap();
    assert_eq!(restored.subject_for_track_key(key), Some(subject));
    assert_eq!(restored.track_uid(key), Some(uid));
    assert_eq!(restored.gallery().subjects().len(), 1);
    assert!(!restored.gallery().embeddings.entries().is_empty());

    // Continue ingest: same local track id should reuse motion state / uid mapping.
    let stamp1 = FrameStamp::new(SourceId(1), 1, MediaTime::new(1, 30).unwrap(), None);
    let tracked1 = restored
        .ingest_detections(stamp1, &[det(42.0, 42.0, 62.0, 62.0, 0.88)])
        .unwrap();
    assert_eq!(tracked1[0].track_key.local_track_id, key.local_track_id);
    assert_eq!(tracked1[0].track_uid, uid);
    assert_eq!(
        restored.index().tracks.samples().last().unwrap().subject_id,
        Some(subject)
    );
}
