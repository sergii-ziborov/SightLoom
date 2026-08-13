//! Auto appearances / visits / subject profiles / redaction provenance.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use sightloom::{IndexSession, MemoryAutoRebuild};
use sightloom_core::{FrameStamp, MediaTime, Rect, SourceId};
use sightloom_index::{MemoryBuildConfig, RedactionIntent, SourceEntry};
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

    session.set_subject_label(seed.subject_id, "seed-person");
    let n_subj = session.rebuild_subject_profiles();
    assert_eq!(n_subj, 1);
    let profile = &session.index().subjects[0];
    assert_eq!(profile.subject_id, seed.subject_id);
    assert_eq!(profile.label.as_deref(), Some("seed-person"));
    assert!(profile.appearance_count >= 1);
    assert_eq!(profile.source_count, 1);
    assert!(profile.first_seen.is_some());
    assert!(profile.last_seen.is_some());

    let n_redact = session.plan_redaction_subject(seed.subject_id, 42);
    assert!(n_redact >= 1);
    assert_eq!(session.index().redaction_intervals.len(), n_redact);
    assert_eq!(
        session.index().redaction_intervals[0].intent,
        RedactionIntent::BlurSubject
    );
    assert_eq!(
        session.index().redaction_intervals[0].subject_id,
        Some(seed.subject_id)
    );
    assert_eq!(session.index().redaction_intervals[0].tag, 42);
    let json = session.export_redaction_intervals_json().unwrap();
    assert!(json.contains("blur_subject"));
    assert!(json.contains("\"interval_id\""));

    let (a3, v3, s3) = session.rebuild_memory_from_tracks();
    assert_eq!(a3, n_app);
    assert_eq!(v3, n_vis);
    assert_eq!(s3, 1);
    // Label preserved across profile rebuild.
    assert_eq!(
        session.index().subjects[0].label.as_deref(),
        Some("seed-person")
    );
}

#[test]
fn auto_rebuild_memory_every_n_frames() {
    let mut session = IndexSession::new("auto-mem", track_config()).unwrap();
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
    session.set_memory_auto_rebuild(MemoryAutoRebuild {
        every_n_frames: 3,
        rebuild_profiles: true,
    });

    let stamp0 = FrameStamp::new(SourceId(1), 0, MediaTime::new(0, 30).unwrap(), None);
    let seed = session
        .seed_click(stamp0, Rect::new(0.0, 0.0, 20.0, 40.0).unwrap(), 0.9, None)
        .unwrap();
    // seed_click = 1 accepted frame
    assert_eq!(session.frames_since_memory_rebuild(), 1);
    assert!(session.index().appearances.is_empty());

    for frame in 1..=2 {
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
    // On 3rd accepted frame (seed + 2), auto rebuild fires and counter resets.
    assert_eq!(session.frames_since_memory_rebuild(), 0);
    let last = session.last_auto_memory_rebuild().expect("auto rebuild");
    assert!(last.0 >= 1);
    assert_eq!(last.1, 1);
    assert_eq!(last.2, 1);
    assert!(!session.index().appearances.is_empty());
    assert_eq!(session.index().subjects[0].subject_id, seed.subject_id);

    // Batch path also counts frames.
    let batch = vec![
        (
            FrameStamp::new(SourceId(1), 3, MediaTime::new(3, 30).unwrap(), None),
            vec![
                sightloom_core::Detection::new(
                    Rect::new(3.0, 0.0, 23.0, 40.0).unwrap(),
                    0.9,
                    None,
                    None,
                )
                .unwrap(),
            ],
        ),
        (
            FrameStamp::new(SourceId(1), 4, MediaTime::new(4, 30).unwrap(), None),
            vec![
                sightloom_core::Detection::new(
                    Rect::new(4.0, 0.0, 24.0, 40.0).unwrap(),
                    0.9,
                    None,
                    None,
                )
                .unwrap(),
            ],
        ),
    ];
    let out = session.ingest_detection_batch(&batch).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(session.frames_since_memory_rebuild(), 2);
}
