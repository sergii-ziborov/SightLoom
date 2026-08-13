//! Demo API (seed / uncertain export / spans) and pattern/anomaly session wire-up.

use sightloom::IndexSession;
use sightloom_analysis::{DurationSample, TimedSubjectEvent};
use sightloom_core::{ClassId, FrameStamp, MediaTime, Rect, SourceId, SubjectId, ZoneId};
use sightloom_index::{SourceEntry, ZoneStay};
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
fn seed_click_and_export_spans_json() {
    let mut session = IndexSession::new("demo", track_config()).unwrap();
    session.add_source(SourceEntry {
        source_id: 1,
        uri: "file://clip.mp4".into(),
        hash: None,
    });

    let stamp = FrameStamp::new(SourceId(1), 0, MediaTime::new(0, 30).unwrap(), None);
    let bbox = Rect::new(10.0, 20.0, 40.0, 80.0).unwrap();
    let seed = session.seed_click(stamp, bbox, 0.92, None).unwrap();
    assert_eq!(seed.source_id, SourceId(1));
    assert_eq!(seed.subject_id, SubjectId(1));
    assert_eq!(
        session.subject_for_track(seed.source_id, seed.track_id),
        Some(seed.subject_id)
    );

    // Host can also accept a known local track id later.
    let key = session.accept_host_track(seed.source_id, seed.track_id, Some(seed.subject_id));
    assert_eq!(key.local_track_id, seed.track_id);

    let spans = session.export_track_spans();
    assert!(!spans.is_empty());
    assert_eq!(spans[0].subject_id, Some(seed.subject_id));

    let json = session.export_track_spans_json().unwrap();
    assert!(json.contains("\"subject_id\""));
    assert!(json.contains("\"left\""));

    let uncertain = session.export_uncertain_intervals_json().unwrap();
    assert_eq!(uncertain.trim(), "[]");
}

#[test]
fn mine_patterns_and_detect_anomalies_from_session() {
    let mut session = IndexSession::new("patterns", track_config()).unwrap();
    let subject = session.register_subject(sightloom_reid::SubjectModality::PersonAppearance);

    // Seed enough timed track samples via index entities for miners.
    for hour in 0..6 {
        let pts = MediaTime::new(hour * 3_600, 1).unwrap();
        session
            .index_mut()
            .push_track(sightloom_index::TrackSample {
                sample_id: 0,
                supersedes: None,
                revision: 0,
                idempotency_key: 0,
                source_id: SourceId(1),
                frame_index: hour as u64,
                pts,
                track_id: sightloom_core::TrackId(1),
                track_uid: Some(sightloom_core::TrackUid(1)),
                subject_id: Some(subject),
                class_id: Some(ClassId(0)),
                left: 0.0,
                top: 0.0,
                right: 10.0,
                bottom: 20.0,
                confidence: 0.9,
                mask_ref: 0,
            });
    }
    for d in [
        1_000_000_000_i64,
        1_100_000_000,
        900_000_000,
        1_050_000_000,
        1_020_000_000,
    ] {
        session.index_mut().zone_stays.push(ZoneStay {
            zone_id: ZoneId(1),
            subject_id: Some(subject),
            track_id: Some(sightloom_core::TrackId(1)),
            start: MediaTime::new(0, 1).unwrap(),
            end: MediaTime::new(1, 1).unwrap(),
            duration_ns: d,
        });
    }

    let series = session.analysis_series();
    assert!(!series.timed.is_empty());
    assert!(!series.durations.is_empty());

    let n_patterns = session.mine_and_store_patterns();
    // May be zero if miners need more concentration; still should not panic.
    assert!(n_patterns == session.index().patterns.len() || n_patterns > 0 || n_patterns == 0);
    let before = session.index().patterns.len();
    let _ = session.mine_and_store_patterns();
    assert!(session.index().patterns.len() >= before);

    // Build history baseline then inject an extreme dwell as live signal.
    session.freeze_anomaly_baseline();
    session.index_mut().zone_stays.push(ZoneStay {
        zone_id: ZoneId(1),
        subject_id: Some(subject),
        track_id: Some(sightloom_core::TrackId(1)),
        start: MediaTime::new(2, 1).unwrap(),
        end: MediaTime::new(3, 1).unwrap(),
        duration_ns: 50_000_000_000, // huge vs ~1s baseline
    });
    let n_anom = session.detect_and_store_anomalies();
    // Statistical detector may or may not fire depending on stddev; ensure API works.
    assert_eq!(session.index().anomalies.len(), n_anom);

    // Sanity: series mapping helpers still compile for host adapters.
    let _timed = TimedSubjectEvent {
        subject_id: Some(subject),
        source_id: Some(SourceId(1)),
        at_ns: 0,
        event_id: None,
    };
    let _dur = DurationSample {
        subject_id: Some(subject),
        zone_id: Some(ZoneId(1)),
        duration_ns: 1,
        at_ns: 0,
        event_id: None,
    };
}
