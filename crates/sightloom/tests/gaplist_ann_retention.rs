//! ANN track search, retention policy, Prometheus metrics text.

use sightloom::{IndexSession, RetentionPolicy};
use sightloom_core::{FrameStamp, MediaTime, Rect, SourceId};
use sightloom_index::{QueryNode, SourceEntry, SubjectPredicate};
use sightloom_reid::AnnKind;
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
fn track_ann_search_and_retention_and_prom() {
    let mut session = IndexSession::new("ann", track_config()).unwrap();
    session.add_source(SourceEntry {
        source_id: 1,
        uri: "file://a.mp4".into(),
        hash: None,
    });
    session.set_track_ann_kind(Some(AnnKind::Lsh {
        bits: 12,
        multiprobe: 2,
    }));

    let stamp = FrameStamp::new(SourceId(1), 0, MediaTime::new(0, 30).unwrap(), None);
    let seed = session
        .seed_click(stamp, Rect::new(0.0, 0.0, 10.0, 20.0).unwrap(), 0.9, None)
        .unwrap();
    session
        .note_track_embedding(seed.track_key(), [1.0_f32, 0.0, 0.0], stamp.pts)
        .unwrap();

    let stamp1 = FrameStamp::new(SourceId(1), 1, MediaTime::new(1, 30).unwrap(), None);
    let tracked = session
        .ingest_detections(
            stamp1,
            &[sightloom_core::Detection::new(
                Rect::new(80.0, 0.0, 100.0, 20.0).unwrap(),
                0.9,
                None,
                None,
            )
            .unwrap()],
        )
        .unwrap();
    let other = tracked[0].track_key;
    session
        .note_track_embedding(other, [0.0_f32, 1.0, 0.0], stamp1.pts)
        .unwrap();

    let hits = session
        .search_tracks_by_embedding([0.99_f32, 0.01, 0.0], 2)
        .unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].track_key, seed.track_key());

    // Retention: keep only 1 track sample.
    session.set_retention_policy(RetentionPolicy {
        max_track_samples: 1,
        ..RetentionPolicy::default()
    });
    let report = session.apply_retention();
    assert!(report.dropped_tracks >= 1);
    assert_eq!(session.index().tracks.samples().len(), 1);

    let prom = session.prometheus_metrics();
    assert!(prom.contains("sightloom_ingest_accepted"));
    assert!(prom.contains("session=\"ann\""));

    // Query AST still works on remaining labeled samples if any subject remains.
    let _ = session.query_ast(&QueryNode::pred(SubjectPredicate::SeenOn(SourceId(1))));
}

trait TrackKeyExt {
    fn track_key(self) -> sightloom_core::TrackKey;
}

impl TrackKeyExt for sightloom::SeedResult {
    fn track_key(self) -> sightloom_core::TrackKey {
        sightloom_core::TrackKey::new(self.source_id, self.track_id)
    }
}
