//! Observation revisions, idempotency, spatial query, detector adapter.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use sightloom::{DetectorAdapter, FrameView, IndexSession, PixelFormat, SessionError};
use sightloom_core::{
    Detection, EvidenceRef, FrameStamp, MediaTime, ObservationId, Rect, SourceId,
};
use sightloom_index::{Observation, SourceEntry, SpatialQuery};
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

struct FakeDetector {
    box_x: f32,
}

impl DetectorAdapter for FakeDetector {
    type Error = &'static str;

    fn detect(
        &mut self,
        _stamp: FrameStamp,
        _frame: &FrameView<'_>,
    ) -> Result<Vec<Detection>, Self::Error> {
        Ok(vec![
            Detection::new(
                Rect::new(self.box_x, 0.0, self.box_x + 10.0, 20.0).unwrap(),
                0.9,
                None,
                None,
            )
            .unwrap(),
        ])
    }
}

#[test]
fn observation_revision_and_idempotent_ingest() {
    let mut session = IndexSession::new("gap", track_config()).unwrap();
    session.add_source(SourceEntry {
        source_id: 1,
        uri: "file://a.mp4".into(),
        hash: None,
    });

    let stamp = FrameStamp::new(SourceId(1), 0, MediaTime::new(0, 30).unwrap(), None);
    let bbox = Rect::new(0.0, 0.0, 10.0, 20.0).unwrap();
    let obs = Observation::new(ObservationId(0), stamp, bbox, 0.8, EvidenceRef(1))
        .unwrap()
        .with_idempotency_key(42);
    session.push_observation(obs);
    assert_eq!(session.effective_observations().len(), 1);
    let prior = session.effective_observations()[0].id.0;

    let revised = Observation::new(
        ObservationId(0),
        stamp,
        Rect::new(1.0, 1.0, 11.0, 21.0).unwrap(),
        0.95,
        EvidenceRef(2),
    )
    .unwrap();
    assert!(session.revise_observation(prior, revised));
    let effective = session.effective_observations();
    assert_eq!(effective.len(), 1);
    assert!((effective[0].confidence - 0.95).abs() < 1e-5);
    assert!(effective[0].supersedes.is_some());
    assert!(effective[0].revision >= 2);

    let det = Detection::new(bbox, 0.9, None, None).unwrap();
    session.ingest_detections_keyed(stamp, &[det], 100).unwrap();
    let err = session
        .ingest_detections_keyed(
            FrameStamp::new(SourceId(1), 1, MediaTime::new(1, 30).unwrap(), None),
            &[Detection::new(bbox, 0.9, None, None).unwrap()],
            100,
        )
        .unwrap_err();
    assert_eq!(err, SessionError::DuplicateIdempotencyKey);
}

#[test]
fn spatial_query_and_detect_and_ingest() {
    let mut session = IndexSession::new("spatial", track_config()).unwrap();
    session.add_source(SourceEntry {
        source_id: 1,
        uri: "file://a.mp4".into(),
        hash: None,
    });

    let mut det = FakeDetector { box_x: 5.0 };
    let pixels = [0_u8; 16];
    let frame = FrameView::new(4, 4, 4, PixelFormat::Gray8, &pixels);
    let stamp = FrameStamp::new(SourceId(1), 0, MediaTime::new(0, 30).unwrap(), None);
    let tracked = session.detect_and_ingest(stamp, &frame, &mut det).unwrap();
    assert_eq!(tracked.len(), 1);

    let hits = session.query_spatial(
        &SpatialQuery::new(0.0, 0.0, 20.0, 30.0)
            .on_source(SourceId(1))
            .with_min_confidence(0.5),
    );
    assert!(!hits.is_empty());
    assert!(hits[0].iou > 0.0);

    let miss = session.query_spatial(&SpatialQuery::new(500.0, 500.0, 600.0, 600.0));
    assert!(miss.is_empty());
}
