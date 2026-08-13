//! Evidence reels and extended subject query tests.

#![allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]

use sightloom_core::{MediaTime, SourceId, SubjectId, TrackId, ZoneId};
use sightloom_index::{
    EvidenceReelBuilder, Route, SubjectQuery, TrackSample, VisionIndex, ZoneStay,
    execute_subject_query,
};

fn sample(subject: u64, track: u32, frame: u64, left: f32) -> TrackSample {
    TrackSample {
        sample_id: 0,
        supersedes: None,
        revision: 0,
        idempotency_key: 0,
        source_id: SourceId(1),
        frame_index: frame,
        pts: MediaTime::new(frame as i64, 30).unwrap(),
        track_id: TrackId(track),
        track_uid: None,
        subject_id: Some(SubjectId(subject)),
        class_id: None,
        left,
        top: 0.0,
        right: left + 10.0,
        bottom: 20.0,
        confidence: 0.9,
        mask_ref: 0,
    }
}

#[test]
fn coalesced_reel_merges_close_samples() {
    let mut index = VisionIndex::new("reel");
    for f in 0..5 {
        index.push_track(sample(1, 7, f, f as f32));
    }
    let mut builder = EvidenceReelBuilder::new();
    // 1/30s between frames; allow 100ms gap → coalesce all
    let reel = builder.from_subject_coalesced(&index, SubjectId(1), 100_000_000, 0);
    assert_eq!(reel.segments.len(), 1);
    assert_eq!(reel.segments[0].track_id, Some(TrackId(7)));
    assert!(reel.span_ns().unwrap() > 0);

    let per_sample = builder.from_subject_samples(&index, SubjectId(1), 0);
    assert_eq!(per_sample.segments.len(), 5);
}

#[test]
fn route_and_then_seen_query() {
    let mut index = VisionIndex::new("q");
    index.push_track(sample(1, 1, 0, 0.0));
    index.zone_stays.push(ZoneStay {
        zone_id: ZoneId(1),
        subject_id: Some(SubjectId(1)),
        track_id: Some(TrackId(1)),
        start: MediaTime::new(0, 30).unwrap(),
        end: MediaTime::new(1, 30).unwrap(),
        duration_ns: 33_333_333,
    });
    index.zone_stays.push(ZoneStay {
        zone_id: ZoneId(2),
        subject_id: Some(SubjectId(1)),
        track_id: Some(TrackId(1)),
        start: MediaTime::new(2, 30).unwrap(),
        end: MediaTime::new(3, 30).unwrap(),
        duration_ns: 33_333_333,
    });
    index.routes.push(Route {
        subject_id: SubjectId(1),
        zones: vec![ZoneId(1), ZoneId(2), ZoneId(3)],
        sources: vec![SourceId(1)],
        start: MediaTime::new(0, 30).unwrap(),
        end: MediaTime::new(3, 30).unwrap(),
    });

    let hits = execute_subject_query(
        &index,
        &SubjectQuery::new()
            .then_seen_in(ZoneId(1), ZoneId(2), 0)
            .route_contains(vec![ZoneId(1), ZoneId(2)]),
    );
    assert_eq!(hits.len(), 1);
    assert!(!hits[0].routes.is_empty());
}
