//! `VisionIndex` M0 contract tests.

use sightloom_core::{
    AppearanceId, EventEnvelope, EventId, EventKind, EventPayload, FrameStamp, MediaTime, SourceId,
    SubjectId, TrackId, VisitId, ZoneId,
};
use sightloom_memory::{
    Appearance, ModelProvenance, Severity, SourceEntry, SubjectProfile, TrackSample,
    VISION_INDEX_VERSION, VisionIndex, VisionIndexHeader, Visit,
};

#[test]
fn header_json_roundtrip() {
    let mut header = VisionIndexHeader::new("lobby");
    header.sources.push(SourceEntry {
        source_id: 1,
        uri: "file://lobby.mp4".into(),
        hash: None,
    });
    header.provenance = Some(ModelProvenance::new("detector", "1", 0.25, Some(0.5)));
    let json = header.to_json().unwrap();
    let parsed = VisionIndexHeader::from_json(&json).unwrap();
    assert_eq!(parsed.version, VISION_INDEX_VERSION);
    assert_eq!(parsed.name, "lobby");
    assert_eq!(parsed.sources.len(), 1);
}

#[test]
fn vision_index_holds_core_entities_and_events() {
    let mut index = VisionIndex::new("cam-a");
    index.add_source(SourceEntry {
        source_id: 1,
        uri: "rtsp://cam-a".into(),
        hash: None,
    });

    index.push_track(TrackSample {
        source_id: SourceId(1),
        frame_index: 0,
        pts: MediaTime::new(0, 30).unwrap(),
        track_id: TrackId(1),
        subject_id: Some(SubjectId(17)),
        class_id: None,
        left: 0.0,
        top: 0.0,
        right: 10.0,
        bottom: 20.0,
        confidence: 0.9,
        mask_ref: 0,
    });

    let stamp = FrameStamp::new(SourceId(1), 0, MediaTime::new(0, 30).unwrap(), None);
    index.push_event(
        EventEnvelope::new(EventId(1), stamp, EventKind::Zone)
            .with_track(TrackId(1))
            .with_subject(SubjectId(17))
            .with_zone(ZoneId(2))
            .with_payload(EventPayload::Entered {
                zone_id: ZoneId(2),
                class_id: None,
            }),
    );

    index.appearances.push(Appearance {
        appearance_id: AppearanceId(1),
        subject_id: Some(SubjectId(17)),
        track_id: Some(TrackId(1)),
        source_id: SourceId(1),
        start: MediaTime::new(0, 30).unwrap(),
        end: MediaTime::new(30, 30).unwrap(),
        class_id: None,
        peak_confidence: 0.9,
        evidence: None,
    });
    index.visits.push(Visit {
        visit_id: VisitId(1),
        subject_id: Some(SubjectId(17)),
        start: MediaTime::new(0, 30).unwrap(),
        end: MediaTime::new(30, 30).unwrap(),
        source_count: 1,
        duration_ns: 1_000_000_000,
    });
    index.subjects.push(SubjectProfile {
        subject_id: SubjectId(17),
        label: Some("person-17".into()),
        appearance_count: 1,
        source_count: 1,
        total_duration_ns: 1_000_000_000,
        first_seen: Some(MediaTime::new(0, 30).unwrap()),
        last_seen: Some(MediaTime::new(30, 30).unwrap()),
        embedding: None,
    });

    assert_eq!(index.events.len(), 1);
    assert_eq!(index.tracks.for_subject(SubjectId(17)).len(), 1);
    assert_eq!(index.subjects[0].subject_id, SubjectId(17));
    assert!(Severity::Low < Severity::Critical);
    index.validate().unwrap();
}
