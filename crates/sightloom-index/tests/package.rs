//! On-disk `VisionIndex` package round-trip.

use sightloom_core::{
    AppearanceId, EventEnvelope, EventId, EventKind, EventPayload, FrameStamp, MediaTime, SourceId,
    SubjectId, TrackId, VisitId, ZoneId,
};
use sightloom_index::{
    Appearance, SourceEntry, TrackSample, VisionIndex, VisionIndexPackage, Visit,
};
use tempfile::tempdir;

#[test]
fn package_save_load_roundtrip() {
    let mut index = VisionIndex::new("cam-a");
    index.add_source(SourceEntry {
        source_id: 1,
        uri: "file://cam-a.mp4".into(),
        hash: None,
    });
    index.push_track(TrackSample {
        sample_id: 0,
        supersedes: None,
        revision: 0,
        idempotency_key: 0,
        source_id: SourceId(1),
        frame_index: 3,
        pts: MediaTime::new(3, 30).unwrap(),
        track_id: TrackId(7),
        track_uid: None,
        subject_id: Some(SubjectId(17)),
        class_id: None,
        left: 1.0,
        top: 2.0,
        right: 11.0,
        bottom: 22.0,
        confidence: 0.91,
        mask_ref: 0,
    });
    let mask = index.masks.insert([1_u8, 0, 1, 1]);
    // attach mask on a correction sample
    index.push_track(TrackSample {
        sample_id: 0,
        supersedes: None,
        revision: 0,
        idempotency_key: 0,
        source_id: SourceId(1),
        frame_index: 4,
        pts: MediaTime::new(4, 30).unwrap(),
        track_id: TrackId(7),
        track_uid: None,
        subject_id: Some(SubjectId(17)),
        class_id: None,
        left: 2.0,
        top: 2.0,
        right: 12.0,
        bottom: 22.0,
        confidence: 0.92,
        mask_ref: mask.0,
    });
    let stamp = FrameStamp::new(SourceId(1), 4, MediaTime::new(4, 30).unwrap(), None);
    index.push_event(
        EventEnvelope::new(EventId(9), stamp, EventKind::Zone)
            .with_track(TrackId(7))
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
        track_id: Some(TrackId(7)),
        source_id: SourceId(1),
        start: MediaTime::new(0, 30).unwrap(),
        end: MediaTime::new(4, 30).unwrap(),
        class_id: None,
        peak_confidence: 0.92,
        evidence: None,
    });
    index.visits.push(Visit {
        visit_id: VisitId(1),
        subject_id: Some(SubjectId(17)),
        start: MediaTime::new(0, 30).unwrap(),
        end: MediaTime::new(4, 30).unwrap(),
        source_count: 1,
        duration_ns: 133_333_333,
    });

    let dir = tempdir().unwrap();
    VisionIndexPackage::save(&index, dir.path()).unwrap();

    // Transactional generation layout.
    assert!(dir.path().join("CURRENT").exists());
    let generation =
        sightloom_index::VisionIndexPackage::current_generation(dir.path()).unwrap();
    assert!(generation.starts_with("gen-"));
    let gen_dir = dir.path().join(&generation);
    assert!(gen_dir.join("manifest.json").exists());
    assert!(gen_dir.join("tracks.cbor").exists());
    assert!(gen_dir.join("masks.bin").exists());
    assert!(gen_dir.join("events.cbor").exists());
    assert!(gen_dir.join("entities.json").exists());
    assert!(gen_dir.join("checksums.json").exists());

    let loaded = VisionIndexPackage::load(dir.path()).unwrap();
    assert_eq!(loaded.header.name, "cam-a");
    assert_eq!(loaded.tracks.samples().len(), 2);
    assert_eq!(loaded.tracks.samples()[1].mask_ref, mask.0);
    assert_eq!(loaded.masks.get(mask).unwrap(), &[1, 0, 1, 1]);
    assert_eq!(loaded.events.len(), 1);
    assert_eq!(loaded.events[0].subject_id, Some(SubjectId(17)));
    assert_eq!(loaded.appearances.len(), 1);
    assert_eq!(loaded.visits.len(), 1);
}

#[test]
fn validate_full_detects_unknown_source_and_mask() {
    use sightloom_index::{ValidationSeverity, VisionIndex};

    let mut index = VisionIndex::new("broken");
    index.push_track(TrackSample {
        sample_id: 0,
        supersedes: None,
        revision: 0,
        idempotency_key: 0,
        source_id: SourceId(99),
        frame_index: 0,
        pts: MediaTime::new(0, 1).unwrap(),
        track_id: TrackId(1),
        track_uid: None,
        subject_id: None,
        class_id: None,
        left: 0.0,
        top: 0.0,
        right: 10.0,
        bottom: 10.0,
        confidence: 0.5,
        mask_ref: 42,
    });
    // No sources registered and no masks → full validation must error.
    let report = index.validate_full();
    assert!(report.has_errors());
    assert!(
        report
            .issues
            .iter()
            .any(|i| i.severity == ValidationSeverity::Error && i.path.contains("source_id"))
    );
    assert!(
        report
            .issues
            .iter()
            .any(|i| i.path.contains("mask_ref"))
    );
    let plan = index.repair_plan();
    assert!(!plan.is_empty());
}

#[cfg(feature = "sqlite")]
#[test]
fn sqlite_event_index_queryable() {
    let mut index = VisionIndex::new("q");
    let stamp = FrameStamp::new(SourceId(1), 0, MediaTime::new(0, 1).unwrap(), None);
    index.push_event(
        EventEnvelope::new(EventId(1), stamp, EventKind::Zone)
            .with_subject(SubjectId(42))
            .with_payload(EventPayload::Empty),
    );
    let dir = tempdir().unwrap();
    VisionIndexPackage::save(&index, dir.path()).unwrap();
    let db = dir.path().join("events.sqlite");
    assert!(db.exists());
    let count = sightloom_index::sqlite_query::count_events_for_subject(&db, 42).unwrap();
    assert_eq!(count, 1);
}
