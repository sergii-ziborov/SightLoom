//! Video memory package tests.

use sightloom_core::{MediaTime, SourceId, SubjectId, TrackId};
use sightloom_memory::{
    MemoryManifest, ModelProvenance, SourceEntry, SourceHash, TrackSample, VideoMemory,
};

#[test]
fn manifest_json_roundtrip() {
    let mut manifest = MemoryManifest::new("cam-lobby");
    manifest.sources.push(SourceEntry {
        source_id: 1,
        uri: "file://lobby.mp4".into(),
        hash: Some(SourceHash {
            algorithm: "sha256".into(),
            digest_hex: "abc".into(),
        }),
    });
    manifest.provenance = Some(ModelProvenance::new("yolo", "v8", 0.25, Some(0.45)));
    let json = manifest.to_json().unwrap();
    let parsed = MemoryManifest::from_json(&json).unwrap();
    assert_eq!(parsed.name, "cam-lobby");
    assert_eq!(parsed.sources.len(), 1);
}

#[test]
fn video_memory_tracks_masks_events() {
    let mut memory = VideoMemory::new("demo");
    memory.add_source(SourceEntry {
        source_id: 1,
        uri: "rtsp://cam".into(),
        hash: None,
    });
    let mask = memory.masks.insert([1_u8, 0, 1, 1]);
    memory.tracks.push(TrackSample {
        source_id: SourceId(1),
        frame_index: 3,
        pts: MediaTime::new(3, 30).unwrap(),
        track_id: TrackId(7),
        subject_id: Some(SubjectId(17)),
        class_id: None,
        left: 1.0,
        top: 2.0,
        right: 3.0,
        bottom: 4.0,
        confidence: 0.9,
        mask_ref: mask.0,
    });
    memory.events.insert(
        "dwell_ended",
        Some(TrackId(7)),
        Some(SubjectId(17)),
        None,
        1_000,
        Some(500),
    );

    assert_eq!(memory.tracks.for_subject(SubjectId(17)).len(), 1);
    assert_eq!(memory.events.by_track(TrackId(7)).len(), 1);
    assert_eq!(memory.masks.get(mask).unwrap(), &[1, 0, 1, 1]);
    memory.validate().unwrap();
}
