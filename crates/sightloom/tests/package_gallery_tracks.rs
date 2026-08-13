//! Package gallery sidecar + unlabeled track embedding search.

use sightloom::IndexSession;
use sightloom_core::{FrameStamp, MediaTime, Rect, SourceId};
use sightloom_index::SourceEntry;
use sightloom_tracking::ByteTrackConfig;
use tempfile::tempdir;

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
fn package_roundtrip_restores_gallery_and_track_index() {
    let dir = tempdir().unwrap();
    let mut session = IndexSession::new("pkg", track_config()).unwrap();
    session.add_source(SourceEntry {
        source_id: 1,
        uri: "file://v.mp4".into(),
        hash: None,
    });

    let subject = session
        .enroll_subject_photos(
            sightloom_reid::SubjectModality::PersonAppearance,
            &[vec![1.0_f32, 0.0]],
        )
        .unwrap();

    let stamp = FrameStamp::new(SourceId(1), 0, MediaTime::new(0, 30).unwrap(), None);
    let seed = session
        .seed_click(
            stamp,
            Rect::new(0.0, 0.0, 10.0, 20.0).unwrap(),
            0.9,
            Some(subject),
        )
        .unwrap();
    session
        .note_track_embedding(seed.track_key(), [0.99_f32, 0.01], stamp.pts)
        .unwrap();

    // Unlabeled second track with embedding only.
    let stamp1 = FrameStamp::new(SourceId(1), 1, MediaTime::new(1, 30).unwrap(), None);
    let tracked = session
        .ingest_detections(
            stamp1,
            &[sightloom_core::Detection::new(
                Rect::new(100.0, 0.0, 120.0, 40.0).unwrap(),
                0.9,
                None,
                None,
            )
            .unwrap()],
        )
        .unwrap();
    let other = tracked[0].track_key;
    session
        .note_track_embedding(other, [0.0_f32, 1.0], stamp1.pts)
        .unwrap();

    session.save_package(dir.path()).unwrap();
    assert!(
        sightloom_index::VisionIndexPackage::active_payload_dir(dir.path())
            .join(sightloom_index::GALLERY_FILE)
            .exists()
    );

    let mut loaded = IndexSession::load_package(dir.path(), track_config()).unwrap();
    assert_eq!(loaded.gallery().subjects().len(), 1);
    assert!(!loaded.gallery().embeddings.entries().is_empty());

    let hits = loaded
        .search_tracks_by_embedding([0.0_f32, 1.0], 5)
        .unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].track_key, other);
    assert!(hits[0].score > 0.9);

    let gallery_hits = loaded
        .search_by_photo(
            [1.0_f32, 0.0],
            sightloom_reid::SubjectModality::PersonAppearance,
            3,
        )
        .unwrap();
    assert_eq!(gallery_hits[0].subject_id, subject);
}

trait TrackKeyExt {
    fn track_key(self) -> sightloom_core::TrackKey;
}

impl TrackKeyExt for sightloom::SeedResult {
    fn track_key(self) -> sightloom_core::TrackKey {
        sightloom_core::TrackKey::new(self.source_id, self.track_id)
    }
}
