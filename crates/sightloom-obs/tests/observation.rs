//! Observation construction tests.

use sightloom_core::{
    ClassId, Detection, EvidenceRef, FrameStamp, MaskRef, MediaTime, ObservationId, Rect, SourceId,
    SubjectId, TrackId,
};
use sightloom_obs::Observation;

fn stamp() -> FrameStamp {
    FrameStamp::new(SourceId(1), 10, MediaTime::new(10, 30).unwrap(), None)
}

#[test]
fn from_detection_preserves_fields() {
    let detection = Detection::new(
        Rect::new(1.0, 2.0, 3.0, 4.0).unwrap(),
        0.77,
        Some(ClassId(3)),
        Some(TrackId(9)),
    )
    .unwrap();
    let obs = Observation::from_detection(ObservationId(100), stamp(), detection, EvidenceRef(5))
        .unwrap()
        .with_subject_id(SubjectId(17))
        .with_mask(MaskRef(42));

    assert_eq!(obs.id, ObservationId(100));
    assert!((obs.confidence - 0.77).abs() < f32::EPSILON);
    assert_eq!(obs.class_id, Some(ClassId(3)));
    assert_eq!(obs.track_id, Some(TrackId(9)));
    assert_eq!(obs.subject_id, Some(SubjectId(17)));
    assert_eq!(obs.mask, Some(MaskRef(42)));
    assert_eq!(obs.provenance, EvidenceRef(5));

    let back = obs.to_detection().unwrap();
    assert!((back.score() - 0.77).abs() < f32::EPSILON);
    assert_eq!(back.track_id(), Some(TrackId(9)));
}
