//! Contract tests for detections and caller-owned batches.

use sightloom_core::{ClassId, CoreError, Detection, DetectionBatch, Rect, TrackId, ZoneId};

fn bbox() -> Rect {
    Rect::new(0.0, 0.0, 2.0, 2.0).expect("test rectangle must be valid")
}

fn detection(score: f32) -> Detection {
    Detection::new(bbox(), score, Some(ClassId(3)), Some(TrackId(17)))
        .expect("test detection must be valid")
}

fn assert_approx_eq(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 1.0e-6,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn detection_rejects_non_finite_scores() {
    for score in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            Detection::new(bbox(), score, None, None),
            Err(CoreError::NonFinite)
        );
    }
}

#[test]
fn detection_preserves_typed_metadata() {
    let detection = detection(0.75);

    assert_eq!(detection.bbox(), bbox());
    assert_approx_eq(detection.score(), 0.75);
    assert_eq!(detection.class_id(), Some(ClassId(3)));
    assert_eq!(detection.track_id(), Some(TrackId(17)));

    let zone = ZoneId(5);
    assert_eq!(zone.0, 5);
}

#[test]
fn batch_never_silently_truncates_or_overwrites() {
    let first = detection(0.9);
    let second = detection(0.8);
    let mut storage = [Detection::default(); 1];
    let mut batch = DetectionBatch::new(&mut storage);

    batch.push(first).expect("first slot must be available");
    assert_eq!(batch.as_slice(), &[first]);

    assert_eq!(batch.push(second), Err(CoreError::InsufficientCapacity));
    assert_eq!(batch.as_slice(), &[first]);
}

#[test]
fn filled_batch_exposes_the_complete_caller_slice() {
    let mut storage = [detection(0.9), detection(0.8)];
    let expected = storage;
    let mut batch = DetectionBatch::from_filled(&mut storage);

    assert_eq!(batch.as_slice(), expected.as_slice());
    assert_eq!(
        batch.push(detection(0.7)),
        Err(CoreError::InsufficientCapacity)
    );
    assert_eq!(batch.as_slice(), expected.as_slice());
}
