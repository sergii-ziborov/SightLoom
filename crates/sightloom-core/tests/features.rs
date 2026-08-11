//! Public contracts for allocation-backed conveniences.
#![cfg(feature = "alloc")]

use sightloom_core::{
    ClassId, CoreError, Detection, NmsConfig, NmsMode, OverlapMetric, OwnedDetectionBatch, Rect,
    nms_in_place,
};

#[test]
fn owned_batch_nms_matches_slice_first_nms() {
    let first = Detection::new(
        Rect::new(0.0, 0.0, 4.0, 4.0).expect("literal rectangle must be valid"),
        0.9,
        Some(ClassId(1)),
        None,
    )
    .expect("literal detection must be valid");
    let second = Detection::new(
        Rect::new(0.0, 0.0, 4.0, 4.0).expect("literal rectangle must be valid"),
        0.8,
        Some(ClassId(2)),
        None,
    )
    .expect("literal detection must be valid");
    let third = Detection::new(
        Rect::new(8.0, 0.0, 10.0, 2.0).expect("literal rectangle must be valid"),
        0.7,
        Some(ClassId(1)),
        None,
    )
    .expect("literal detection must be valid");
    let mut caller_owned = [first, second, third];
    let mut owned = OwnedDetectionBatch::new();
    for detection in caller_owned {
        owned.push(detection);
    }
    let config = NmsConfig {
        threshold: 0.5,
        mode: NmsMode::ClassAgnostic,
        metric: OverlapMetric::IoU,
    };
    let mut order = [0; 3];
    let mut suppressed = [false; 3];

    let expected = nms_in_place(&mut caller_owned, &mut order, &mut suppressed, config)
        .expect("slice-first NMS must succeed");
    let actual = owned.nms(config).expect("owned NMS must succeed");

    assert_eq!(actual, expected);
    assert_eq!(owned.as_slice(), &caller_owned[..expected]);
}

#[test]
fn owned_batch_nms_preserves_detections_on_invalid_threshold() {
    let first = Detection::new(
        Rect::new(0.0, 0.0, 4.0, 4.0).expect("literal rectangle must be valid"),
        0.9,
        Some(ClassId(1)),
        None,
    )
    .expect("literal detection must be valid");
    let second = Detection::new(
        Rect::new(0.0, 0.0, 4.0, 4.0).expect("literal rectangle must be valid"),
        0.8,
        Some(ClassId(1)),
        None,
    )
    .expect("literal detection must be valid");
    let mut owned = OwnedDetectionBatch::new();
    owned.push(first);
    owned.push(second);
    let before = owned.as_slice().to_vec();

    let result = owned.nms(NmsConfig {
        threshold: 1.1,
        mode: NmsMode::ClassAware,
        metric: OverlapMetric::IoU,
    });

    assert_eq!(result, Err(CoreError::InvalidThreshold));
    assert_eq!(owned.as_slice(), before.as_slice());
}
