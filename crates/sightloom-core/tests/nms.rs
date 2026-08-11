//! Contract tests for deterministic caller-owned non-maximum suppression.

use sightloom_core::{
    ClassId, CoreError, Detection, NmsConfig, NmsMode, OverlapMetric, Rect, TrackId, nms_in_place,
};

fn detection(bounds: [f32; 4], score: f32, class: u16) -> Detection {
    let bbox = Rect::new(bounds[0], bounds[1], bounds[2], bounds[3])
        .expect("test rectangle must be valid");
    Detection::new(bbox, score, Some(ClassId(class)), None).expect("test detection must be valid")
}

fn config(threshold: f32, mode: NmsMode, metric: OverlapMetric) -> NmsConfig {
    NmsConfig {
        threshold,
        mode,
        metric,
    }
}

#[test]
fn class_aware_preserves_identical_boxes_from_different_classes() {
    let first = detection([0.0, 0.0, 4.0, 4.0], 0.9, 1);
    let second = detection([0.0, 0.0, 4.0, 4.0], 0.8, 2);
    let mut detections = [first, second];
    let mut order = [0; 2];
    let mut suppressed = [false; 2];

    let kept = nms_in_place(
        &mut detections,
        &mut order,
        &mut suppressed,
        config(0.5, NmsMode::ClassAware, OverlapMetric::IoU),
    )
    .expect("NMS must succeed");

    assert_eq!(kept, 2);
    assert_eq!(&detections[..kept], &[first, second]);
}

#[test]
fn class_agnostic_suppresses_identical_boxes_from_different_classes() {
    let first = detection([0.0, 0.0, 4.0, 4.0], 0.9, 1);
    let second = detection([0.0, 0.0, 4.0, 4.0], 0.8, 2);
    let mut detections = [first, second];
    let mut order = [0; 2];
    let mut suppressed = [false; 2];

    let kept = nms_in_place(
        &mut detections,
        &mut order,
        &mut suppressed,
        config(0.5, NmsMode::ClassAgnostic, OverlapMetric::IoU),
    )
    .expect("NMS must succeed");

    assert_eq!(kept, 1);
    assert_eq!(&detections[..kept], &[first]);
}

#[test]
fn exact_threshold_overlap_survives() {
    let first = detection([0.0, 0.0, 3.0, 1.0], 0.9, 1);
    let second = detection([1.0, 0.0, 4.0, 1.0], 0.8, 1);
    let mut detections = [first, second];
    let mut order = [0; 2];
    let mut suppressed = [false; 2];

    let kept = nms_in_place(
        &mut detections,
        &mut order,
        &mut suppressed,
        config(0.5, NmsMode::ClassAware, OverlapMetric::IoU),
    )
    .expect("NMS must succeed");

    assert_eq!(kept, 2);
    assert_eq!(&detections[..kept], &[first, second]);
}

#[test]
fn equal_scores_keep_the_lower_original_index() {
    let first = detection([0.0, 0.0, 4.0, 4.0], 0.9, 1);
    let second = detection([0.0, 0.0, 4.0, 4.0], 0.9, 1);
    let mut detections = [first, second];
    let mut order = [0; 2];
    let mut suppressed = [false; 2];

    let kept = nms_in_place(
        &mut detections,
        &mut order,
        &mut suppressed,
        config(0.5, NmsMode::ClassAware, OverlapMetric::IoU),
    )
    .expect("NMS must succeed");

    assert_eq!(kept, 1);
    assert_eq!(&detections[..kept], &[first]);
}

#[test]
fn signed_zero_scores_keep_the_lower_original_index() {
    let bbox = Rect::new(0.0, 0.0, 4.0, 4.0).expect("test rectangle must be valid");
    let first = Detection::new(bbox, -0.0, Some(ClassId(1)), Some(TrackId(0)))
        .expect("test detection must be valid");
    let second = Detection::new(bbox, 0.0, Some(ClassId(1)), Some(TrackId(1)))
        .expect("test detection must be valid");
    let mut detections = [first, second];
    let mut order = [0; 2];
    let mut suppressed = [false; 2];

    let kept = nms_in_place(
        &mut detections,
        &mut order,
        &mut suppressed,
        config(0.5, NmsMode::ClassAware, OverlapMetric::IoU),
    )
    .expect("NMS must succeed");

    assert_eq!(kept, 1);
    assert_eq!(detections[0].track_id(), Some(TrackId(0)));
}

#[test]
fn class_aware_compares_absent_class_identifiers() {
    let bbox = Rect::new(0.0, 0.0, 4.0, 4.0).expect("test rectangle must be valid");
    let first = Detection::new(bbox, 0.9, None, None).expect("test detection must be valid");
    let second = Detection::new(bbox, 0.8, None, None).expect("test detection must be valid");
    let mut detections = [first, second];
    let mut order = [0; 2];
    let mut suppressed = [false; 2];

    let kept = nms_in_place(
        &mut detections,
        &mut order,
        &mut suppressed,
        config(0.5, NmsMode::ClassAware, OverlapMetric::IoU),
    )
    .expect("NMS must succeed");

    assert_eq!(kept, 1);
    assert_eq!(&detections[..kept], &[first]);
}

#[test]
fn retained_detections_are_compacted_in_original_input_order() {
    let lower_score = detection([0.0, 0.0, 1.0, 1.0], 0.1, 1);
    let higher_score = detection([3.0, 3.0, 4.0, 4.0], 0.9, 1);
    let mut detections = [lower_score, higher_score];
    let mut order = [0; 2];
    let mut suppressed = [false; 2];

    let kept = nms_in_place(
        &mut detections,
        &mut order,
        &mut suppressed,
        config(0.5, NmsMode::ClassAware, OverlapMetric::IoU),
    )
    .expect("NMS must succeed");

    assert_eq!(kept, 2);
    assert_eq!(&detections[..kept], &[lower_score, higher_score]);
}

#[test]
fn containment_changes_suppression_between_iou_and_ios() {
    let large = detection([0.0, 0.0, 4.0, 4.0], 0.9, 1);
    let contained = detection([1.0, 1.0, 3.0, 3.0], 0.8, 1);
    let mut union_detections = [large, contained];
    let mut union_order = [0; 2];
    let mut union_suppressed = [false; 2];
    let mut smaller_area_detections = [large, contained];
    let mut smaller_area_order = [0; 2];
    let mut smaller_area_suppressed = [false; 2];

    let union_kept = nms_in_place(
        &mut union_detections,
        &mut union_order,
        &mut union_suppressed,
        config(0.5, NmsMode::ClassAware, OverlapMetric::IoU),
    )
    .expect("IoU NMS must succeed");
    let smaller_area_kept = nms_in_place(
        &mut smaller_area_detections,
        &mut smaller_area_order,
        &mut smaller_area_suppressed,
        config(0.5, NmsMode::ClassAware, OverlapMetric::IoS),
    )
    .expect("IoS NMS must succeed");

    assert_eq!(union_kept, 2);
    assert_eq!(smaller_area_kept, 1);
    assert_eq!(&smaller_area_detections[..smaller_area_kept], &[large]);
}

#[test]
fn invalid_thresholds_leave_every_caller_slice_unchanged() {
    for threshold in [-0.1, 1.1, f32::NAN, f32::INFINITY] {
        let mut detections = [
            detection([0.0, 0.0, 4.0, 4.0], 0.9, 1),
            detection([0.0, 0.0, 4.0, 4.0], 0.8, 1),
        ];
        let original_detections = detections;
        let mut order = [19, 23];
        let original_order = order;
        let mut suppressed = [true, false];
        let original_suppressed = suppressed;

        assert_eq!(
            nms_in_place(
                &mut detections,
                &mut order,
                &mut suppressed,
                config(threshold, NmsMode::ClassAware, OverlapMetric::IoU),
            ),
            Err(CoreError::InvalidThreshold)
        );
        assert_eq!(detections, original_detections);
        assert_eq!(order, original_order);
        assert_eq!(suppressed, original_suppressed);
    }
}

#[test]
fn invalid_threshold_precedes_insufficient_scratch_without_mutation() {
    let mut detections = [detection([0.0, 0.0, 4.0, 4.0], 0.9, 1)];
    let original_detections = detections;
    let mut order = [];
    let mut suppressed = [];

    assert_eq!(
        nms_in_place(
            &mut detections,
            &mut order,
            &mut suppressed,
            config(f32::NAN, NmsMode::ClassAware, OverlapMetric::IoU),
        ),
        Err(CoreError::InvalidThreshold)
    );
    assert_eq!(detections, original_detections);
    assert!(order.is_empty());
    assert!(suppressed.is_empty());
}

#[test]
fn insufficient_scratch_leaves_every_caller_slice_unchanged() {
    for (order_len, suppressed_len) in [(1, 2), (2, 1)] {
        let mut detections = [
            detection([0.0, 0.0, 4.0, 4.0], 0.9, 1),
            detection([0.0, 0.0, 4.0, 4.0], 0.8, 1),
        ];
        let original_detections = detections;
        let mut order = [19, 23];
        let original_order = order;
        let mut suppressed = [true, false];
        let original_suppressed = suppressed;

        let result = nms_in_place(
            &mut detections,
            &mut order[..order_len],
            &mut suppressed[..suppressed_len],
            config(0.5, NmsMode::ClassAware, OverlapMetric::IoU),
        );

        assert_eq!(result, Err(CoreError::InsufficientScratch));
        assert_eq!(detections, original_detections);
        assert_eq!(order, original_order);
        assert_eq!(suppressed, original_suppressed);
    }
}

#[test]
fn empty_input_needs_no_scratch() {
    let mut detections = [];
    let mut order = [];
    let mut suppressed = [];

    assert_eq!(
        nms_in_place(
            &mut detections,
            &mut order,
            &mut suppressed,
            config(0.5, NmsMode::ClassAware, OverlapMetric::IoU),
        ),
        Ok(0)
    );
}
