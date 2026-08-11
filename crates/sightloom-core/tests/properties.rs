//! Property tests for public geometry and caller-owned NMS contracts.

use proptest::prelude::*;
use sightloom_core::{
    ClassId, Detection, NmsConfig, NmsMode, OverlapMetric, Rect, ios, iou, nms_in_place,
};

fn rect_strategy() -> impl Strategy<Value = Rect> {
    (-128_i16..=128, -128_i16..=128, 0_i16..=64, 0_i16..=64).prop_map(
        |(left, top, width, height)| {
            Rect::new(
                f32::from(left),
                f32::from(top),
                f32::from(left + width),
                f32::from(top + height),
            )
            .expect("bounded ordered coordinates are valid")
        },
    )
}

fn positive_rect_strategy() -> impl Strategy<Value = Rect> {
    (-128_i16..=128, -128_i16..=128, 1_i16..=64, 1_i16..=64).prop_map(
        |(left, top, width, height)| {
            Rect::new(
                f32::from(left),
                f32::from(top),
                f32::from(left + width),
                f32::from(top + height),
            )
            .expect("bounded positive coordinates are valid")
        },
    )
}

fn detection_strategy() -> impl Strategy<Value = Detection> {
    (rect_strategy(), -1_000_i16..=1_000, 0_u16..=3).prop_map(|(bbox, score, class)| {
        Detection::new(bbox, f32::from(score) / 1_000.0, Some(ClassId(class)), None)
            .expect("bounded score is finite")
    })
}

fn degenerate_detection_strategy() -> impl Strategy<Value = Detection> {
    (
        -128_i16..=128,
        -128_i16..=128,
        0_i16..=64,
        any::<bool>(),
        0_u16..=3,
    )
        .prop_map(|(x, y, span, vertical, class)| {
            let (right, bottom) = if vertical {
                (x, y + span)
            } else {
                (x + span, y)
            };
            let bbox = Rect::new(
                f32::from(x),
                f32::from(y),
                f32::from(right),
                f32::from(bottom),
            )
            .expect("bounded degenerate coordinates are valid");
            Detection::new(bbox, f32::from(class), Some(ClassId(class)), None)
                .expect("generated score is finite")
        })
}

fn config(metric: OverlapMetric) -> NmsConfig {
    NmsConfig {
        threshold: 0.0,
        mode: NmsMode::ClassAgnostic,
        metric,
    }
}

proptest! {
    #[test]
    fn iou_is_symmetric_and_bounded(a in rect_strategy(), b in rect_strategy()) {
        let forward = iou(a, b);
        let reverse = iou(b, a);

        prop_assert!((forward - reverse).abs() <= 1.0e-6);
        prop_assert!((0.0..=1.0).contains(&forward));
    }

    #[test]
    fn positive_area_rectangle_has_unit_self_iou(rect in positive_rect_strategy()) {
        prop_assert!((iou(rect, rect) - 1.0).abs() <= 1.0e-6);
    }

    #[test]
    fn nms_is_idempotent(input in prop::collection::vec(detection_strategy(), 0..32)) {
        let mut first = input;
        let mut first_order = vec![0; first.len()];
        let mut first_suppressed = vec![false; first.len()];
        let first_kept = nms_in_place(
            &mut first,
            &mut first_order,
            &mut first_suppressed,
            config(OverlapMetric::IoU),
        ).expect("valid threshold and matching scratch must succeed");
        let expected = first[..first_kept].to_vec();

        let mut second = expected.clone();
        let mut second_order = vec![0; second.len()];
        let mut second_suppressed = vec![false; second.len()];
        let second_kept = nms_in_place(
            &mut second,
            &mut second_order,
            &mut second_suppressed,
            config(OverlapMetric::IoU),
        ).expect("a retained valid batch must remain suppressible");

        prop_assert_eq!(second_kept, expected.len());
        prop_assert_eq!(&second[..second_kept], expected.as_slice());
    }

    #[test]
    fn degenerate_geometry_has_zero_overlap_and_stable_nms(
        input in prop::collection::vec(degenerate_detection_strategy(), 0..32),
    ) {
        for detection in &input {
            prop_assert!(iou(detection.bbox(), detection.bbox()).abs() <= f32::EPSILON);
            prop_assert!(ios(detection.bbox(), detection.bbox()).abs() <= f32::EPSILON);
        }

        for metric in [OverlapMetric::IoU, OverlapMetric::IoS] {
            let mut detections = input.clone();
            let mut order = vec![0; detections.len()];
            let mut suppressed = vec![false; detections.len()];
            let kept = nms_in_place(&mut detections, &mut order, &mut suppressed, config(metric))
                .expect("valid degenerate geometry must not make NMS fail");

            prop_assert_eq!(kept, input.len());
            prop_assert_eq!(&detections[..kept], input.as_slice());
        }
    }
}
