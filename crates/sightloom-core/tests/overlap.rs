//! Contract tests for rectangle overlap metrics.

use sightloom_core::{Rect, intersection_area, ios, iou};

fn rect(left: f32, top: f32, right: f32, bottom: f32) -> Rect {
    Rect::new(left, top, right, bottom).expect("test rectangle must be valid")
}

fn assert_approx_eq(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 1.0e-6,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn overlap_metrics_match_known_values() {
    let a = rect(0.0, 0.0, 4.0, 4.0);
    let b = rect(2.0, 2.0, 6.0, 6.0);

    assert_approx_eq(intersection_area(a, b), 4.0);
    assert_approx_eq(iou(a, b), 1.0 / 7.0);
    assert_approx_eq(ios(a, b), 0.25);
}

#[test]
fn disjoint_boxes_produce_zero_overlap() {
    let a = rect(0.0, 0.0, 1.0, 1.0);
    let b = rect(2.0, 2.0, 3.0, 3.0);

    assert_approx_eq(intersection_area(a, b), 0.0);
    assert_approx_eq(iou(a, b), 0.0);
    assert_approx_eq(ios(a, b), 0.0);
}

#[test]
fn zero_area_boxes_produce_zero_overlap() {
    let zero = rect(1.0, 1.0, 1.0, 4.0);
    let full = rect(0.0, 0.0, 4.0, 4.0);

    assert_approx_eq(iou(zero, full), 0.0);
    assert_approx_eq(ios(zero, full), 0.0);
}
