//! Public geometry contract tests.
#![allow(clippy::float_cmp)]

use sightloom_core::{GeometryError, Point, Rect};

#[test]
fn point_rejects_non_finite_coordinates() {
    assert_eq!(Point::new(f32::NAN, 1.0), Err(GeometryError::NonFinite));
}

#[test]
fn rectangle_rejects_inverted_bounds_but_accepts_zero_area() {
    assert_eq!(
        Rect::new(2.0, 0.0, 1.0, 1.0),
        Err(GeometryError::InvertedBounds)
    );

    let zero = Rect::new(1.0, 2.0, 1.0, 5.0).unwrap();
    assert_eq!(zero.area(), 0.0);
}

#[test]
fn rectangle_reports_geometry_without_clamping() {
    let rect = Rect::new(-2.0, 1.0, 4.0, 5.0).unwrap();
    assert_eq!(rect.width(), 6.0);
    assert_eq!(rect.height(), 4.0);
    assert_eq!(rect.area(), 24.0);
    assert_eq!(rect.center(), Point::new(1.0, 3.0).unwrap());
}

#[test]
fn rectangle_intersection_handles_touching_edges() {
    let left = Rect::new(0.0, 0.0, 2.0, 2.0).unwrap();
    let right = Rect::new(2.0, 0.0, 4.0, 2.0).unwrap();
    assert_eq!(left.intersection(right).area(), 0.0);
}
