//! Public zone-geometry contract tests.

use sightloom_core::{
    GeometryError, LineSegment, LineSide, Point, Polygon, crosses_segment, line_side,
};

fn point(x: f32, y: f32) -> Point {
    Point::new(x, y).unwrap()
}

fn segment(start: (f32, f32), end: (f32, f32)) -> LineSegment {
    LineSegment::new(point(start.0, start.1), point(end.0, end.1)).unwrap()
}

// Catches accepting an endpoint pair that cannot define an orientation.
#[test]
fn line_segment_rejects_a_directionless_segment() {
    let endpoint = point(1.0, 2.0);
    assert_eq!(
        LineSegment::new(endpoint, endpoint),
        Err(GeometryError::DegenerateSegment)
    );
}

// Catches reversed orientation, treating collinearity as approximate, or f32 overflow.
#[test]
fn line_side_uses_an_exact_algebraic_orientation() {
    let horizontal = segment((0.0, 0.0), (4.0, 0.0));
    assert_eq!(line_side(horizontal, point(2.0, 1.0)), LineSide::Left);
    assert_eq!(line_side(horizontal, point(2.0, -1.0)), LineSide::Right);
    assert_eq!(line_side(horizontal, point(2.0, 0.0)), LineSide::On);

    let large = segment((0.0, 0.0), (f32::MAX, f32::MAX * 0.5));
    assert_eq!(
        line_side(large, point(f32::MAX * 0.5, f32::MAX)),
        LineSide::Left
    );
}

// Catches infinite-line intersection in place of closed finite-segment intersection.
#[test]
fn finite_segments_do_not_cross_only_on_their_extensions() {
    assert!(crosses_segment(
        segment((0.0, 0.0), (4.0, 0.0)),
        segment((2.0, -1.0), (2.0, 1.0))
    ));
    assert!(crosses_segment(
        segment((0.0, 0.0), (2.0, 0.0)),
        segment((2.0, 0.0), (2.0, 2.0))
    ));
    assert!(crosses_segment(
        segment((0.0, 0.0), (3.0, 0.0)),
        segment((2.0, 0.0), (4.0, 0.0))
    ));
    assert!(!crosses_segment(
        segment((0.0, 0.0), (1.0, 0.0)),
        segment((2.0, 0.0), (3.0, 0.0))
    ));
    assert!(!crosses_segment(
        segment((0.0, 0.0), (1.0, 0.0)),
        segment((2.0, -1.0), (2.0, 1.0))
    ));
}

// Catches constructing a polygon that lacks enough supplied vertices for edges.
#[test]
fn polygon_rejects_fewer_than_three_supplied_points() {
    let points = [point(0.0, 0.0), point(1.0, 0.0)];
    assert_eq!(Polygon::new(&points), Err(GeometryError::TooFewPoints));
}

// Catches omitted boundary handling or incorrect ordinary polygon membership.
#[test]
fn polygon_membership_covers_inside_outside_and_boundary_points() {
    let points = [
        point(0.0, 0.0),
        point(4.0, 0.0),
        point(4.0, 4.0),
        point(0.0, 4.0),
    ];
    let polygon = Polygon::new(&points).unwrap();
    assert!(polygon.contains(point(2.0, 2.0)));
    assert!(!polygon.contains(point(5.0, 2.0)));
    assert!(polygon.contains(point(0.0, 2.0)));
    assert!(polygon.contains(point(0.0, 0.0)));
    assert_eq!(polygon.points(), &points);
}

// Catches winding-only logic rather than even-odd parity across concave crossings.
#[test]
fn polygon_uses_even_odd_membership_for_concave_and_self_intersecting_shapes() {
    let concave = [
        point(0.0, 0.0),
        point(4.0, 0.0),
        point(4.0, 4.0),
        point(2.0, 2.0),
        point(0.0, 4.0),
    ];
    let concave = Polygon::new(&concave).unwrap();
    assert!(concave.contains(point(3.0, 1.0)));
    assert!(!concave.contains(point(2.0, 3.0)));

    let bow_tie = [
        point(0.0, 0.0),
        point(4.0, 4.0),
        point(0.0, 4.0),
        point(4.0, 0.0),
    ];
    let bow_tie = Polygon::new(&bow_tie).unwrap();
    assert!(bow_tie.contains(point(2.0, 0.5)));
    assert!(!bow_tie.contains(point(0.5, 2.0)));
    assert!(bow_tie.contains(point(2.0, 2.0)));
}

// Catches parity toggling from zero-length edges or rejecting optional closure.
#[test]
fn polygon_accepts_repeated_adjacent_and_closing_vertices() {
    let points = [
        point(0.0, 0.0),
        point(4.0, 0.0),
        point(4.0, 0.0),
        point(4.0, 4.0),
        point(0.0, 4.0),
        point(0.0, 0.0),
    ];
    assert!(Polygon::new(&points).unwrap().contains(point(2.0, 2.0)));
}
