//! Finite line-segment geometry.

use core::cmp::Ordering;

use crate::{GeometryError, Point, orientation::orientation_sign};

/// A finite, directed line segment with distinct endpoints.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineSegment {
    start: Point,
    end: Point,
}

impl LineSegment {
    /// Creates a line segment with distinct endpoints.
    ///
    /// # Errors
    ///
    /// Returns [`GeometryError::DegenerateSegment`] when both endpoints are
    /// equal.
    pub fn new(start: Point, end: Point) -> Result<Self, GeometryError> {
        if start == end {
            return Err(GeometryError::DegenerateSegment);
        }

        Ok(Self { start, end })
    }

    /// Returns the directed segment's starting endpoint.
    #[must_use]
    pub const fn start(self) -> Point {
        self.start
    }

    /// Returns the directed segment's ending endpoint.
    #[must_use]
    pub const fn end(self) -> Point {
        self.end
    }
}

/// The algebraic side of a point relative to a directed line.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LineSide {
    /// The point has positive orientation relative to the line.
    Left,
    /// The point is exactly collinear with the line.
    On,
    /// The point has negative orientation relative to the line.
    Right,
}

/// Returns the algebraic side of a point relative to a directed segment.
#[must_use]
pub fn line_side(segment: LineSegment, point: Point) -> LineSide {
    match orientation_sign(segment.start, segment.end, point) {
        Ordering::Greater => LineSide::Left,
        Ordering::Equal => LineSide::On,
        Ordering::Less => LineSide::Right,
    }
}

/// Returns whether two closed finite segments intersect.
#[must_use]
pub fn crosses_segment(first: LineSegment, second: LineSegment) -> bool {
    let first_start_side = line_side(first, second.start);
    let first_end_side = line_side(first, second.end);
    let second_start_side = line_side(second, first.start);
    let second_end_side = line_side(second, first.end);

    if opposite(first_start_side, first_end_side) && opposite(second_start_side, second_end_side) {
        return true;
    }

    (first_start_side == LineSide::On && point_on_segment(first.start, first.end, second.start))
        || (first_end_side == LineSide::On && point_on_segment(first.start, first.end, second.end))
        || (second_start_side == LineSide::On
            && point_on_segment(second.start, second.end, first.start))
        || (second_end_side == LineSide::On
            && point_on_segment(second.start, second.end, first.end))
}

/// Returns whether a point belongs to the closed segment between two points.
#[must_use]
pub(crate) fn point_on_segment(start: Point, end: Point, point: Point) -> bool {
    orientation_sign(start, end, point) == Ordering::Equal
        && point.x() >= start.x().min(end.x())
        && point.x() <= start.x().max(end.x())
        && point.y() >= start.y().min(end.y())
        && point.y() <= start.y().max(end.y())
}

fn opposite(first: LineSide, second: LineSide) -> bool {
    matches!(
        (first, second),
        (LineSide::Left, LineSide::Right) | (LineSide::Right, LineSide::Left)
    )
}
