//! Borrowed polygon membership geometry.

use core::cmp::Ordering;

use crate::{GeometryError, Point, line::point_on_segment, orientation::orientation_sign};

/// A polygon borrowing its caller-owned vertex slice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Polygon<'a> {
    points: &'a [Point],
}

impl<'a> Polygon<'a> {
    /// Creates a polygon from at least three supplied points.
    ///
    /// # Errors
    ///
    /// Returns [`GeometryError::TooFewPoints`] when fewer than three points
    /// are supplied.
    pub fn new(points: &'a [Point]) -> Result<Self, GeometryError> {
        if points.len() < 3 {
            return Err(GeometryError::TooFewPoints);
        }

        Ok(Self { points })
    }

    /// Returns the caller-owned vertex slice.
    #[must_use]
    pub const fn points(self) -> &'a [Point] {
        self.points
    }

    /// Returns whether a point is inside or on this polygon's boundary.
    #[must_use]
    pub fn contains(self, point: Point) -> bool {
        let mut inside = false;

        for index in 0..self.points.len() {
            let start = self.points[index];
            let end = self.points[(index + 1) % self.points.len()];
            if point_on_segment(start, end, point) {
                return true;
            }

            if (start.y() > point.y()) != (end.y() > point.y()) {
                let side = orientation_sign(start, end, point);
                if (end.y() > start.y() && side == Ordering::Greater)
                    || (end.y() < start.y() && side == Ordering::Less)
                {
                    inside = !inside;
                }
            }
        }

        inside
    }
}
