//! Borrowed polygon membership geometry.

use crate::{GeometryError, Point, line::point_on_segment};

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
                let horizontal_at_ray = f64::from(start.x())
                    + (f64::from(end.x()) - f64::from(start.x()))
                        * (f64::from(point.y()) - f64::from(start.y()))
                        / (f64::from(end.y()) - f64::from(start.y()));
                if f64::from(point.x()) < horizontal_at_ray {
                    inside = !inside;
                }
            }
        }

        inside
    }
}
