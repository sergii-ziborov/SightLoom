//! Polygon-backed mask bounds.

use crate::MaskError;
use sightloom_core::{Point, Polygon, Rect};

/// A polygonal mask stored as validated vertices in full-frame coordinates.
#[derive(Clone, Copy, Debug)]
pub struct PolygonMask<'a> {
    polygon: Polygon<'a>,
}

impl<'a> PolygonMask<'a> {
    /// Creates a polygon mask from at least three finite points.
    ///
    /// # Errors
    ///
    /// Propagates [`sightloom_core::GeometryError`] mapped to [`MaskError`].
    pub fn new(points: &'a [Point]) -> Result<Self, MaskError> {
        let polygon = Polygon::new(points).map_err(map_geometry)?;
        Ok(Self { polygon })
    }

    /// Returns the underlying polygon.
    #[must_use]
    pub const fn polygon(self) -> Polygon<'a> {
        self.polygon
    }

    /// Returns whether the point lies inside the polygon (even-odd rule).
    #[must_use]
    pub fn contains(&self, point: Point) -> bool {
        self.polygon.contains(point)
    }

    /// Returns the axis-aligned bounding box of the vertices.
    #[must_use]
    pub fn bbox(&self) -> Option<Rect> {
        let points = self.polygon.points();
        if points.is_empty() {
            return None;
        }
        let mut left = points[0].x();
        let mut top = points[0].y();
        let mut right = left;
        let mut bottom = top;
        for point in points.iter().skip(1) {
            left = left.min(point.x());
            top = top.min(point.y());
            right = right.max(point.x());
            bottom = bottom.max(point.y());
        }
        Rect::new(left, top, right, bottom).ok()
    }
}

fn map_geometry(error: sightloom_core::GeometryError) -> MaskError {
    match error {
        sightloom_core::GeometryError::NonFinite => MaskError::NonFinite,
        sightloom_core::GeometryError::TooFewPoints
        | sightloom_core::GeometryError::InvertedBounds
        | sightloom_core::GeometryError::DegenerateSegment => MaskError::EmptyPolygon,
    }
}
