//! Validated two-dimensional geometry primitives.

use crate::GeometryError;

/// A finite point in two-dimensional pixel space.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    x: f32,
    y: f32,
}

impl Point {
    /// Creates a point when both coordinates are finite.
    ///
    /// # Errors
    ///
    /// Returns [`GeometryError::NonFinite`] when either coordinate is NaN or
    /// infinite.
    pub fn new(x: f32, y: f32) -> Result<Self, GeometryError> {
        if !x.is_finite() || !y.is_finite() {
            return Err(GeometryError::NonFinite);
        }

        Ok(Self { x, y })
    }

    /// Returns the horizontal coordinate.
    #[must_use]
    pub const fn x(self) -> f32 {
        self.x
    }

    /// Returns the vertical coordinate.
    #[must_use]
    pub const fn y(self) -> f32 {
        self.y
    }
}

/// A validated axis-aligned rectangle using half-open pixel coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rect {
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
}

impl Rect {
    /// Creates a rectangle from finite, non-inverted bounds.
    ///
    /// # Errors
    ///
    /// Returns [`GeometryError::NonFinite`] when any bound is NaN or infinite,
    /// and [`GeometryError::InvertedBounds`] when the right or bottom edge
    /// precedes its opposite edge.
    pub fn new(left: f32, top: f32, right: f32, bottom: f32) -> Result<Self, GeometryError> {
        if !left.is_finite() || !top.is_finite() || !right.is_finite() || !bottom.is_finite() {
            return Err(GeometryError::NonFinite);
        }
        if right < left || bottom < top {
            return Err(GeometryError::InvertedBounds);
        }

        Ok(Self {
            left,
            top,
            right,
            bottom,
        })
    }

    /// Returns the left edge.
    #[must_use]
    pub const fn left(self) -> f32 {
        self.left
    }

    /// Returns the top edge.
    #[must_use]
    pub const fn top(self) -> f32 {
        self.top
    }

    /// Returns the right edge.
    #[must_use]
    pub const fn right(self) -> f32 {
        self.right
    }

    /// Returns the bottom edge.
    #[must_use]
    pub const fn bottom(self) -> f32 {
        self.bottom
    }

    /// Returns the rectangle width.
    #[must_use]
    pub fn width(self) -> f32 {
        self.right - self.left
    }

    /// Returns the rectangle height.
    #[must_use]
    pub fn height(self) -> f32 {
        self.bottom - self.top
    }

    /// Returns the rectangle area.
    #[must_use]
    pub fn area(self) -> f32 {
        self.width() * self.height()
    }

    /// Returns the center point without clamping the rectangle.
    #[must_use]
    pub fn center(self) -> Point {
        Point {
            x: self.left * 0.5 + self.right * 0.5,
            y: self.top * 0.5 + self.bottom * 0.5,
        }
    }

    /// Returns the geometric intersection, including a valid zero-area result
    /// when the rectangles are disjoint or only touch at an edge.
    #[must_use]
    pub fn intersection(self, other: Self) -> Self {
        let left = self.left.max(other.left);
        let top = self.top.max(other.top);
        let right = self.right.min(other.right).max(left);
        let bottom = self.bottom.min(other.bottom).max(top);

        Self {
            left,
            top,
            right,
            bottom,
        }
    }
}
