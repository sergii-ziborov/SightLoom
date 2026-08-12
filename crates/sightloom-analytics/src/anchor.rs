//! Anchor policies for zone membership tests.

use sightloom_core::{Point, Rect};

/// Which point of a detection is tested against a zone.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AnchorPolicy {
    /// Geometric center of the bounding box.
    #[default]
    Center,
    /// Bottom-center (feet / ground contact approximation).
    BottomCenter,
    /// Explicit mask centroid when provided by the caller.
    MaskCentroid,
}

impl AnchorPolicy {
    /// Resolves the anchor point for a bounding box.
    ///
    /// For [`AnchorPolicy::MaskCentroid`], pass the precomputed centroid as
    /// `mask_centroid`; when `None`, falls back to center.
    #[must_use]
    pub fn anchor(self, bbox: Rect, mask_centroid: Option<Point>) -> Point {
        match self {
            Self::Center => bbox.center(),
            Self::BottomCenter => Point::new(bbox.left() * 0.5 + bbox.right() * 0.5, bbox.bottom())
                .unwrap_or_else(|_| bbox.center()),
            Self::MaskCentroid => mask_centroid.unwrap_or_else(|| bbox.center()),
        }
    }
}
