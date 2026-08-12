//! Oriented rectangle placeholder for deferred P1 support.

use sightloom_core::{CoreError, Point};

/// A center-based oriented rectangle (degrees clockwise from +X).
///
/// Stored so Observation can carry the field; full oriented-NMS and geometry
/// ops are intentionally deferred.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrientedRect {
    /// Center of the box.
    pub center: Point,
    /// Full width along the local X axis.
    pub width: f32,
    /// Full height along the local Y axis.
    pub height: f32,
    /// Rotation in degrees.
    pub angle_deg: f32,
}

impl OrientedRect {
    /// Creates an oriented rect when all numbers are finite and sizes non-negative.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::NonFinite`] for non-finite inputs.
    pub fn new(center: Point, width: f32, height: f32, angle_deg: f32) -> Result<Self, CoreError> {
        if !width.is_finite()
            || !height.is_finite()
            || !angle_deg.is_finite()
            || width < 0.0
            || height < 0.0
        {
            return Err(CoreError::NonFinite);
        }
        Ok(Self {
            center,
            width,
            height,
            angle_deg,
        })
    }
}
