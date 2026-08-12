//! Mask construction and operation errors.

/// An error produced by a mask operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaskError {
    /// A numeric input is NaN or infinite.
    NonFinite,
    /// Width or height is zero.
    EmptyDimensions,
    /// Buffer length does not match the declared geometry.
    LengthMismatch,
    /// Caller-owned output storage is too small.
    InsufficientCapacity,
    /// Coordinates fall outside the mask bounds.
    OutOfBounds,
    /// Polygon conversion produced no area.
    EmptyPolygon,
}
