//! Non-allocating core error definitions.

use core::fmt;

/// An error produced by a core processing operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoreError {
    /// A numeric input is NaN or infinite.
    NonFinite,
    /// Caller-owned storage has no room for another value.
    InsufficientCapacity,
    /// A suppression threshold is non-finite or outside `0.0..=1.0`.
    InvalidThreshold,
    /// Caller-owned NMS scratch is shorter than the detections slice.
    InsufficientScratch,
    /// A media time has a zero timescale.
    InvalidMediaTime,
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NonFinite => "non-finite numeric value",
            Self::InsufficientCapacity => "insufficient capacity",
            Self::InvalidThreshold => "invalid threshold",
            Self::InsufficientScratch => "insufficient NMS scratch",
            Self::InvalidMediaTime => "invalid media time (zero timescale)",
        })
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CoreError {}

/// An error produced while constructing validated geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeometryError {
    /// At least one coordinate is NaN or infinite.
    NonFinite,
    /// A rectangle's right or bottom edge precedes its opposite edge.
    InvertedBounds,
    /// A line segment's endpoints are identical.
    DegenerateSegment,
    /// A polygon has fewer than three supplied points.
    TooFewPoints,
}

impl fmt::Display for GeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NonFinite => "non-finite geometry coordinate",
            Self::InvertedBounds => "inverted rectangle bounds",
            Self::DegenerateSegment => "degenerate line segment",
            Self::TooFewPoints => "polygon has too few points",
        })
    }
}

#[cfg(feature = "std")]
impl std::error::Error for GeometryError {}
