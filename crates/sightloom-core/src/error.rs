//! Geometry error definitions.

/// An error produced while constructing validated geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeometryError {
    /// At least one coordinate is NaN or infinite.
    NonFinite,
    /// A rectangle's right or bottom edge precedes its opposite edge.
    InvertedBounds,
}
