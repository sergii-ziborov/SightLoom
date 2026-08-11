//! Non-allocating core error definitions.

/// An error produced by a core processing operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoreError {
    /// A numeric input is NaN or infinite.
    NonFinite,
    /// Caller-owned storage has no room for another value.
    InsufficientCapacity,
}

/// An error produced while constructing validated geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeometryError {
    /// At least one coordinate is NaN or infinite.
    NonFinite,
    /// A rectangle's right or bottom edge precedes its opposite edge.
    InvertedBounds,
}
