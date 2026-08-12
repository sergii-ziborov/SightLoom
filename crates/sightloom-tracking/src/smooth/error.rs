//! Smoothing errors.

/// An error produced by smoothing operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SmoothError {
    /// A numeric input is NaN or infinite.
    NonFinite,
    /// Caller-owned storage is too small.
    InsufficientCapacity,
    /// A configuration value is invalid.
    InvalidConfig,
}
