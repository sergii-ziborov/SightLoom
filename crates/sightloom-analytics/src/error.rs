//! Analytics errors.

/// An error produced by zone analytics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnalyticsError {
    /// A numeric input is NaN or infinite.
    NonFinite,
    /// Caller-owned storage is full.
    InsufficientCapacity,
    /// A configuration value is invalid.
    InvalidConfig,
}
