//! Tracker errors.

/// An error produced by tracking operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrackError {
    /// A numeric input is NaN or infinite.
    NonFinite,
    /// Caller-owned storage cannot hold another track or assignment.
    InsufficientCapacity,
    /// A configuration threshold is invalid.
    InvalidConfig,
}
