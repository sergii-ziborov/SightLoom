//! Tracker errors.

use core::fmt;

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

impl fmt::Display for TrackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NonFinite => "non-finite tracking value",
            Self::InsufficientCapacity => "insufficient tracking capacity",
            Self::InvalidConfig => "invalid tracker config",
        })
    }
}

#[cfg(feature = "std")]
impl std::error::Error for TrackError {}
