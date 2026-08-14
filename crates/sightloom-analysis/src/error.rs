//! Analytics errors.

use core::fmt;

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

impl fmt::Display for AnalyticsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NonFinite => "non-finite analytics value",
            Self::InsufficientCapacity => "insufficient analytics capacity",
            Self::InvalidConfig => "invalid analytics config",
        })
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AnalyticsError {}
