//! Memory store errors.

use core::fmt;

/// An error produced by sidecar memory operations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryError {
    /// A numeric or structural value is invalid.
    Invalid,
    /// Structured validation failure summary.
    #[cfg(feature = "std")]
    Validation(String),
    /// Underlying I/O failure (host only).
    #[cfg(feature = "std")]
    Io(String),
    /// Serialization failure.
    #[cfg(feature = "std")]
    Serde(String),
    /// Requested entity was not found.
    NotFound,
    /// Caller-owned buffer is too small.
    InsufficientCapacity,
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid => f.write_str("invalid memory value"),
            #[cfg(feature = "std")]
            Self::Validation(msg) => write!(f, "validation failed: {msg}"),
            #[cfg(feature = "std")]
            Self::Io(msg) => write!(f, "io error: {msg}"),
            #[cfg(feature = "std")]
            Self::Serde(msg) => write!(f, "serde error: {msg}"),
            Self::NotFound => f.write_str("entity not found"),
            Self::InsufficientCapacity => f.write_str("insufficient capacity"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MemoryError {}
