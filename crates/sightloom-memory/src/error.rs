//! Memory store errors.

/// An error produced by sidecar memory operations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryError {
    /// A numeric or structural value is invalid.
    Invalid,
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
