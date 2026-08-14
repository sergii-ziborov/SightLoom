//! Host package errors.

use core::fmt;

/// Errors from host model config, registry, preprocess, or adapters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostError {
    /// Invalid configuration.
    Config(String),
    /// Model file / registry path missing.
    ModelNotFound(String),
    /// Preprocess failure.
    Preprocess(String),
    /// Adapter / runtime failure.
    Runtime(String),
    /// Download not available or failed (step 2+).
    Download(String),
    /// I/O failure.
    Io(String),
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(m) => write!(f, "host config: {m}"),
            Self::ModelNotFound(m) => write!(f, "model not found: {m}"),
            Self::Preprocess(m) => write!(f, "preprocess: {m}"),
            Self::Runtime(m) => write!(f, "runtime: {m}"),
            Self::Download(m) => write!(f, "download: {m}"),
            Self::Io(m) => write!(f, "io: {m}"),
        }
    }
}

impl std::error::Error for HostError {}
