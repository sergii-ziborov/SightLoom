//! Versioned memory package manifest.

use crate::{MemoryError, ModelProvenance, SourceHash};

/// Current manifest schema version written by this crate.
pub const MANIFEST_VERSION: u32 = 1;

/// Top-level sidecar package description.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct MemoryManifest {
    /// Schema version.
    pub version: u32,
    /// Human-readable package name.
    pub name: String,
    /// Media sources included in this package.
    pub sources: Vec<SourceEntry>,
    /// Relative path to the track sample stream (CBOR/Arrow later).
    pub track_stream_path: String,
    /// Relative path to the compact mask store.
    pub mask_store_path: String,
    /// Relative path to the `SQLite` event/subject index.
    pub event_index_path: String,
    /// Optional model provenance for the package.
    pub provenance: Option<ModelProvenance>,
}

/// One media source recorded in the manifest.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct SourceEntry {
    /// Stable source id used in frame stamps.
    pub source_id: u32,
    /// Original URI or path (opaque to `SightLoom`).
    pub uri: String,
    /// Optional content hash.
    pub hash: Option<SourceHash>,
}

impl MemoryManifest {
    /// Creates a v1 manifest with default relative paths.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            version: MANIFEST_VERSION,
            name: name.into(),
            sources: Vec::new(),
            track_stream_path: "tracks.cbor".into(),
            mask_store_path: "masks.bin".into(),
            event_index_path: "events.sqlite".into(),
            provenance: None,
        }
    }

    /// Validates schema version and required paths.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Invalid`] when the version or paths are unusable.
    pub fn validate(&self) -> Result<(), MemoryError> {
        if self.version == 0 || self.version > MANIFEST_VERSION {
            return Err(MemoryError::Invalid);
        }
        if self.name.is_empty()
            || self.track_stream_path.is_empty()
            || self.mask_store_path.is_empty()
            || self.event_index_path.is_empty()
        {
            return Err(MemoryError::Invalid);
        }
        Ok(())
    }
}

#[cfg(feature = "std")]
impl MemoryManifest {
    /// Serializes the manifest to pretty JSON.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Serde`] on serialization failure.
    pub fn to_json(&self) -> Result<String, MemoryError> {
        serde_json::to_string_pretty(self).map_err(|error| MemoryError::Serde(error.to_string()))
    }

    /// Parses a manifest from JSON.
    ///
    /// # Errors
    ///
    /// Returns serde or validation errors.
    pub fn from_json(text: &str) -> Result<Self, MemoryError> {
        let manifest: Self =
            serde_json::from_str(text).map_err(|error| MemoryError::Serde(error.to_string()))?;
        manifest.validate()?;
        Ok(manifest)
    }
}
