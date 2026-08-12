//! Package-level video memory facade.

use crate::{
    EventIndex, MaskStore, MemoryError, MemoryManifest, ModelProvenance, SourceEntry, TrackStream,
};

/// In-memory `SightLoom` video memory package.
///
/// Host I/O (writing JSON/CBOR/SQLite files under a directory) is intentionally
/// thin; this type owns the queryable structures first.
#[cfg(feature = "std")]
#[derive(Clone, Debug)]
pub struct VideoMemory {
    /// Package manifest.
    pub manifest: MemoryManifest,
    /// Track sample stream.
    pub tracks: TrackStream,
    /// Compact mask store.
    pub masks: MaskStore,
    /// Event/subject index.
    pub events: EventIndex,
}

#[cfg(feature = "std")]
impl VideoMemory {
    /// Creates an empty named package.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            manifest: MemoryManifest::new(name),
            tracks: TrackStream::new(),
            masks: MaskStore::new(),
            events: EventIndex::new(),
        }
    }

    /// Registers a media source on the manifest.
    pub fn add_source(&mut self, entry: SourceEntry) {
        self.manifest.sources.push(entry);
    }

    /// Attaches model provenance.
    pub fn set_provenance(&mut self, provenance: ModelProvenance) {
        self.manifest.provenance = Some(provenance);
    }

    /// Validates the package manifest.
    ///
    /// # Errors
    ///
    /// Propagates manifest validation errors.
    pub fn validate(&self) -> Result<(), MemoryError> {
        self.manifest.validate()
    }
}
