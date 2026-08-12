//! Legacy package facade retained for callers migrating to [`VisionIndex`].

use crate::{
    EventIndex, MaskStore, MemoryError, MemoryManifest, ModelProvenance, SourceEntry, TrackStream,
    VisionIndex,
};

/// In-memory package convenience wrapper.
///
/// Prefer [`VisionIndex`] for new code. This type remains as a thin subset
/// focused on tracks, masks, and a simple event index.
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

    /// Upgrades this package into a full [`VisionIndex`] document shell.
    #[must_use]
    pub fn into_vision_index(self) -> VisionIndex {
        let mut index = VisionIndex::new(self.manifest.name.clone());
        index.header.sources = self.manifest.sources;
        index.header.track_stream_path = self.manifest.track_stream_path;
        index.header.mask_store_path = self.manifest.mask_store_path;
        index.header.event_index_path = self.manifest.event_index_path;
        index.header.provenance = self.manifest.provenance;
        index.tracks = self.tracks;
        index.masks = self.masks;
        index.event_index = self.events;
        index
    }
}
