//! Compact mask blob store with opaque handles.

use crate::MemoryError;
use sightloom_core::MaskRef;

/// In-memory compact mask store.
///
/// Stores raw encoded mask bytes (RLE or cropped) keyed by [`MaskRef`].
#[cfg(feature = "std")]
#[derive(Clone, Debug, Default)]
pub struct MaskStore {
    next_id: u64,
    blobs: Vec<(MaskRef, Vec<u8>)>,
}

#[cfg(feature = "std")]
impl MaskStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: 1,
            blobs: Vec::new(),
        }
    }

    /// Inserts mask bytes and returns a new handle.
    pub fn insert(&mut self, bytes: impl Into<Vec<u8>>) -> MaskRef {
        let handle = MaskRef(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        self.blobs.push((handle, bytes.into()));
        handle
    }

    /// Looks up mask bytes by handle.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::NotFound`] when the handle is unknown.
    pub fn get(&self, handle: MaskRef) -> Result<&[u8], MemoryError> {
        self.blobs
            .iter()
            .find(|(key, _)| *key == handle)
            .map(|(_, bytes)| bytes.as_slice())
            .ok_or(MemoryError::NotFound)
    }

    /// Returns all stored `(handle, bytes)` pairs in insertion order.
    #[must_use]
    pub fn entries(&self) -> &[(MaskRef, Vec<u8>)] {
        &self.blobs
    }

    /// Rebuilds the store from explicit handle/bytes pairs (used by package load).
    #[must_use]
    pub fn from_entries(entries: Vec<(MaskRef, Vec<u8>)>) -> Self {
        let next_id = entries
            .iter()
            .map(|(handle, _)| handle.0)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
            .max(1);
        Self {
            next_id,
            blobs: entries,
        }
    }
}
