//! Queryable video memory sidecar storage for `SightLoom`.
//!
//! `SightLoom` returns data (tracks, masks, appearances, evidence handles), not
//! drawn pixels. This crate stores and indexes that data for later query,
//! pattern mining, and evidence reels.
//!
//! Minimum surface:
//! - versioned manifest
//! - track sample stream (in-memory; CBOR/Arrow files later)
//! - compact mask store
//! - event/subject index (in-memory; `SQLite` files later)
//! - source hashes and model/threshold provenance

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

mod error;
mod event_index;
mod manifest;
mod mask_store;
mod provenance;
mod store;
mod track_stream;

pub use error::MemoryError;
pub use event_index::EventRecord;
pub use manifest::{MANIFEST_VERSION, MemoryManifest, SourceEntry};
pub use provenance::{ModelProvenance, SourceHash};
pub use track_stream::TrackSample;

#[cfg(feature = "std")]
pub use event_index::EventIndex;
#[cfg(feature = "std")]
pub use mask_store::MaskStore;
#[cfg(feature = "std")]
pub use store::VideoMemory;
#[cfg(feature = "std")]
pub use track_stream::TrackStream;
