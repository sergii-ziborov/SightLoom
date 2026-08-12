//! Queryable video memory and `VisionIndex` storage for `SightLoom`.
//!
//! `SightLoom` owns the **`VisionIndex`** document: detections, tracks, masks,
//! identities, appearances, visits, events, patterns, anomalies, and evidence.
//!
//! Sibling products own separate documents that must not be mixed in:
//! - `CaptureProject` (capture media / non-destructive edits)
//! - `SemanticEditPlan` (intent / selectors / policies)
//! - `RenderGraph` (executable media graph)
//! - `ExecutionPlan` (stage schedule for rendering)
//!
//! This crate returns data handles, not drawn pixels.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

mod entities;
mod error;
mod event_index;
mod manifest;
mod mask_store;
mod provenance;
mod store;
mod track_stream;
mod vision_index;

pub use entities::{
    AnomalyEvent, AnomalyReason, Appearance, CoOccurrence, PatternKind, PatternRecord, Route,
    Severity, SourceTransition, SubjectProfile, Visit, ZoneStay,
};
pub use error::MemoryError;
pub use event_index::EventRecord;
pub use manifest::{MANIFEST_VERSION, MemoryManifest, SourceEntry};
pub use provenance::{ModelProvenance, SourceHash};
pub use track_stream::TrackSample;
pub use vision_index::{VISION_INDEX_VERSION, VisionIndexHeader, source_entry};

#[cfg(feature = "std")]
pub use event_index::EventIndex;
#[cfg(feature = "std")]
pub use mask_store::MaskStore;
#[cfg(feature = "std")]
pub use store::VideoMemory;
#[cfg(feature = "std")]
pub use track_stream::TrackStream;
#[cfg(feature = "std")]
pub use vision_index::VisionIndex;
