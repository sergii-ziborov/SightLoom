//! Facade crate for `SightLoom` host integration.
//!
//! Workspace layout:
//! - `sightloom-core` — geometry, detections, NMS, zones, envelopes
//! - `sightloom-tracking` — ByteTrack-style tracking, smoothers, trajectories
//! - `sightloom-index` — observations, masks, `VisionIndex` storage
//! - `sightloom-analysis` — zone analytics, patterns, anomalies
//! - `sightloom-reid` — subject references and identity contracts
//! - `sightloom` — this facade
//!
//! ```text
//! detections → tracks → zone events → VisionIndex snapshot
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
mod session;

#[cfg(feature = "std")]
pub use session::{IndexSession, SessionError};

pub use sightloom_analysis as analysis;
pub use sightloom_core as core;
pub use sightloom_index as index;
pub use sightloom_reid as reid;
pub use sightloom_tracking as tracking;
