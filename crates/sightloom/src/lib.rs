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
//! detections → tracks → (optional re-id) → zone events → VisionIndex snapshot
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
mod analysis_bridge;
#[cfg(feature = "std")]
mod ingest;
#[cfg(feature = "std")]
mod session;

#[cfg(feature = "std")]
pub use analysis_bridge::{
    DemoSpanDto, RedactionIntervalExportDto, SeedResult, UncertainIntervalDto,
    analysis_series_from_index, anomaly_reason_label, baseline_from_index,
    detect_anomalies_from_index, mine_patterns_from_index,
};
#[cfg(feature = "std")]
pub use ingest::{
    DropPolicy, IngestDecision, IngestMetrics, IngestPolicy, LateFramePolicy, OutOfOrderPolicy,
    SourceLifecycle, SourceWatermark, evaluate_stamp,
};
#[cfg(feature = "std")]
pub use session::{
    IndexSession, PhotoSearchResult, SessionError, TrackEmbeddingHit, TrackSpanExport,
};

pub use sightloom_analysis as analysis;
pub use sightloom_core as core;
pub use sightloom_index as index;
pub use sightloom_reid as reid;
pub use sightloom_tracking as tracking;
