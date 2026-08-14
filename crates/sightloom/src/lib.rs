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
//!
//! # Quick start (host owns pixels and models)
//!
//! ```rust,no_run
//! use sightloom::core::{Detection, FrameStamp, MediaTime, Rect, SourceId};
//! use sightloom::tracking::ByteTrackConfig;
//! use sightloom::{IndexSession, SourceLifecycle};
//!
//! let mut session = IndexSession::new("demo", ByteTrackConfig::default()).unwrap();
//! let stamp = FrameStamp::new(
//!     SourceId(1),
//!     0,
//!     MediaTime::new(0, 1_000_000_000).unwrap(),
//!     None,
//! );
//! let det = Detection::new(
//!     Rect::new(10.0, 10.0, 40.0, 80.0).unwrap(),
//!     0.9,
//!     None,
//!     None,
//! )
//! .unwrap();
//! session.ingest_detections(stamp, &[det]).unwrap();
//!
//! // Camera dropped — clear motion state for that source.
//! session.apply_source_lifecycle(&SourceLifecycle::Reset {
//!     source_id: SourceId(1),
//! });
//! ```
//!
//! See also: `examples/host_sketch.rs`, `examples/host_model_stub.rs`.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
mod analysis_bridge;
#[cfg(feature = "std")]
mod detector;
#[cfg(feature = "std")]
mod ingest;
#[cfg(feature = "std")]
mod privacy;
#[cfg(feature = "std")]
mod quality;
#[cfg(feature = "std")]
mod session;
#[cfg(feature = "std")]
mod telemetry;

#[cfg(feature = "std")]
pub use analysis_bridge::{
    DemoSpanDto, RedactionIntervalExportDto, SeedResult, UncertainIntervalDto,
    analysis_series_from_index, anomaly_reason_label, baseline_from_index,
    detect_anomalies_from_index, mine_patterns_from_index,
};
#[cfg(feature = "std")]
pub use detector::{
    DetectorAdapter, EmbeddingTask, FrameView, PhotoEmbeddingAdapter, PhotoView, PixelFormat,
};
#[cfg(feature = "std")]
pub use ingest::{
    DropPolicy, FrameQueue, IngestDecision, IngestMetrics, IngestPolicy, LateFramePolicy,
    OutOfOrderPolicy, QueuePushResult, QueuedFrame, SourceLifecycle, SourceWatermark,
    evaluate_stamp, prometheus_text,
};
#[cfg(feature = "std")]
pub use privacy::{RetentionPolicy, RetentionReport, SourceTtl};
#[cfg(feature = "std")]
pub use quality::{
    RedactionPixelSample, RedactionQualityReport, ReidQualityReport, TrackingQualityReport,
    evaluate_redaction_pixels, redaction_coverage_gap,
};
#[cfg(feature = "std")]
pub use session::{
    IndexSession, MemoryAutoRebuild, PhotoSearchResult, SessionError, TrackEmbeddingHit,
    TrackSpanExport,
};
#[cfg(feature = "std")]
pub use telemetry::{
    BufferMetricsExporter, MetricKind, MetricPoint, MetricsExporter, NullMetricsExporter,
    SpanEvent, SpanExporter, SpanStatus, export_prometheus, ingest_frame_span,
    ingest_metric_points, otlp_metrics_json, spans_to_json,
};

pub use sightloom_analysis as analysis;
pub use sightloom_core as core;
pub use sightloom_index as index;
pub use sightloom_reid as reid;
pub use sightloom_tracking as tracking;
