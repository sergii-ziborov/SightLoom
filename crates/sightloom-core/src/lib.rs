#![cfg_attr(not(feature = "std"), no_std)]
//! Portable, allocation-conscious primitives for `SightLoom` vision pipelines.
//!
//! This crate is the compact foundation: geometry, detections, overlap, NMS,
//! and basic zone events. Rich observations, masks, tracking, smoothing, and
//! memory live in higher crates.

#[cfg(feature = "alloc")]
extern crate alloc;

mod detection;
mod envelope;
mod error;
mod event;
mod geometry;
mod ids;
mod line;
mod nms;
mod orientation;
mod overlap;
#[cfg(feature = "alloc")]
mod owned;
mod polygon;
mod stamp;
#[cfg(feature = "alloc")]
mod tiling;
mod zone;

pub use detection::{Detection, DetectionBatch};
pub use envelope::{EventEnvelope, EventKind, EventPayload};
pub use error::{CoreError, GeometryError};
pub use event::{Direction, VisionEvent};
pub use geometry::{Point, Rect};
pub use ids::{
    AnomalyId, AppearanceId, ClassId, EmbeddingRef, EventId, EvidenceRef, KeypointSetRef,
    LocalTrackId, MaskRef, ObservationId, PatternId, RedactionIntervalId, SourceId, SubjectId,
    TrackId, TrackKey, TrackUid, VisitId, ZoneId,
};
pub use line::{LineSegment, LineSide, crosses_segment, line_side};
pub use nms::{
    NmsConfig, NmsMode, OverlapMetric, SoftNmsConfig, SoftNmsMethod, merge_nms_in_place,
    nms_in_place, soft_nms_in_place,
};
pub use overlap::{intersection_area, ios, iou};
#[cfg(feature = "alloc")]
pub use owned::OwnedDetectionBatch;
pub use polygon::Polygon;
pub use stamp::{FrameStamp, MediaTime};
#[cfg(feature = "alloc")]
pub use tiling::{TileWindow, generate_tiles, tile_to_global};
pub use zone::{LineZoneMonitor, PolygonZoneMonitor};
