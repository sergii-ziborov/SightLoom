#![cfg_attr(not(feature = "std"), no_std)]
//! Portable, allocation-conscious primitives for `SightLoom` vision pipelines.

mod detection;
mod error;
mod geometry;
mod nms;
mod overlap;

pub use detection::{ClassId, Detection, DetectionBatch, TrackId, ZoneId};
pub use error::{CoreError, GeometryError};
pub use geometry::{Point, Rect};
pub use nms::{NmsConfig, NmsMode, OverlapMetric, nms_in_place};
pub use overlap::{intersection_area, ios, iou};
