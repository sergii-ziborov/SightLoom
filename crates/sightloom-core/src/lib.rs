#![cfg_attr(not(feature = "std"), no_std)]
//! Portable, allocation-conscious primitives for `SightLoom` vision pipelines.

mod error;
mod geometry;

pub use error::GeometryError;
pub use geometry::{Point, Rect};
