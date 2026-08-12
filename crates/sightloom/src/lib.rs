//! Facade crate for `SightLoom` host integration.
//!
//! Provides an [`IndexSession`] that materializes detector outputs into a
//! serialized [`VisionIndex`] package:
//!
//! ```text
//! detections → tracks → zone events → VisionIndex snapshot
//! ```
//!
//! This crate does not decode video, draw pixels, or own capture/edit documents.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
mod session;

#[cfg(feature = "std")]
pub use session::{IndexSession, SessionError};

pub use sightloom_analytics as analytics;
pub use sightloom_core as core;
pub use sightloom_memory as memory;
pub use sightloom_obs as obs;
pub use sightloom_track as track;
