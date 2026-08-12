#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::cast_precision_loss, clippy::similar_names)]
//! ByteTrack-compatible multi-object tracking for `SightLoom`.
//!
//! Consumes model-neutral [`sightloom_core::Detection`] values and produces
//! stable [`sightloom_core::TrackId`] assignments. No inference, video I/O, or
//! pixel drawing is included.

#[cfg(feature = "alloc")]
extern crate alloc;

mod config;
mod error;
mod kalman;
mod matching;
mod track;

#[cfg(feature = "alloc")]
mod bytetrack;

pub use config::ByteTrackConfig;
pub use error::TrackError;
pub use kalman::KalmanState;
pub use matching::{AssignResult, AssignScratch, MatchCandidate, greedy_iou_assign};
pub use track::{Track, TrackState};

#[cfg(feature = "alloc")]
pub use bytetrack::ByteTracker;
