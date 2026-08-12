#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::cast_precision_loss, clippy::similar_names)]
//! Multi-object tracking, detection smoothing, and trajectory history.
//!
//! Combines Kalman / association tracking with exponential smoothers and
//! fixed-capacity trajectory buffers.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
mod bytetrack;
mod config;
mod error;
mod kalman;
mod matching;
mod track;

pub mod smooth;

pub use config::ByteTrackConfig;
pub use error::TrackError;
pub use kalman::KalmanState;
pub use matching::{AssignResult, AssignScratch, MatchCandidate, greedy_iou_assign};
pub use track::{Track, TrackState};

#[cfg(feature = "alloc")]
pub use bytetrack::ByteTracker;

pub use smooth::{
    DetectionSmoother, SmoothConfig, SmoothError, TrajectoryHistory, TrajectorySample,
    interpolate_bbox,
};
