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
#[cfg(feature = "alloc")]
mod metrics;
#[cfg(feature = "std")]
mod multi_source;
#[cfg(feature = "alloc")]
mod synthetic;
mod track;

pub mod smooth;

pub use config::ByteTrackConfig;
pub use error::TrackError;
pub use kalman::KalmanState;
pub use matching::{AssignResult, AssignScratch, MatchCandidate, greedy_iou_assign};
pub use track::{Track, TrackState};

#[cfg(feature = "alloc")]
pub use bytetrack::{ByteTracker, TrackerSnapshot};
#[cfg(feature = "alloc")]
pub use metrics::{BaselineMotMetrics, MotFrame, MotObject, evaluate_baseline_mot, mot_from_track};
#[cfg(feature = "std")]
pub use multi_source::{
    MultiSourceCheckpoint, MultiSourceTracker, SourceTrackerCheckpoint, TrackedDetection,
    UidMapEntry,
};
#[cfg(feature = "alloc")]
pub use synthetic::{run_synthetic_crossing, run_synthetic_parallel_walk};

pub use smooth::{
    DetectionSmoother, SmoothConfig, SmoothError, TrajectoryHistory, TrajectorySample,
    interpolate_bbox,
};
