#![cfg_attr(not(feature = "std"), no_std)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::similar_names
)]
//! Detection and trajectory smoothing for `SightLoom`.

#[cfg(feature = "alloc")]
extern crate alloc;

mod error;
mod smoother;
mod trajectory;

pub use error::SmoothError;
pub use smoother::{DetectionSmoother, SmoothConfig, interpolate_bbox};
pub use trajectory::{TrajectoryHistory, TrajectorySample};
