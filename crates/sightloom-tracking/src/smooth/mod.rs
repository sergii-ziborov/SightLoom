//! Detection smoothing and trajectory history.

mod error;
mod smoother;
mod trajectory;

pub use error::SmoothError;
pub use smoother::{DetectionSmoother, SmoothConfig, interpolate_bbox};
pub use trajectory::{TrajectoryHistory, TrajectorySample};
