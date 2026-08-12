#![cfg_attr(not(feature = "std"), no_std)]
//! Zone analytics, pattern records, and backend-neutral anomaly events.

#[cfg(feature = "alloc")]
extern crate alloc;

mod anchor;
mod config;
mod envelope;
mod error;
mod events;
mod zone;

#[cfg(feature = "alloc")]
mod anomaly;
#[cfg(feature = "alloc")]
mod pattern;

pub use anchor::AnchorPolicy;
pub use config::ZoneAnalyticsConfig;
pub use envelope::{analytics_to_envelope, track_of};
pub use error::AnalyticsError;
pub use events::AnalyticsEvent;
pub use zone::ZoneAnalytics;

#[cfg(feature = "alloc")]
pub use anomaly::{AnomalyEvent, AnomalyReason, Severity};
#[cfg(feature = "alloc")]
pub use pattern::{PatternKind, PatternRecord};
