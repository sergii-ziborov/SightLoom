#![cfg_attr(not(feature = "std"), no_std)]
//! Zone analytics layered above core enter/exit/cross events.
//!
//! Adds dwell, occupancy, hysteresis, class filtering, and anchor policies.
//! Core [`sightloom_core::PolygonZoneMonitor`] remains available for minimal
//! embedded use.

#[cfg(feature = "alloc")]
extern crate alloc;

mod anchor;
mod config;
mod error;
mod events;
mod zone;

pub use anchor::AnchorPolicy;
pub use config::ZoneAnalyticsConfig;
pub use error::AnalyticsError;
pub use events::AnalyticsEvent;
pub use zone::ZoneAnalytics;
