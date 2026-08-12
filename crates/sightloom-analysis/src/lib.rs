#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::cast_precision_loss)]
//! Zone analytics, pattern mining, and backend-neutral anomaly events.

#[cfg(feature = "alloc")]
extern crate alloc;

mod anchor;
mod config;
mod envelope;
mod error;
mod events;
mod stats;
mod zone;

#[cfg(feature = "alloc")]
mod anomaly;
#[cfg(feature = "alloc")]
mod input;
#[cfg(feature = "alloc")]
mod pattern;
#[cfg(feature = "alloc")]
mod pattern_miners;
#[cfg(feature = "alloc")]
mod stat_anomaly;

pub use anchor::AnchorPolicy;
pub use config::ZoneAnalyticsConfig;
pub use envelope::{analytics_to_envelope, track_of};
pub use error::AnalyticsError;
pub use events::AnalyticsEvent;
pub use stats::{day_of_week_ns, hour_of_day_ns, mean, median, stddev, z_score};
pub use zone::ZoneAnalytics;

#[cfg(feature = "alloc")]
pub use anomaly::{AnomalyEvent, AnomalyReason, Severity};
#[cfg(feature = "alloc")]
pub use input::{AnalysisSeries, DurationSample, PairSample, RouteSample, TimedSubjectEvent};
#[cfg(feature = "alloc")]
pub use pattern::{PatternKind, PatternRecord};
#[cfg(feature = "alloc")]
pub use pattern_miners::{
    mine_co_occurrence, mine_day_of_week, mine_dwell_distribution, mine_expected_absence,
    mine_group_formation, mine_patterns, mine_route_sequences, mine_time_of_day,
    mine_visit_periodicity,
};
#[cfg(feature = "alloc")]
pub use stat_anomaly::{BaselineStats, StatAnomalyConfig, build_baseline, detect_statistical};
