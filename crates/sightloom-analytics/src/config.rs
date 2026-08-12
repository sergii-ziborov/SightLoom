//! Zone analytics configuration.

use crate::{AnalyticsError, AnchorPolicy};
use sightloom_core::ClassId;

/// Configuration for [`crate::ZoneAnalytics`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoneAnalyticsConfig {
    /// Which box anchor is tested for membership.
    pub anchor: AnchorPolicy,
    /// Consecutive inside samples required to confirm enter/dwell start.
    pub enter_hysteresis: u32,
    /// Consecutive outside samples required to confirm exit/dwell end.
    pub exit_hysteresis: u32,
    /// Missed frames tolerated before counting as outside.
    pub missed_frame_tolerance: u32,
    /// Optional class filter; `None` accepts all classes.
    pub class_filter: Option<ClassId>,
    /// Debounce dwell start in nanoseconds after enter confirmed.
    pub dwell_start_debounce_ns: i64,
}

impl Default for ZoneAnalyticsConfig {
    fn default() -> Self {
        Self {
            anchor: AnchorPolicy::Center,
            enter_hysteresis: 1,
            exit_hysteresis: 1,
            missed_frame_tolerance: 0,
            class_filter: None,
            dwell_start_debounce_ns: 0,
        }
    }
}

impl ZoneAnalyticsConfig {
    /// Validates hysteresis and debounce fields.
    ///
    /// # Errors
    ///
    /// Returns [`AnalyticsError::InvalidConfig`] when hysteresis is zero.
    pub fn validate(self) -> Result<Self, AnalyticsError> {
        if self.enter_hysteresis == 0 || self.exit_hysteresis == 0 {
            return Err(AnalyticsError::InvalidConfig);
        }
        if self.dwell_start_debounce_ns < 0 {
            return Err(AnalyticsError::InvalidConfig);
        }
        Ok(self)
    }
}
