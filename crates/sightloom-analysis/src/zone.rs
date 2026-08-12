//! Enhanced polygon-zone analytics with dwell and occupancy.

use crate::{AnalyticsError, AnalyticsEvent, ZoneAnalyticsConfig};
use sightloom_core::{ClassId, MediaTime, Point, Polygon, Rect, TrackId, ZoneId};

#[derive(Clone, Copy, Debug)]
struct TrackAnalytics {
    track_id: TrackId,
    class_id: Option<ClassId>,
    inside_streak: u32,
    outside_streak: u32,
    missed: u32,
    confirmed_inside: bool,
    dwell_active: bool,
    dwell_started_at: Option<MediaTime>,
    visit_count: u32,
    last_seen_frame: u64,
}

/// Polygon zone analytics with hysteresis, dwell, and occupancy.
pub struct ZoneAnalytics<'a, const N: usize> {
    zone_id: ZoneId,
    polygon: Polygon<'a>,
    config: ZoneAnalyticsConfig,
    slots: [Option<TrackAnalytics>; N],
    occupancy: u32,
}

impl<'a, const N: usize> ZoneAnalytics<'a, N> {
    /// Creates analytics for `polygon` with a validated configuration.
    ///
    /// # Errors
    ///
    /// Returns config validation errors.
    pub fn new(
        zone_id: ZoneId,
        polygon: Polygon<'a>,
        config: ZoneAnalyticsConfig,
    ) -> Result<Self, AnalyticsError> {
        let config = config.validate()?;
        Ok(Self {
            zone_id,
            polygon,
            config,
            slots: [None; N],
            occupancy: 0,
        })
    }

    /// Current confirmed occupancy.
    #[must_use]
    pub const fn occupancy(&self) -> u32 {
        self.occupancy
    }

    /// Updates one track sample and writes zero or more events to `output`.
    ///
    /// # Errors
    ///
    /// Returns [`AnalyticsError::InsufficientCapacity`] when no track slot or
    /// event capacity remains. On error, monitor state is unchanged for the
    /// capacity path when possible; partial event writes do not occur.
    #[allow(clippy::too_many_arguments, clippy::missing_panics_doc)]
    pub fn update(
        &mut self,
        track_id: TrackId,
        bbox: Rect,
        class_id: Option<ClassId>,
        mask_centroid: Option<Point>,
        frame_index: u64,
        now: MediaTime,
        output: &mut [AnalyticsEvent],
    ) -> Result<usize, AnalyticsError> {
        if let Some(filter) = self.config.class_filter
            && class_id != Some(filter)
        {
            return Ok(0);
        }

        let anchor = self.config.anchor.anchor(bbox, mask_centroid);
        let raw_inside = self.polygon.contains(anchor);
        let index = self.find_or_alloc(track_id, class_id, frame_index)?;

        // Work on a local copy so capacity failures leave state intact.
        let mut state = self.slots[index].expect("allocated");
        let previous_occupancy = self.occupancy;
        let mut events = [None; 4];
        let mut event_count = 0_usize;

        if raw_inside {
            state.missed = 0;
            state.inside_streak = state.inside_streak.saturating_add(1);
            state.outside_streak = 0;
        } else {
            state.outside_streak = state.outside_streak.saturating_add(1);
            state.inside_streak = 0;
        }
        state.last_seen_frame = frame_index;
        state.class_id = class_id.or(state.class_id);

        if !state.confirmed_inside && state.inside_streak >= self.config.enter_hysteresis {
            state.confirmed_inside = true;
            self.occupancy = self.occupancy.saturating_add(1);
            events[event_count] = Some(AnalyticsEvent::Entered {
                track_id,
                zone_id: self.zone_id,
                class_id: state.class_id,
            });
            event_count += 1;

            let debounce_ok = self.config.dwell_start_debounce_ns == 0
                || state.dwell_started_at.is_none_or(|start| {
                    now.duration_since_ns(start) >= self.config.dwell_start_debounce_ns
                });
            if debounce_ok && !state.dwell_active {
                state.dwell_active = true;
                state.dwell_started_at = Some(now);
                events[event_count] = Some(AnalyticsEvent::DwellStarted {
                    track_id,
                    zone_id: self.zone_id,
                    at: now,
                });
                event_count += 1;
            }
        }

        if state.confirmed_inside && state.outside_streak >= self.config.exit_hysteresis {
            state.confirmed_inside = false;
            self.occupancy = self.occupancy.saturating_sub(1);
            events[event_count] = Some(AnalyticsEvent::Exited {
                track_id,
                zone_id: self.zone_id,
                class_id: state.class_id,
            });
            event_count += 1;

            if state.dwell_active {
                let started = state.dwell_started_at.unwrap_or(now);
                let duration_ns = now.duration_since_ns(started);
                state.dwell_active = false;
                state.visit_count = state.visit_count.saturating_add(1);
                events[event_count] = Some(AnalyticsEvent::DwellEnded {
                    track_id,
                    zone_id: self.zone_id,
                    duration_ns,
                    visit_count: state.visit_count,
                });
                event_count += 1;
                state.dwell_started_at = None;
            }
        }

        if self.occupancy != previous_occupancy {
            events[event_count] = Some(AnalyticsEvent::OccupancyChanged {
                zone_id: self.zone_id,
                occupancy: self.occupancy,
            });
            event_count += 1;
        }

        if event_count > output.len() {
            // roll back occupancy
            self.occupancy = previous_occupancy;
            return Err(AnalyticsError::InsufficientCapacity);
        }

        self.slots[index] = Some(state);
        for (i, event) in events.into_iter().flatten().enumerate() {
            output[i] = event;
        }
        Ok(event_count)
    }

    /// Notes a missed frame for a track; may trigger exit after tolerance.
    ///
    /// # Errors
    ///
    /// Returns capacity errors when emitting exit events.
    #[allow(clippy::missing_panics_doc)]
    pub fn note_miss(
        &mut self,
        track_id: TrackId,
        now: MediaTime,
        output: &mut [AnalyticsEvent],
    ) -> Result<usize, AnalyticsError> {
        let Some(index) = self.find(track_id) else {
            return Ok(0);
        };
        let mut state = self.slots[index].expect("found");
        let previous_occupancy = self.occupancy;
        state.missed = state.missed.saturating_add(1);
        if state.missed <= self.config.missed_frame_tolerance {
            self.slots[index] = Some(state);
            return Ok(0);
        }
        state.outside_streak = state.outside_streak.saturating_add(1);
        state.inside_streak = 0;

        let mut written = 0_usize;
        if state.confirmed_inside && state.outside_streak >= self.config.exit_hysteresis {
            if output.len() < 2 {
                return Err(AnalyticsError::InsufficientCapacity);
            }
            state.confirmed_inside = false;
            self.occupancy = self.occupancy.saturating_sub(1);
            output[written] = AnalyticsEvent::Exited {
                track_id,
                zone_id: self.zone_id,
                class_id: state.class_id,
            };
            written += 1;
            if state.dwell_active {
                let started = state.dwell_started_at.unwrap_or(now);
                state.dwell_active = false;
                state.visit_count = state.visit_count.saturating_add(1);
                output[written] = AnalyticsEvent::DwellEnded {
                    track_id,
                    zone_id: self.zone_id,
                    duration_ns: now.duration_since_ns(started),
                    visit_count: state.visit_count,
                };
                written += 1;
                state.dwell_started_at = None;
            }
            if written < output.len() && self.occupancy != previous_occupancy {
                output[written] = AnalyticsEvent::OccupancyChanged {
                    zone_id: self.zone_id,
                    occupancy: self.occupancy,
                };
                written += 1;
            }
        }
        self.slots[index] = Some(state);
        Ok(written)
    }

    /// Visit count for a track, if known.
    #[must_use]
    pub fn visit_count(&self, track_id: TrackId) -> Option<u32> {
        self.find(track_id)
            .and_then(|i| self.slots[i].map(|s| s.visit_count))
    }

    /// Time currently spent inside for an active dwell, in nanoseconds.
    #[must_use]
    pub fn time_in_zone_ns(&self, track_id: TrackId, now: MediaTime) -> Option<i64> {
        let state = self.find(track_id).and_then(|i| self.slots[i])?;
        if !state.dwell_active {
            return None;
        }
        Some(now.duration_since_ns(state.dwell_started_at?))
    }

    fn find(&self, track_id: TrackId) -> Option<usize> {
        self.slots.iter().position(|slot| {
            slot.as_ref()
                .is_some_and(|state| state.track_id == track_id)
        })
    }

    fn find_or_alloc(
        &mut self,
        track_id: TrackId,
        class_id: Option<ClassId>,
        frame_index: u64,
    ) -> Result<usize, AnalyticsError> {
        if let Some(index) = self.find(track_id) {
            return Ok(index);
        }
        let free = self
            .slots
            .iter()
            .position(Option::is_none)
            .ok_or(AnalyticsError::InsufficientCapacity)?;
        self.slots[free] = Some(TrackAnalytics {
            track_id,
            class_id,
            inside_streak: 0,
            outside_streak: 0,
            missed: 0,
            confirmed_inside: false,
            dwell_active: false,
            dwell_started_at: None,
            visit_count: 0,
            last_seen_frame: frame_index,
        });
        Ok(free)
    }
}
