//! Extended zone analytics events layered above core enter/exit/cross.

use sightloom_core::{ClassId, MediaTime, TrackId, ZoneId};

/// Analytics events emitted by enhanced zone monitors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AnalyticsEvent {
    /// A track entered a zone (mirrors core, with optional class filter context).
    Entered {
        /// Track identifier.
        track_id: TrackId,
        /// Zone identifier.
        zone_id: ZoneId,
        /// Optional class at entry.
        class_id: Option<ClassId>,
    },
    /// A track exited a zone.
    Exited {
        /// Track identifier.
        track_id: TrackId,
        /// Zone identifier.
        zone_id: ZoneId,
        /// Optional class at exit.
        class_id: Option<ClassId>,
    },
    /// Dwell timing started after hysteresis confirmed presence.
    DwellStarted {
        /// Track identifier.
        track_id: TrackId,
        /// Zone identifier.
        zone_id: ZoneId,
        /// Media time when dwell began.
        at: MediaTime,
    },
    /// Dwell timing ended; includes total dwell nanoseconds.
    DwellEnded {
        /// Track identifier.
        track_id: TrackId,
        /// Zone identifier.
        zone_id: ZoneId,
        /// Total dwell duration in nanoseconds.
        duration_ns: i64,
        /// Visit count for this track in this zone after this visit.
        visit_count: u32,
    },
    /// Occupancy changed inside the zone.
    OccupancyChanged {
        /// Zone identifier.
        zone_id: ZoneId,
        /// Current number of confirmed occupants.
        occupancy: u32,
    },
}
