//! Compact public event vocabulary for zone monitors.

use crate::{TrackId, ZoneId};

/// The direction in which a track crossed a directed line segment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Direction {
    /// The track moved from the line's algebraic left half-plane to its right.
    LeftToRight,
    /// The track moved from the line's algebraic right half-plane to its left.
    RightToLeft,
}

/// A membership or crossing transition observed for a track and zone.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionEvent {
    /// A track entered a polygon zone.
    Entered {
        /// The track that entered.
        track_id: TrackId,
        /// The polygon zone that was entered.
        zone_id: ZoneId,
    },
    /// A track exited a polygon zone.
    Exited {
        /// The track that exited.
        track_id: TrackId,
        /// The polygon zone that was exited.
        zone_id: ZoneId,
    },
    /// A track crossed a finite line zone.
    Crossed {
        /// The track that crossed.
        track_id: TrackId,
        /// The line zone that was crossed.
        zone_id: ZoneId,
        /// The crossing direction relative to the directed line segment.
        direction: Direction,
    },
}
