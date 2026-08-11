//! Fixed-capacity polygon membership monitor.

use crate::{CoreError, Point, Polygon, TrackId, VisionEvent, ZoneId};

#[derive(Clone, Copy)]
enum TrackSlot {
    Empty,
    Occupied { track_id: TrackId, inside: bool },
}

/// Tracks polygon membership transitions for at most `N` tracks.
pub struct PolygonZoneMonitor<'a, const N: usize> {
    zone_id: ZoneId,
    polygon: Polygon<'a>,
    slots: [TrackSlot; N],
}

impl<'a, const N: usize> PolygonZoneMonitor<'a, N> {
    /// Creates a monitor for `polygon` using caller-selected fixed track capacity.
    #[must_use]
    pub const fn new(zone_id: ZoneId, polygon: Polygon<'a>) -> Self {
        Self {
            zone_id,
            polygon,
            slots: [TrackSlot::Empty; N],
        }
    }

    /// Records a point sample and writes its membership event to `output[0]` when needed.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InsufficientCapacity`] when no track slot or required
    /// event-output slot is available. On error, the monitor state is unchanged.
    pub fn update(
        &mut self,
        track_id: TrackId,
        point: Point,
        output: &mut [VisionEvent],
    ) -> Result<usize, CoreError> {
        let (existing, free) = self.find_slot(track_id);
        let Some(index) = existing.or(free) else {
            return Err(CoreError::InsufficientCapacity);
        };
        let inside = self.polygon.contains(point);
        let (next, event) = match self.slots[index] {
            TrackSlot::Empty => (
                TrackSlot::Occupied { track_id, inside },
                membership_event(track_id, self.zone_id, false, inside),
            ),
            TrackSlot::Occupied {
                inside: previous_inside,
                ..
            } => (
                TrackSlot::Occupied { track_id, inside },
                membership_event(track_id, self.zone_id, previous_inside, inside),
            ),
        };

        if event.is_some() && output.is_empty() {
            return Err(CoreError::InsufficientCapacity);
        }

        self.slots[index] = next;
        if let Some(event) = event {
            output[0] = event;
            Ok(1)
        } else {
            Ok(0)
        }
    }

    /// Removes a track's stored membership state without emitting an event.
    pub fn forget_track(&mut self, track_id: TrackId) -> bool {
        for slot in &mut self.slots {
            match *slot {
                TrackSlot::Occupied {
                    track_id: stored_id,
                    ..
                } if stored_id == track_id => {
                    *slot = TrackSlot::Empty;
                    return true;
                }
                TrackSlot::Empty | TrackSlot::Occupied { .. } => {}
            }
        }
        false
    }

    fn find_slot(&self, track_id: TrackId) -> (Option<usize>, Option<usize>) {
        let mut free = None;
        for (index, slot) in self.slots.iter().enumerate() {
            match *slot {
                TrackSlot::Occupied {
                    track_id: stored_id,
                    ..
                } if stored_id == track_id => return (Some(index), free),
                TrackSlot::Empty if free.is_none() => free = Some(index),
                TrackSlot::Empty | TrackSlot::Occupied { .. } => {}
            }
        }
        (None, free)
    }
}

fn membership_event(
    track_id: TrackId,
    zone_id: ZoneId,
    was_inside: bool,
    is_inside: bool,
) -> Option<VisionEvent> {
    match (was_inside, is_inside) {
        (false, true) => Some(VisionEvent::Entered { track_id, zone_id }),
        (true, false) => Some(VisionEvent::Exited { track_id, zone_id }),
        (false, false) | (true, true) => None,
    }
}
