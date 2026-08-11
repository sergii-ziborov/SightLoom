//! Fixed-capacity finite-line crossing monitor.

use crate::{
    CoreError, Direction, LineSegment, LineSide, Point, TrackId, VisionEvent, ZoneId,
    crosses_segment, line_side,
};

use super::slots::TrackSlot;

#[derive(Clone, Copy)]
struct LineTrackState {
    previous: Point,
    last_non_on: Option<LineSide>,
}

/// Tracks crossings of a finite directed line segment for at most `N` tracks.
pub struct LineZoneMonitor<const N: usize> {
    zone_id: ZoneId,
    segment: LineSegment,
    slots: [TrackSlot<LineTrackState>; N],
}

impl<const N: usize> LineZoneMonitor<N> {
    /// Creates a monitor for `segment` using caller-selected fixed track capacity.
    #[must_use]
    pub const fn new(zone_id: ZoneId, segment: LineSegment) -> Self {
        Self {
            zone_id,
            segment,
            slots: [TrackSlot::Empty; N],
        }
    }

    /// Records a point sample and writes its crossing event to `output[0]` when needed.
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
        let (existing, free) = TrackSlot::find_slot(&self.slots, track_id);
        let Some(index) = existing.or(free) else {
            return Err(CoreError::InsufficientCapacity);
        };
        let side = line_side(self.segment, point);
        let (next, event) = match self.slots[index] {
            TrackSlot::Empty => (
                TrackSlot::Occupied {
                    track_id,
                    state: LineTrackState {
                        previous: point,
                        last_non_on: non_on_side(side),
                    },
                },
                None,
            ),
            TrackSlot::Occupied {
                state:
                    LineTrackState {
                        previous,
                        last_non_on,
                    },
                ..
            } => {
                let event = crossing_event(
                    self.zone_id,
                    self.segment,
                    track_id,
                    previous,
                    last_non_on,
                    point,
                    side,
                );
                (
                    TrackSlot::Occupied {
                        track_id,
                        state: LineTrackState {
                            previous: point,
                            last_non_on: non_on_side(side).or(last_non_on),
                        },
                    },
                    event,
                )
            }
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

    /// Removes a track's stored crossing state without emitting an event.
    pub fn forget_track(&mut self, track_id: TrackId) -> bool {
        TrackSlot::forget_track(&mut self.slots, track_id)
    }
}

fn non_on_side(side: LineSide) -> Option<LineSide> {
    (side != LineSide::On).then_some(side)
}

#[allow(clippy::too_many_arguments)]
fn crossing_event(
    zone_id: ZoneId,
    segment: LineSegment,
    track_id: TrackId,
    previous: Point,
    last_non_on: Option<LineSide>,
    point: Point,
    side: LineSide,
) -> Option<VisionEvent> {
    let previous_side = last_non_on?;
    if side == LineSide::On || side == previous_side || previous == point {
        return None;
    }

    let motion = LineSegment::new(previous, point).ok()?;
    if !crosses_segment(segment, motion) {
        return None;
    }

    let direction = match (previous_side, side) {
        (LineSide::Left, LineSide::Right) => Direction::LeftToRight,
        (LineSide::Right, LineSide::Left) => Direction::RightToLeft,
        _ => return None,
    };
    Some(VisionEvent::Crossed {
        track_id,
        zone_id,
        direction,
    })
}
