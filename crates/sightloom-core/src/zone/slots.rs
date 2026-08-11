//! Shared fixed-capacity track-slot lifecycle.

use crate::TrackId;

#[derive(Clone, Copy)]
pub(crate) enum TrackSlot<State> {
    Empty,
    Occupied { track_id: TrackId, state: State },
}

impl<State: Copy> TrackSlot<State> {
    pub(crate) fn find_slot(slots: &[Self], track_id: TrackId) -> (Option<usize>, Option<usize>) {
        let mut free = None;
        for (index, slot) in slots.iter().enumerate() {
            match *slot {
                Self::Occupied {
                    track_id: stored_id,
                    ..
                } if stored_id == track_id => return (Some(index), free),
                Self::Empty if free.is_none() => free = Some(index),
                Self::Empty | Self::Occupied { .. } => {}
            }
        }
        (None, free)
    }

    pub(crate) fn forget_track(slots: &mut [Self], track_id: TrackId) -> bool {
        for slot in slots {
            match *slot {
                Self::Occupied {
                    track_id: stored_id,
                    ..
                } if stored_id == track_id => {
                    *slot = Self::Empty;
                    return true;
                }
                Self::Empty | Self::Occupied { .. } => {}
            }
        }
        false
    }
}
