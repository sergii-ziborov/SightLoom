//! Lightweight in-memory event/subject index (SQLite-shaped API).

use sightloom_core::{SubjectId, TrackId, ZoneId};

/// One indexed analytics/event row.
#[derive(Clone, Debug, PartialEq)]
pub struct EventRecord {
    /// Monotonic event id within the package.
    pub event_id: u64,
    /// Event kind tag (`entered`, `exited`, `dwell_ended`, ...).
    pub kind: String,
    /// Optional track.
    pub track_id: Option<TrackId>,
    /// Optional subject.
    pub subject_id: Option<SubjectId>,
    /// Optional zone.
    pub zone_id: Option<ZoneId>,
    /// Event time in nanoseconds.
    pub time_ns: i64,
    /// Optional duration payload (dwell).
    pub duration_ns: Option<i64>,
}

/// In-memory stand-in for the `SQLite` event/subject index.
///
/// On-disk `SQLite` binding can replace the storage without changing the record
/// shape.
#[cfg(feature = "std")]
#[derive(Clone, Debug, Default)]
pub struct EventIndex {
    next_id: u64,
    events: Vec<EventRecord>,
}

#[cfg(feature = "std")]
impl EventIndex {
    /// Creates an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: 1,
            events: Vec::new(),
        }
    }

    /// Inserts an event and assigns an id.
    pub fn insert(
        &mut self,
        kind: impl Into<String>,
        track_id: Option<TrackId>,
        subject_id: Option<SubjectId>,
        zone_id: Option<ZoneId>,
        time_ns: i64,
        duration_ns: Option<i64>,
    ) -> EventRecord {
        let record = EventRecord {
            event_id: self.next_id,
            kind: kind.into(),
            track_id,
            subject_id,
            zone_id,
            time_ns,
            duration_ns,
        };
        self.next_id = self.next_id.saturating_add(1);
        self.events.push(record.clone());
        record
    }

    /// Returns all events.
    #[must_use]
    pub fn events(&self) -> &[EventRecord] {
        &self.events
    }

    /// Events for a subject.
    #[must_use]
    pub fn by_subject(&self, subject_id: SubjectId) -> Vec<&EventRecord> {
        self.events
            .iter()
            .filter(|event| event.subject_id == Some(subject_id))
            .collect()
    }

    /// Events for a track.
    #[must_use]
    pub fn by_track(&self, track_id: TrackId) -> Vec<&EventRecord> {
        self.events
            .iter()
            .filter(|event| event.track_id == Some(track_id))
            .collect()
    }
}
