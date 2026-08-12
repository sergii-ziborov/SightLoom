//! Convert analytics events into portable [`EventEnvelope`] records.

use crate::AnalyticsEvent;
use sightloom_core::{
    EventEnvelope, EventId, EventKind, EventPayload, FrameStamp, SubjectId, TrackId,
};

/// Builds an [`EventEnvelope`] from an analytics event.
///
/// `stamp` should match the frame that produced the analytics update. When the
/// payload carries its own media time (dwell), the envelope stamp is still the
/// frame stamp used by the caller for index ordering.
#[must_use]
pub fn analytics_to_envelope(
    event_id: EventId,
    stamp: FrameStamp,
    event: AnalyticsEvent,
    subject_id: Option<SubjectId>,
) -> EventEnvelope {
    let mut envelope = EventEnvelope::new(event_id, stamp, EventKind::Zone);
    match event {
        AnalyticsEvent::Entered {
            track_id,
            zone_id,
            class_id,
        } => {
            envelope = envelope
                .with_track(track_id)
                .with_zone(zone_id)
                .with_payload(EventPayload::Entered { zone_id, class_id });
            envelope.kind = EventKind::Zone;
        }
        AnalyticsEvent::Exited {
            track_id,
            zone_id,
            class_id,
        } => {
            envelope = envelope
                .with_track(track_id)
                .with_zone(zone_id)
                .with_payload(EventPayload::Exited { zone_id, class_id });
            envelope.kind = EventKind::Zone;
        }
        AnalyticsEvent::DwellStarted {
            track_id,
            zone_id,
            at: _,
        } => {
            envelope = envelope
                .with_track(track_id)
                .with_zone(zone_id)
                .with_payload(EventPayload::DwellStarted { zone_id });
            envelope.kind = EventKind::Dwell;
        }
        AnalyticsEvent::DwellEnded {
            track_id,
            zone_id,
            duration_ns,
            visit_count,
        } => {
            envelope = envelope
                .with_track(track_id)
                .with_zone(zone_id)
                .with_payload(EventPayload::DwellEnded {
                    zone_id,
                    duration_ns,
                    visit_count,
                });
            envelope.kind = EventKind::Dwell;
        }
        AnalyticsEvent::OccupancyChanged { zone_id, occupancy } => {
            envelope = envelope
                .with_zone(zone_id)
                .with_payload(EventPayload::Occupancy { zone_id, occupancy });
            envelope.kind = EventKind::Occupancy;
        }
    }
    if let Some(subject_id) = subject_id {
        envelope = envelope.with_subject(subject_id);
    }
    envelope
}

/// Convenience for envelopes that already know their track id from the event.
#[must_use]
pub fn track_of(event: AnalyticsEvent) -> Option<TrackId> {
    match event {
        AnalyticsEvent::Entered { track_id, .. }
        | AnalyticsEvent::Exited { track_id, .. }
        | AnalyticsEvent::DwellStarted { track_id, .. }
        | AnalyticsEvent::DwellEnded { track_id, .. } => Some(track_id),
        AnalyticsEvent::OccupancyChanged { .. } => None,
    }
}
