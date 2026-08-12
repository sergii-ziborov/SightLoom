//! Event envelope contract tests.

use sightloom_core::{
    EventEnvelope, EventId, EventKind, EventPayload, FrameStamp, MediaTime, SourceId, TrackId,
    ZoneId,
};

#[test]
fn envelope_builders_preserve_associations() {
    let stamp = FrameStamp::new(SourceId(2), 7, MediaTime::new(7, 30).unwrap(), None);
    let event = EventEnvelope::new(EventId(1), stamp, EventKind::Zone)
        .with_track(TrackId(9))
        .with_zone(ZoneId(3))
        .with_payload(EventPayload::Entered {
            zone_id: ZoneId(3),
            class_id: None,
        });

    assert_eq!(event.event_id, EventId(1));
    assert_eq!(event.track_id, Some(TrackId(9)));
    assert_eq!(event.zone_id, Some(ZoneId(3)));
    assert_eq!(event.kind, EventKind::Zone);
    assert!(matches!(
        event.payload,
        EventPayload::Entered {
            zone_id: ZoneId(3),
            class_id: None
        }
    ));
}
