//! Frame stamp and media-time contracts.

use sightloom_core::{CoreError, FrameStamp, MediaTime, SourceId};

#[test]
fn media_time_rejects_zero_timescale() {
    assert_eq!(MediaTime::new(1, 0), Err(CoreError::InvalidMediaTime));
}

#[test]
fn media_time_nanos_and_duration() {
    let a = MediaTime::new(30, 30).expect("valid");
    let b = MediaTime::new(90, 30).expect("valid");
    assert_eq!(a.as_nanos(), 1_000_000_000);
    assert_eq!(b.duration_since_ns(a), 2_000_000_000);
}

#[test]
fn frame_stamp_carries_source_and_pts() {
    let pts = MediaTime::new(10, 25).expect("valid");
    let stamp = FrameStamp::new(SourceId(3), 42, pts, Some(1_700_000_000_000_000_000));
    assert_eq!(stamp.source_id, SourceId(3));
    assert_eq!(stamp.frame_index, 42);
    assert_eq!(stamp.pts, pts);
    assert_eq!(stamp.wall_clock_ns, Some(1_700_000_000_000_000_000));
}
