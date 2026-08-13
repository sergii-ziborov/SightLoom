//! Bounded frame queue + soft batch / drain ingest.

use sightloom::{
    DropPolicy, FrameQueue, IndexSession, IngestPolicy, LateFramePolicy, OutOfOrderPolicy,
    QueuePushResult,
};
use sightloom_core::{Detection, FrameStamp, MediaTime, Rect, SourceId};
use sightloom_index::SourceEntry;
use sightloom_tracking::ByteTrackConfig;

fn track_config() -> ByteTrackConfig {
    ByteTrackConfig {
        track_high_thresh: 0.5,
        track_activation_thresh: 0.5,
        track_low_thresh: 0.1,
        match_thresh: 0.3,
        max_time_lost: 30,
        class_aware: false,
    }
}

fn det(x: f32) -> Detection {
    Detection::new(Rect::new(x, 0.0, x + 10.0, 20.0).unwrap(), 0.9, None, None).unwrap()
}

#[test]
fn frame_queue_drop_oldest_and_drain() {
    let mut queue = FrameQueue::new(2, DropPolicy::DropOldest);
    assert_eq!(
        queue.push(
            FrameStamp::new(SourceId(1), 0, MediaTime::new(0, 30).unwrap(), None),
            vec![det(0.0)]
        ),
        QueuePushResult::Enqueued
    );
    assert_eq!(
        queue.push(
            FrameStamp::new(SourceId(1), 1, MediaTime::new(1, 30).unwrap(), None),
            vec![det(1.0)]
        ),
        QueuePushResult::Enqueued
    );
    assert_eq!(
        queue.push(
            FrameStamp::new(SourceId(1), 2, MediaTime::new(2, 30).unwrap(), None),
            vec![det(2.0)]
        ),
        QueuePushResult::DroppedOldest
    );
    assert_eq!(queue.len(), 2);
    assert_eq!(queue.dropped(), 1);
    assert_eq!(queue.front().unwrap().stamp.frame_index, 1);

    let mut session = IndexSession::new("q", track_config()).unwrap();
    session.add_source(SourceEntry {
        source_id: 1,
        uri: "file://a.mp4".into(),
        hash: None,
    });
    let tracked = session.drain_frame_queue(&mut queue, None).unwrap();
    assert_eq!(tracked.len(), 2);
    assert!(queue.is_empty());
    assert!(session.ingest_metrics().queue_hwm >= 2);
}

#[test]
fn soft_batch_skips_late_frames() {
    let mut session = IndexSession::new("soft", track_config()).unwrap();
    session.add_source(SourceEntry {
        source_id: 1,
        uri: "file://a.mp4".into(),
        hash: None,
    });
    session.set_ingest_policy(IngestPolicy {
        max_queue_depth: 8,
        drop_policy: DropPolicy::RejectNew,
        late_frame_policy: LateFramePolicy::Reject,
        out_of_order_policy: OutOfOrderPolicy::Reject,
        max_lateness_ns: 0,
    });

    let frames = vec![
        (
            FrameStamp::new(SourceId(1), 0, MediaTime::new(10, 30).unwrap(), None),
            vec![det(0.0)],
        ),
        (
            // Late / older pts than watermark after first accept.
            FrameStamp::new(SourceId(1), 1, MediaTime::new(5, 30).unwrap(), None),
            vec![det(1.0)],
        ),
        (
            FrameStamp::new(SourceId(1), 2, MediaTime::new(11, 30).unwrap(), None),
            vec![det(2.0)],
        ),
    ];
    let accepted = session.ingest_detection_batch_soft(&frames).unwrap();
    assert_eq!(accepted.len(), 2);
    assert!(session.ingest_metrics().rejected_late >= 1);
}
