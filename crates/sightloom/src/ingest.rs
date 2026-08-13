//! Production-oriented streaming lifecycle contracts for host ingest.
//!
//! `SightLoom` is not a full media broker (no Savant-scale dynamic sources /
//! OpenTelemetry stack here). Facades and hosts still need explicit policy for:
//! bounded queues, drop/late/out-of-order frames, watermarks, reset, metrics.

use std::collections::VecDeque;

use sightloom_core::{Detection, FrameStamp, MediaTime, SourceId};

/// What to do when the ingest queue is full.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DropPolicy {
    /// Reject the new frame with an error.
    #[default]
    RejectNew,
    /// Drop the oldest queued frame.
    DropOldest,
    /// Drop the newest frame silently (count as dropped).
    DropNewest,
}

/// How to treat frames older than the source watermark.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LateFramePolicy {
    /// Reject late frames.
    #[default]
    Reject,
    /// Accept but mark metrics.
    AcceptMark,
    /// Drop late frames.
    Drop,
}

/// Out-of-order timestamp policy within a source.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OutOfOrderPolicy {
    /// Reject frames with pts &lt; last pts.
    #[default]
    Reject,
    /// Accept and rely on host reordering / revision.
    Accept,
    /// Drop out-of-order frames.
    Drop,
}

/// Host ingest policy for a session or source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IngestPolicy {
    /// Maximum pending frames in a host-side queue (`0` = unlimited).
    pub max_queue_depth: usize,
    /// Full-queue behavior.
    pub drop_policy: DropPolicy,
    /// Late frame behavior vs watermark.
    pub late_frame_policy: LateFramePolicy,
    /// Out-of-order pts behavior.
    pub out_of_order_policy: OutOfOrderPolicy,
    /// Allowed lateness behind watermark (nanoseconds).
    pub max_lateness_ns: i64,
}

impl Default for IngestPolicy {
    fn default() -> Self {
        Self {
            max_queue_depth: 64,
            drop_policy: DropPolicy::RejectNew,
            late_frame_policy: LateFramePolicy::Reject,
            out_of_order_policy: OutOfOrderPolicy::Reject,
            max_lateness_ns: 0,
        }
    }
}

/// Per-source checkpoint watermark.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SourceWatermark {
    /// Source.
    pub source_id: SourceId,
    /// Highest accepted presentation time.
    pub high_pts: Option<MediaTime>,
    /// Highest accepted frame index.
    pub high_frame_index: u64,
    /// Monotonic ingest sequence for this source.
    pub sequence: u64,
}

impl SourceWatermark {
    /// Empty watermark for a source.
    #[must_use]
    pub const fn new(source_id: SourceId) -> Self {
        Self {
            source_id,
            high_pts: None,
            high_frame_index: 0,
            sequence: 0,
        }
    }

    /// Advances after a successfully accepted frame.
    pub fn advance(&mut self, stamp: FrameStamp) {
        self.high_pts = Some(stamp.pts);
        self.high_frame_index = self.high_frame_index.max(stamp.frame_index);
        self.sequence = self.sequence.saturating_add(1);
    }
}

/// Decision for a candidate frame under policy + watermark.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IngestDecision {
    /// Accept into the tracker/index.
    Accept,
    /// Drop without error (counted).
    Drop,
    /// Reject with a soft error to the host.
    RejectLate,
    /// Reject out-of-order pts.
    RejectOutOfOrder,
}

/// Evaluates late / out-of-order policy for one stamp.
#[must_use]
pub fn evaluate_stamp(
    policy: &IngestPolicy,
    watermark: &SourceWatermark,
    stamp: FrameStamp,
) -> IngestDecision {
    if let Some(high) = watermark.high_pts {
        let delta = high.as_nanos().saturating_sub(stamp.pts.as_nanos());
        if delta > policy.max_lateness_ns && stamp.pts.as_nanos() < high.as_nanos() {
            return match policy.late_frame_policy {
                LateFramePolicy::Reject => IngestDecision::RejectLate,
                LateFramePolicy::Drop => IngestDecision::Drop,
                LateFramePolicy::AcceptMark => IngestDecision::Accept,
            };
        }
        if stamp.pts.as_nanos() < high.as_nanos() {
            return match policy.out_of_order_policy {
                OutOfOrderPolicy::Reject => IngestDecision::RejectOutOfOrder,
                OutOfOrderPolicy::Drop => IngestDecision::Drop,
                OutOfOrderPolicy::Accept => IngestDecision::Accept,
            };
        }
    }
    IngestDecision::Accept
}

/// Lightweight counters hosts can scrape or export to Prometheus.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IngestMetrics {
    /// Frames accepted.
    pub accepted: u64,
    /// Frames dropped by policy.
    pub dropped: u64,
    /// Frames rejected as late.
    pub rejected_late: u64,
    /// Frames rejected as out-of-order.
    pub rejected_ooo: u64,
    /// Queue depth high-water mark (host-maintained).
    pub queue_hwm: u64,
    /// Source resets observed.
    pub source_resets: u64,
    /// Checkpoint saves.
    pub checkpoints: u64,
}

impl IngestMetrics {
    /// Records an ingest decision.
    pub fn record(&mut self, decision: IngestDecision) {
        match decision {
            IngestDecision::Accept => self.accepted = self.accepted.saturating_add(1),
            IngestDecision::Drop => self.dropped = self.dropped.saturating_add(1),
            IngestDecision::RejectLate => self.rejected_late = self.rejected_late.saturating_add(1),
            IngestDecision::RejectOutOfOrder => {
                self.rejected_ooo = self.rejected_ooo.saturating_add(1);
            }
        }
    }
}

/// One pending frame waiting to be ingested (host-side queue item).
#[derive(Clone, Debug, PartialEq)]
pub struct QueuedFrame {
    /// Frame stamp.
    pub stamp: FrameStamp,
    /// Detections for this frame.
    pub detections: Vec<Detection>,
}

/// Result of trying to push into a bounded [`FrameQueue`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueuePushResult {
    /// Frame stored.
    Enqueued,
    /// New frame rejected (queue full + [`DropPolicy::RejectNew`]).
    RejectedNew,
    /// New frame dropped (queue full + [`DropPolicy::DropNewest`]).
    DroppedNewest,
    /// Oldest frame was dropped to make room ([`DropPolicy::DropOldest`]).
    DroppedOldest,
}

/// Bounded per-host (or per-source) frame queue implementing [`IngestPolicy`]
/// depth + [`DropPolicy`].
///
/// `SightLoom` does not own a media thread; hosts push frames here, then pop
/// and call [`crate::IndexSession::ingest_detections`].
#[derive(Clone, Debug, Default)]
pub struct FrameQueue {
    items: VecDeque<QueuedFrame>,
    max_depth: usize,
    drop_policy: DropPolicy,
    /// High-water mark of queue length.
    hwm: usize,
    /// Frames dropped or rejected by the queue itself (not late/OOO).
    dropped: u64,
    rejected_new: u64,
}

impl FrameQueue {
    /// Creates a queue from ingest policy depth/drop settings.
    ///
    /// `max_queue_depth == 0` means unlimited.
    #[must_use]
    pub fn from_policy(policy: &IngestPolicy) -> Self {
        Self {
            items: VecDeque::new(),
            max_depth: policy.max_queue_depth,
            drop_policy: policy.drop_policy,
            hwm: 0,
            dropped: 0,
            rejected_new: 0,
        }
    }

    /// Creates a queue with explicit capacity and drop policy.
    #[must_use]
    pub fn new(max_depth: usize, drop_policy: DropPolicy) -> Self {
        Self {
            items: VecDeque::new(),
            max_depth,
            drop_policy,
            hwm: 0,
            dropped: 0,
            rejected_new: 0,
        }
    }

    /// Current depth.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// True when empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// High-water mark of length.
    #[must_use]
    pub const fn high_water_mark(&self) -> usize {
        self.hwm
    }

    /// Frames dropped under [`DropPolicy::DropOldest`] / [`DropPolicy::DropNewest`].
    #[must_use]
    pub const fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Frames rejected under [`DropPolicy::RejectNew`] when full.
    #[must_use]
    pub const fn rejected_new(&self) -> u64 {
        self.rejected_new
    }

    /// Pushes a frame, applying drop policy when full.
    pub fn push(&mut self, stamp: FrameStamp, detections: Vec<Detection>) -> QueuePushResult {
        let unlimited = self.max_depth == 0;
        if !unlimited && self.items.len() >= self.max_depth {
            match self.drop_policy {
                DropPolicy::RejectNew => {
                    self.rejected_new = self.rejected_new.saturating_add(1);
                    return QueuePushResult::RejectedNew;
                }
                DropPolicy::DropNewest => {
                    self.dropped = self.dropped.saturating_add(1);
                    return QueuePushResult::DroppedNewest;
                }
                DropPolicy::DropOldest => {
                    let _ = self.items.pop_front();
                    self.dropped = self.dropped.saturating_add(1);
                    self.items.push_back(QueuedFrame { stamp, detections });
                    self.hwm = self.hwm.max(self.items.len());
                    return QueuePushResult::DroppedOldest;
                }
            }
        }
        self.items.push_back(QueuedFrame { stamp, detections });
        self.hwm = self.hwm.max(self.items.len());
        QueuePushResult::Enqueued
    }

    /// Pops the oldest frame, if any.
    pub fn pop_front(&mut self) -> Option<QueuedFrame> {
        self.items.pop_front()
    }

    /// Peeks the oldest frame without removing it.
    #[must_use]
    pub fn front(&self) -> Option<&QueuedFrame> {
        self.items.front()
    }

    /// Clears all pending frames.
    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Drains up to `max` frames (or all when `max` is `None`).
    pub fn drain(&mut self, max: Option<usize>) -> Vec<QueuedFrame> {
        let n = max.unwrap_or(self.items.len()).min(self.items.len());
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            if let Some(item) = self.items.pop_front() {
                out.push(item);
            }
        }
        out
    }
}

/// Host-facing source lifecycle event (for adapters / reconnect).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceLifecycle {
    /// Source became available.
    Added {
        /// Source id.
        source_id: SourceId,
    },
    /// Source disconnected; tracker state may be retained or reset.
    Removed {
        /// Source id.
        source_id: SourceId,
        /// When true, clear per-source tracker state.
        reset_tracker: bool,
    },
    /// Source reconnected after a gap.
    Reconnected {
        /// Source id.
        source_id: SourceId,
    },
    /// Explicit reset of watermarks / motion state.
    Reset {
        /// Source id.
        source_id: SourceId,
    },
}
