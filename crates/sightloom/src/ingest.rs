//! Production-oriented streaming lifecycle contracts for host ingest.
//!
//! `SightLoom` is not a full media broker (no Savant-scale dynamic sources /
//! OpenTelemetry stack here). Facades and hosts still need explicit policy for:
//! bounded queues, drop/late/out-of-order frames, watermarks, reset, metrics.

use sightloom_core::{FrameStamp, MediaTime, SourceId};

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
