//! Per-track trajectory history and motion derivatives.

use crate::SmoothError;
use sightloom_core::{Point, Rect, TrackId};

/// One historical sample of a track box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrajectorySample {
    /// Frame index for the sample.
    pub frame_index: u64,
    /// Bounding box at this sample.
    pub bbox: Rect,
    /// Detection confidence at this sample.
    pub confidence: f32,
}

/// Fixed-capacity ring buffer of trajectory samples for one track.
#[derive(Clone, Debug)]
pub struct TrajectoryHistory<const N: usize> {
    track_id: TrackId,
    samples: [Option<TrajectorySample>; N],
    /// Next write index in the ring.
    head: usize,
    /// Number of valid samples currently stored.
    len: usize,
}

impl<const N: usize> TrajectoryHistory<N> {
    /// Creates an empty history for `track_id`.
    #[must_use]
    pub const fn new(track_id: TrackId) -> Self {
        Self {
            track_id,
            samples: [None; N],
            head: 0,
            len: 0,
        }
    }

    /// Returns the track this history belongs to.
    #[must_use]
    pub const fn track_id(&self) -> TrackId {
        self.track_id
    }

    /// Number of stored samples.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the history is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Pushes a sample, overwriting the oldest when full.
    ///
    /// # Errors
    ///
    /// Returns [`SmoothError::InvalidConfig`] when `N == 0`.
    pub fn push(&mut self, sample: TrajectorySample) -> Result<(), SmoothError> {
        if N == 0 {
            return Err(SmoothError::InvalidConfig);
        }
        self.samples[self.head] = Some(sample);
        self.head = (self.head + 1) % N;
        if self.len < N {
            self.len += 1;
        }
        Ok(())
    }

    /// Copies samples in chronological order into `output`.
    ///
    /// Returns the number written.
    ///
    /// # Errors
    ///
    /// Returns [`SmoothError::InsufficientCapacity`] when `output` is shorter
    /// than the stored length, or when a ring slot is unexpectedly empty.
    pub fn copy_chronological(
        &self,
        output: &mut [TrajectorySample],
    ) -> Result<usize, SmoothError> {
        if output.len() < self.len {
            return Err(SmoothError::InsufficientCapacity);
        }
        let start = if self.len < N { 0 } else { self.head };
        for (offset, slot) in output.iter_mut().enumerate().take(self.len) {
            let index = (start + offset) % N;
            let Some(sample) = self.samples[index] else {
                return Err(SmoothError::InsufficientCapacity);
            };
            *slot = sample;
        }
        Ok(self.len)
    }

    /// Velocity of the box center between the last two samples (pixels/frame).
    #[must_use]
    pub fn velocity(&self) -> Option<Point> {
        if self.len < 2 {
            return None;
        }
        let last = self.nth_from_end(0)?;
        let prev = self.nth_from_end(1)?;
        let df = last.frame_index.saturating_sub(prev.frame_index).max(1) as f32;
        let c0 = prev.bbox.center();
        let c1 = last.bbox.center();
        Point::new((c1.x() - c0.x()) / df, (c1.y() - c0.y()) / df).ok()
    }

    /// Acceleration from the last three samples (pixels/frame²).
    #[must_use]
    pub fn acceleration(&self) -> Option<Point> {
        if self.len < 3 {
            return None;
        }
        let s0 = self.nth_from_end(2)?;
        let s1 = self.nth_from_end(1)?;
        let s2 = self.nth_from_end(0)?;
        let d01 = s1.frame_index.saturating_sub(s0.frame_index).max(1) as f32;
        let d12 = s2.frame_index.saturating_sub(s1.frame_index).max(1) as f32;
        let v01x = (s1.bbox.center().x() - s0.bbox.center().x()) / d01;
        let v01y = (s1.bbox.center().y() - s0.bbox.center().y()) / d01;
        let v12x = (s2.bbox.center().x() - s1.bbox.center().x()) / d12;
        let v12y = (s2.bbox.center().y() - s1.bbox.center().y()) / d12;
        let dt = ((d01 + d12) * 0.5).max(1.0);
        Point::new((v12x - v01x) / dt, (v12y - v01y) / dt).ok()
    }

    /// Mean center displacement magnitude between consecutive samples.
    #[must_use]
    pub fn jitter(&self) -> f32 {
        if self.len < 2 {
            return 0.0;
        }
        let mut total = 0.0_f32;
        let mut count = 0_u32;
        for offset in 0..(self.len - 1) {
            let a = self.nth_from_end(self.len - 1 - offset);
            let b = self.nth_from_end(self.len - 2 - offset);
            if let (Some(a), Some(b)) = (a, b) {
                let dx = a.bbox.center().x() - b.bbox.center().x();
                let dy = a.bbox.center().y() - b.bbox.center().y();
                // L1 displacement keeps the metric allocation-free and no_std-safe.
                let ax = if dx < 0.0 { -dx } else { dx };
                let ay = if dy < 0.0 { -dy } else { dy };
                total += ax + ay;
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            total / count as f32
        }
    }

    fn nth_from_end(&self, n: usize) -> Option<TrajectorySample> {
        if n >= self.len {
            return None;
        }
        let start = if self.len < N { 0 } else { self.head };
        let index = (start + self.len - 1 - n) % N;
        self.samples[index]
    }
}
