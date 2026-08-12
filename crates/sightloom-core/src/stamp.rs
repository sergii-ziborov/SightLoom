//! Frame timing and multi-source stamps.

use crate::{CoreError, SourceId};

/// Presentation timestamp as rational media time.
///
/// Values are stored as `ticks / timescale` seconds. Both fields must be
/// finite-compatible integers; `timescale` must be non-zero.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MediaTime {
    ticks: i64,
    timescale: u32,
}

impl Default for MediaTime {
    fn default() -> Self {
        Self {
            ticks: 0,
            timescale: 1,
        }
    }
}

impl MediaTime {
    /// Creates a media time when `timescale` is non-zero.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidMediaTime`] when `timescale` is zero.
    pub const fn new(ticks: i64, timescale: u32) -> Result<Self, CoreError> {
        if timescale == 0 {
            return Err(CoreError::InvalidMediaTime);
        }
        Ok(Self { ticks, timescale })
    }

    /// Returns the tick count.
    #[must_use]
    pub const fn ticks(self) -> i64 {
        self.ticks
    }

    /// Returns the timescale (ticks per second).
    #[must_use]
    pub const fn timescale(self) -> u32 {
        self.timescale
    }

    /// Converts this media time to whole nanoseconds, saturating on overflow.
    #[must_use]
    pub fn as_nanos(self) -> i64 {
        let ticks = i128::from(self.ticks);
        let scale = i128::from(self.timescale);
        let nanos = ticks
            .saturating_mul(1_000_000_000)
            .checked_div(scale)
            .unwrap_or(0);
        i64::try_from(nanos).unwrap_or(if nanos.is_positive() {
            i64::MAX
        } else {
            i64::MIN
        })
    }

    /// Signed duration between `self` and `earlier` in nanoseconds.
    #[must_use]
    pub fn duration_since_ns(self, earlier: Self) -> i64 {
        self.as_nanos().saturating_sub(earlier.as_nanos())
    }
}

/// Temporal and source identity of a single frame sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameStamp {
    /// Media source that produced the frame.
    pub source_id: SourceId,
    /// Zero-based frame index within the source timeline.
    pub frame_index: u64,
    /// Presentation timestamp for the frame.
    pub pts: MediaTime,
    /// Optional host wall-clock capture time in nanoseconds since the Unix epoch.
    pub wall_clock_ns: Option<i64>,
}

impl FrameStamp {
    /// Creates a frame stamp from media time components.
    #[must_use]
    pub fn new(
        source_id: SourceId,
        frame_index: u64,
        pts: MediaTime,
        wall_clock_ns: Option<i64>,
    ) -> Self {
        Self {
            source_id,
            frame_index,
            pts,
            wall_clock_ns,
        }
    }
}
