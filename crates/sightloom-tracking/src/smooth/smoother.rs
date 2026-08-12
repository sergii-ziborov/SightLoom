//! Exponential detection smoothing and short-gap interpolation.

use super::SmoothError;
use sightloom_core::{Detection, Rect, TrackId};

/// Configuration for [`DetectionSmoother`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmoothConfig {
    /// Exponential weight for the newest sample in `0.0..=1.0`.
    pub alpha: f32,
    /// Maximum consecutive missed frames to interpolate.
    pub max_missed: u32,
}

impl Default for SmoothConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            max_missed: 5,
        }
    }
}

impl SmoothConfig {
    /// Validates alpha and miss budget.
    ///
    /// # Errors
    ///
    /// Returns [`SmoothError::InvalidConfig`] when alpha is out of range.
    pub fn validate(self) -> Result<Self, SmoothError> {
        if !self.alpha.is_finite() || !(0.0..=1.0).contains(&self.alpha) {
            return Err(SmoothError::InvalidConfig);
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug)]
struct Slot {
    track_id: TrackId,
    bbox: Rect,
    score: f32,
    class_id: Option<sightloom_core::ClassId>,
    missed: u32,
    active: bool,
}

/// Fixed-capacity exponential smoother for tracked detections.
pub struct DetectionSmoother<const N: usize> {
    config: SmoothConfig,
    slots: [Option<Slot>; N],
}

impl<const N: usize> DetectionSmoother<N> {
    /// Creates a smoother from a validated configuration.
    ///
    /// # Errors
    ///
    /// Returns config validation errors.
    pub fn new(config: SmoothConfig) -> Result<Self, SmoothError> {
        let config = config.validate()?;
        Ok(Self {
            config,
            slots: [None; N],
        })
    }

    /// Updates the smoother with the current frame's tracked detections.
    ///
    /// Writes smoothed detections (including short interpolations for misses)
    /// into `output` and returns the count written.
    ///
    /// # Errors
    ///
    /// Returns [`SmoothError::InsufficientCapacity`] when output or internal
    /// slots cannot hold the active set.
    pub fn update(
        &mut self,
        detections: &[Detection],
        output: &mut [Detection],
    ) -> Result<usize, SmoothError> {
        // Mark all as potentially missed.
        for slot in &mut self.slots {
            if let Some(entry) = slot.as_mut() {
                entry.active = false;
            }
        }

        for detection in detections {
            let Some(track_id) = detection.track_id() else {
                continue;
            };
            let bbox = detection.bbox();
            if let Some(index) = self.find(track_id) {
                let Some(slot) = self.slots[index].as_mut() else {
                    continue;
                };
                slot.bbox = lerp_rect(slot.bbox, bbox, self.config.alpha)?;
                slot.score = lerp(slot.score, detection.score(), self.config.alpha);
                slot.class_id = detection.class_id().or(slot.class_id);
                slot.missed = 0;
                slot.active = true;
            } else {
                let free = self.free_slot().ok_or(SmoothError::InsufficientCapacity)?;
                self.slots[free] = Some(Slot {
                    track_id,
                    bbox,
                    score: detection.score(),
                    class_id: detection.class_id(),
                    missed: 0,
                    active: true,
                });
            }
        }

        // Age inactive tracks and interpolate briefly.
        for slot in &mut self.slots {
            let Some(entry) = slot.as_mut() else {
                continue;
            };
            if entry.active {
                continue;
            }
            entry.missed = entry.missed.saturating_add(1);
            if entry.missed > self.config.max_missed {
                *slot = None;
            }
        }

        let mut written = 0_usize;
        for slot in &self.slots {
            let Some(entry) = slot else {
                continue;
            };
            if !entry.active && entry.missed == 0 {
                continue;
            }
            if written >= output.len() {
                return Err(SmoothError::InsufficientCapacity);
            }
            output[written] = Detection::new(
                entry.bbox,
                entry.score,
                entry.class_id,
                Some(entry.track_id),
            )
            .map_err(|_| SmoothError::NonFinite)?;
            written += 1;
        }
        Ok(written)
    }

    fn find(&self, track_id: TrackId) -> Option<usize> {
        self.slots.iter().position(|slot| {
            slot.as_ref()
                .is_some_and(|entry| entry.track_id == track_id)
        })
    }

    fn free_slot(&self) -> Option<usize> {
        self.slots.iter().position(Option::is_none)
    }
}

fn lerp(a: f32, b: f32, alpha: f32) -> f32 {
    a * (1.0 - alpha) + b * alpha
}

fn lerp_rect(a: Rect, b: Rect, alpha: f32) -> Result<Rect, SmoothError> {
    Rect::new(
        lerp(a.left(), b.left(), alpha),
        lerp(a.top(), b.top(), alpha),
        lerp(a.right(), b.right(), alpha),
        lerp(a.bottom(), b.bottom(), alpha),
    )
    .map_err(|_| SmoothError::NonFinite)
}

/// Linearly interpolates a box across a gap of `steps` frames.
///
/// `t` is in `0.0..=1.0` from `start` to `end`.
///
/// # Errors
///
/// Returns [`SmoothError::NonFinite`] for invalid rectangles.
pub fn interpolate_bbox(start: Rect, end: Rect, t: f32) -> Result<Rect, SmoothError> {
    if !t.is_finite() {
        return Err(SmoothError::NonFinite);
    }
    let t = t.clamp(0.0, 1.0);
    lerp_rect(start, end, t)
}
