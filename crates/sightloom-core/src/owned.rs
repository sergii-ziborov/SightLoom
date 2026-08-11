//! Allocation-backed detection batch conveniences.

use alloc::{vec, vec::Vec};

use crate::{CoreError, Detection, NmsConfig, nms_in_place};

/// A heap-backed batch of validated detections.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OwnedDetectionBatch {
    detections: Vec<Detection>,
}

impl OwnedDetectionBatch {
    /// Creates an empty owned detection batch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            detections: Vec::new(),
        }
    }

    /// Appends a detection, growing the backing allocation when necessary.
    pub fn push(&mut self, detection: Detection) {
        self.detections.push(detection);
    }

    /// Returns the detections currently in the batch.
    #[must_use]
    pub fn as_slice(&self) -> &[Detection] {
        &self.detections
    }

    /// Applies non-maximum suppression and removes suppressed detections.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`nms_in_place`] without modifying this batch.
    pub fn nms(&mut self, config: NmsConfig) -> Result<usize, CoreError> {
        let len = self.detections.len();
        let mut order = vec![0; len];
        let mut suppressed = vec![false; len];
        let kept = nms_in_place(
            self.detections.as_mut_slice(),
            order.as_mut_slice(),
            suppressed.as_mut_slice(),
            config,
        )?;
        self.detections.truncate(kept);
        Ok(kept)
    }
}
