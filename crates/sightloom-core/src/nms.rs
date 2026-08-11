//! Deterministic non-maximum suppression over caller-owned storage.

use core::cmp::Ordering;

use crate::{CoreError, Detection, ios, iou};

/// The rectangle-overlap metric used to suppress detections.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OverlapMetric {
    /// Intersection over union.
    IoU,
    /// Intersection over the smaller rectangle's area.
    IoS,
}

/// The class-comparison policy used during suppression.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NmsMode {
    /// Compare detections only when their optional class identifiers match.
    ClassAware,
    /// Compare every detection regardless of class identifier.
    ClassAgnostic,
}

/// Immutable settings for a non-maximum suppression pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NmsConfig {
    /// Overlap threshold in the inclusive range `0.0..=1.0`; suppression
    /// occurs only when overlap is strictly greater than this value.
    pub threshold: f32,
    /// Whether class identifiers restrict suppression.
    pub mode: NmsMode,
    /// The overlap metric used for suppression.
    pub metric: OverlapMetric,
}

/// Suppresses overlapping detections without allocating.
///
/// Detections are prioritized by descending score, then ascending original
/// index. The kept detections are compacted into the front of `detections` in
/// their original input order; the remaining tail is unspecified.
///
/// # Errors
///
/// Returns [`CoreError::InvalidThreshold`] when `config.threshold` is not
/// finite or is outside `0.0..=1.0`. Returns
/// [`CoreError::InsufficientScratch`] when either scratch slice is shorter
/// than `detections`. Neither error mutates a caller-owned slice.
pub fn nms_in_place(
    detections: &mut [Detection],
    order_scratch: &mut [usize],
    suppressed_scratch: &mut [bool],
    config: NmsConfig,
) -> Result<usize, CoreError> {
    if !config.threshold.is_finite() || !(0.0..=1.0).contains(&config.threshold) {
        return Err(CoreError::InvalidThreshold);
    }

    let len = detections.len();
    if order_scratch.len() < len || suppressed_scratch.len() < len {
        return Err(CoreError::InsufficientScratch);
    }

    let order = &mut order_scratch[..len];
    let suppressed = &mut suppressed_scratch[..len];
    for (index, slot) in order.iter_mut().enumerate() {
        *slot = index;
    }
    for value in suppressed.iter_mut() {
        *value = false;
    }

    order.sort_unstable_by(|left, right| {
        detections[*right]
            .score()
            .partial_cmp(&detections[*left].score())
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(right))
    });

    for candidate_position in 0..len {
        let candidate_index = order[candidate_position];
        if suppressed[candidate_index] {
            continue;
        }

        for contender_index in order.iter().skip(candidate_position + 1).copied() {
            if suppressed[contender_index]
                || (config.mode == NmsMode::ClassAware
                    && detections[candidate_index].class_id()
                        != detections[contender_index].class_id())
            {
                continue;
            }

            let overlap = match config.metric {
                OverlapMetric::IoU => iou(
                    detections[candidate_index].bbox(),
                    detections[contender_index].bbox(),
                ),
                OverlapMetric::IoS => ios(
                    detections[candidate_index].bbox(),
                    detections[contender_index].bbox(),
                ),
            };
            if overlap > config.threshold {
                suppressed[contender_index] = true;
            }
        }
    }

    let mut kept = 0;
    for index in 0..len {
        if !suppressed[index] {
            let detection = detections[index];
            detections[kept] = detection;
            kept += 1;
        }
    }

    Ok(kept)
}
