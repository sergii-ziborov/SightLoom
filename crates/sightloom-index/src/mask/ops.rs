//! Mask set operations, intersection-over-union, and mask NMS helpers.

use super::{CroppedMask, DenseMask, MaskError};

/// Intersection-over-union for two dense masks of equal size.
///
/// # Errors
///
/// Returns [`MaskError::LengthMismatch`] when dimensions differ.
pub fn dense_mask_iou(a: DenseMask<'_>, b: DenseMask<'_>) -> Result<f32, MaskError> {
    if a.width() != b.width() || a.height() != b.height() {
        return Err(MaskError::LengthMismatch);
    }
    let mut inter = 0_u32;
    let mut uni = 0_u32;
    for (left, right) in a.data().iter().zip(b.data().iter()) {
        let la = *left != 0;
        let ra = *right != 0;
        if la && ra {
            inter += 1;
        }
        if la || ra {
            uni += 1;
        }
    }
    if uni == 0 {
        return Ok(0.0);
    }
    Ok(inter as f32 / uni as f32)
}

/// Pixel-wise union of two equal-size dense masks into `output`.
///
/// # Errors
///
/// Returns dimension or capacity errors without partial guarantees when sizes
/// mismatch.
pub fn dense_mask_union(
    a: DenseMask<'_>,
    b: DenseMask<'_>,
    output: &mut [u8],
) -> Result<(), MaskError> {
    combine_dense(a, b, output, |left, right| u8::from(left || right))
}

/// Pixel-wise difference `a \ b` into `output`.
///
/// # Errors
///
/// Returns dimension or capacity errors when inputs are incompatible.
pub fn dense_mask_difference(
    a: DenseMask<'_>,
    b: DenseMask<'_>,
    output: &mut [u8],
) -> Result<(), MaskError> {
    combine_dense(a, b, output, |left, right| u8::from(left && !right))
}

fn combine_dense(
    a: DenseMask<'_>,
    b: DenseMask<'_>,
    output: &mut [u8],
    op: impl Fn(bool, bool) -> u8,
) -> Result<(), MaskError> {
    if a.width() != b.width() || a.height() != b.height() {
        return Err(MaskError::LengthMismatch);
    }
    let needed = a.data().len();
    if output.len() < needed {
        return Err(MaskError::InsufficientCapacity);
    }
    for (index, (left, right)) in a.data().iter().zip(b.data().iter()).enumerate() {
        output[index] = op(*left != 0, *right != 0);
    }
    Ok(())
}

/// Intersection-over-union for two cropped masks in a shared full-frame space.
#[must_use]
pub fn cropped_mask_iou(a: CroppedMask<'_>, b: CroppedMask<'_>) -> f32 {
    let left = a.origin_x().max(b.origin_x());
    let top = a.origin_y().max(b.origin_y());
    let right = (a.origin_x() + a.width()).min(b.origin_x() + b.width());
    let bottom = (a.origin_y() + a.height()).min(b.origin_y() + b.height());

    let mut inter = 0_u32;
    if right > left && bottom > top {
        for y in top..bottom {
            for x in left..right {
                if a.is_set(x, y) && b.is_set(x, y) {
                    inter += 1;
                }
            }
        }
    }
    let uni = a.area() + b.area() - inter;
    if uni == 0 {
        0.0
    } else {
        inter as f32 / uni as f32
    }
}

/// Mask-level NMS scores: each entry is `(score, mask_index)`.
///
/// Suppresses lower-scoring masks whose intersection-over-union with a kept
/// mask exceeds `threshold`. Kept indices are written to `kept` in descending
/// score order (original index as tie-break). Returns the number of kept indices.
///
/// # Errors
///
/// Returns [`MaskError::InsufficientCapacity`] when scratch or output is short.
pub fn mask_nms_by_iou(
    scores: &[(f32, usize)],
    pairwise_iou: impl Fn(usize, usize) -> f32,
    threshold: f32,
    order_scratch: &mut [usize],
    suppressed_scratch: &mut [bool],
    kept: &mut [usize],
) -> Result<usize, MaskError> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(MaskError::NonFinite);
    }
    let len = scores.len();
    if order_scratch.len() < len || suppressed_scratch.len() < len || kept.len() < len {
        return Err(MaskError::InsufficientCapacity);
    }

    let order = &mut order_scratch[..len];
    let suppressed = &mut suppressed_scratch[..len];
    for (index, slot) in order.iter_mut().enumerate() {
        *slot = index;
    }
    for value in suppressed.iter_mut() {
        *value = false;
    }

    order.sort_unstable_by(|&left, &right| {
        scores[right]
            .0
            .partial_cmp(&scores[left].0)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| left.cmp(&right))
    });

    let mut kept_count = 0_usize;
    for candidate_pos in 0..len {
        let candidate = order[candidate_pos];
        if suppressed[candidate] {
            continue;
        }
        let candidate_mask = scores[candidate].1;
        kept[kept_count] = candidate_mask;
        kept_count += 1;

        for &contender in order.iter().skip(candidate_pos + 1) {
            if suppressed[contender] {
                continue;
            }
            let contender_mask = scores[contender].1;
            if pairwise_iou(candidate_mask, contender_mask) > threshold {
                suppressed[contender] = true;
            }
        }
    }
    Ok(kept_count)
}
