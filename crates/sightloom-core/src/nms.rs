//! Deterministic non-maximum suppression over caller-owned storage.
#![allow(clippy::cast_precision_loss, clippy::needless_range_loop)]

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

/// Soft-NMS decay method.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SoftNmsMethod {
    /// Linear score decay: `score * (1 - overlap)` when overlap &gt; threshold.
    Linear,
    /// Gaussian decay: `score * exp(-(overlap^2) / sigma)`.
    Gaussian,
}

/// Soft-NMS configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftNmsConfig {
    /// Overlap threshold for linear method (ignored by Gaussian except as 0 gate).
    pub threshold: f32,
    /// Gaussian sigma (`> 0`); used when method is Gaussian.
    pub sigma: f32,
    /// Minimum score after decay; detections below are dropped.
    pub score_threshold: f32,
    /// Decay method.
    pub method: SoftNmsMethod,
    /// Class comparison policy.
    pub mode: NmsMode,
    /// Overlap metric.
    pub metric: OverlapMetric,
}

impl Default for SoftNmsConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            sigma: 0.5,
            score_threshold: 0.001,
            method: SoftNmsMethod::Gaussian,
            mode: NmsMode::ClassAgnostic,
            metric: OverlapMetric::IoU,
        }
    }
}

/// Soft-NMS: decays scores of overlapping boxes instead of hard suppression.
///
/// Compacts survivors (score ≥ `score_threshold`) to the front of `detections`
/// in descending final-score order.
///
/// # Errors
///
/// Returns threshold / scratch errors similar to [`nms_in_place`].
pub fn soft_nms_in_place(
    detections: &mut [Detection],
    order_scratch: &mut [usize],
    scores_scratch: &mut [f32],
    config: SoftNmsConfig,
) -> Result<usize, CoreError> {
    if !config.threshold.is_finite()
        || !(0.0..=1.0).contains(&config.threshold)
        || !config.sigma.is_finite()
        || config.sigma <= 0.0
        || !config.score_threshold.is_finite()
    {
        return Err(CoreError::InvalidThreshold);
    }
    let len = detections.len();
    if order_scratch.len() < len || scores_scratch.len() < len {
        return Err(CoreError::InsufficientScratch);
    }
    let order = &mut order_scratch[..len];
    let scores = &mut scores_scratch[..len];
    for (i, d) in detections.iter().enumerate() {
        order[i] = i;
        scores[i] = d.score();
    }

    // Active indices live in order[0..remaining).
    let mut remaining = len;
    while remaining > 0 {
        let mut best_pos = 0_usize;
        for p in 1..remaining {
            let cmp = scores[order[p]]
                .partial_cmp(&scores[order[best_pos]])
                .unwrap_or(Ordering::Equal);
            if cmp == Ordering::Greater || (cmp == Ordering::Equal && order[p] < order[best_pos]) {
                best_pos = p;
            }
        }
        // Move best to the end of the active region and shrink.
        order.swap(best_pos, remaining - 1);
        let cand = order[remaining - 1];
        remaining -= 1;
        if scores[cand] < config.score_threshold {
            scores[cand] = 0.0;
            continue;
        }
        for p in 0..remaining {
            let other = order[p];
            if config.mode == NmsMode::ClassAware
                && detections[cand].class_id() != detections[other].class_id()
            {
                continue;
            }
            let overlap = match config.metric {
                OverlapMetric::IoU => iou(detections[cand].bbox(), detections[other].bbox()),
                OverlapMetric::IoS => ios(detections[cand].bbox(), detections[other].bbox()),
            };
            let weight = match config.method {
                SoftNmsMethod::Linear => {
                    if overlap > config.threshold {
                        1.0 - overlap
                    } else {
                        1.0
                    }
                }
                SoftNmsMethod::Gaussian => exp_neg_approx((overlap * overlap) / config.sigma),
            };
            scores[other] *= weight;
        }
    }

    for (i, slot) in order.iter_mut().enumerate() {
        *slot = i;
    }
    order.sort_unstable_by(|a, b| {
        scores[*b]
            .partial_cmp(&scores[*a])
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.cmp(b))
    });
    let mut out = 0_usize;
    for &i in order.iter() {
        if scores[i] < config.score_threshold {
            break;
        }
        let d = detections[i];
        detections[out] = Detection::new(d.bbox(), scores[i], d.class_id(), d.track_id())
            .map_err(|_| CoreError::NonFinite)?;
        out += 1;
    }
    Ok(out)
}

fn union_find_root(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Merges overlapping detections by averaging boxes and taking max score.
///
/// Compacts merged clusters to the front of `detections`.
///
/// # Errors
///
/// Same threshold / scratch validation as hard NMS.
pub fn merge_nms_in_place(
    detections: &mut [Detection],
    order_scratch: &mut [usize],
    parent_scratch: &mut [usize],
    config: NmsConfig,
) -> Result<usize, CoreError> {
    if !config.threshold.is_finite() || !(0.0..=1.0).contains(&config.threshold) {
        return Err(CoreError::InvalidThreshold);
    }
    let len = detections.len();
    if order_scratch.len() < len || parent_scratch.len() < len {
        return Err(CoreError::InsufficientScratch);
    }
    let parent = &mut parent_scratch[..len];
    for (i, p) in parent.iter_mut().enumerate() {
        *p = i;
    }
    for i in 0..len {
        for j in (i + 1)..len {
            if config.mode == NmsMode::ClassAware
                && detections[i].class_id() != detections[j].class_id()
            {
                continue;
            }
            let overlap = match config.metric {
                OverlapMetric::IoU => iou(detections[i].bbox(), detections[j].bbox()),
                OverlapMetric::IoS => ios(detections[i].bbox(), detections[j].bbox()),
            };
            if overlap > config.threshold {
                let a = union_find_root(parent, i);
                let b = union_find_root(parent, j);
                if a != b {
                    // Prefer higher score as root.
                    if detections[a].score() >= detections[b].score() {
                        parent[b] = a;
                    } else {
                        parent[a] = b;
                    }
                }
            }
        }
    }
    // Aggregate per root
    let used = order_scratch;
    for u in used.iter_mut().take(len) {
        *u = 0;
    }
    let mut out = 0_usize;
    for i in 0..len {
        let r = union_find_root(parent, i);
        if used[r] != 0 {
            continue;
        }
        used[r] = 1;
        let mut sum_l = 0.0_f32;
        let mut sum_t = 0.0_f32;
        let mut sum_r = 0.0_f32;
        let mut sum_b = 0.0_f32;
        let mut max_score = 0.0_f32;
        let mut n = 0_f32;
        let mut class_id = detections[r].class_id();
        let mut track_id = detections[r].track_id();
        for j in 0..len {
            if union_find_root(parent, j) != r {
                continue;
            }
            let b = detections[j].bbox();
            sum_l += b.left();
            sum_t += b.top();
            sum_r += b.right();
            sum_b += b.bottom();
            max_score = max_score.max(detections[j].score());
            n += 1.0;
            if detections[j].class_id().is_some() {
                class_id = detections[j].class_id();
            }
            if detections[j].track_id().is_some() {
                track_id = detections[j].track_id();
            }
        }
        if n <= 0.0 {
            continue;
        }
        let rect = crate::Rect::new(sum_l / n, sum_t / n, sum_r / n, sum_b / n)
            .map_err(|_| CoreError::NonFinite)?;
        detections[out] = Detection::new(rect, max_score, class_id, track_id)
            .map_err(|_| CoreError::NonFinite)?;
        out += 1;
    }
    Ok(out)
}

/// `exp(-x)` approximation for Soft-NMS Gaussian (portable, no libm).
fn exp_neg_approx(x: f32) -> f32 {
    if !x.is_finite() || x < 0.0 {
        return 1.0;
    }
    if x > 20.0 {
        return 0.0;
    }
    // Padé-ish / Taylor for e^{-x}
    let mut term = 1.0_f32;
    let mut sum = 1.0_f32;
    for k in 1..12 {
        term *= -x / k as f32;
        sum += term;
    }
    sum.clamp(0.0, 1.0)
}
