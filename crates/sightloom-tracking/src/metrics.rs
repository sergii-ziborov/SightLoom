//! Baseline multi-object tracking metrics for synthetic regression and smoke benches.
//!
//! These are **CLEAR-style baseline** helpers, not a full TrackEval/HOTA stack.
//! Until MOT17/MOT20/DanceTrack numbers are published, call the tracker a
//! **baseline** association tracker rather than a complete `ByteTrack` port.
#![allow(clippy::cast_possible_truncation, clippy::too_many_lines)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use sightloom_core::{Rect, TrackId, iou};

/// One ground-truth or hypothesis box in a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotObject {
    /// Stable identity (GT id or predicted track id).
    pub id: u32,
    /// Axis-aligned box.
    pub bbox: Rect,
}

/// One frame of GT and hypotheses.
#[derive(Clone, Debug, PartialEq)]
pub struct MotFrame {
    /// Ground-truth objects.
    pub gt: Vec<MotObject>,
    /// Tracker hypotheses.
    pub hyp: Vec<MotObject>,
}

/// Aggregate baseline MOT metrics over a sequence.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BaselineMotMetrics {
    /// Frames evaluated.
    pub frames: u32,
    /// True positives (matched pairs).
    pub true_positives: u32,
    /// False positives.
    pub false_positives: u32,
    /// False negatives (missed GT).
    pub false_negatives: u32,
    /// Identity switches (matched GT whose hyp id changed vs previous frame).
    pub id_switches: u32,
    /// Fragmentations (GT reappears after a gap under a new hyp id).
    pub fragmentations: u32,
    /// CLEAR MOTA in approximately `(-inf, 1]`.
    pub mota: f32,
    /// Precision `TP / (TP + FP)`.
    pub precision: f32,
    /// Recall `TP / (TP + FN)`.
    pub recall: f32,
    /// Simple IDF1 approximation: `2 * IDTP / (2 * IDTP + IDFP + IDFN)`.
    pub idf1: f32,
    /// Detection accuracy `TP / (TP + FP + FN)` (HOTA DetA, in-tree baseline).
    pub deta: f32,
    /// Association accuracy (IDF1 used as AssA stand-in).
    pub assa: f32,
    /// Baseline HOTA `sqrt(DetA * AssA)`. **Not** TrackEval MOT17 HOTA.
    pub hota: f32,
}

/// Evaluates baseline CLEAR metrics with greedy `IoU` matching per frame.
///
/// Matching threshold is applied as minimum `IoU` for a valid association.
#[must_use]
pub fn evaluate_baseline_mot(frames: &[MotFrame], iou_threshold: f32) -> BaselineMotMetrics {
    let mut tp = 0_u32;
    let mut fp = 0_u32;
    let mut fn_ = 0_u32;
    let mut id_switches = 0_u32;
    let mut fragmentations = 0_u32;
    let mut idtp = 0_u32;
    let mut idfp = 0_u32;
    let mut idfn = 0_u32;

    // Last hyp id matched to each GT id.
    let mut last_match: Vec<(u32, u32)> = Vec::new(); // (gt_id, hyp_id)
    // GT ids seen as unmatched in the previous frame (for fragmentation).
    let mut prev_unmatched_gt: Vec<u32> = Vec::new();

    for frame in frames {
        let mut used_gt = vec![false; frame.gt.len()];
        let mut used_hyp = vec![false; frame.hyp.len()];
        let mut pairs: Vec<(usize, usize, f32)> = Vec::new();
        for (gi, g) in frame.gt.iter().enumerate() {
            for (hi, h) in frame.hyp.iter().enumerate() {
                let score = iou(g.bbox, h.bbox);
                if score >= iou_threshold {
                    pairs.push((gi, hi, score));
                }
            }
        }
        pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(core::cmp::Ordering::Equal));

        let mut frame_matches: Vec<(u32, u32)> = Vec::new();
        for (gi, hi, _) in pairs {
            if used_gt[gi] || used_hyp[hi] {
                continue;
            }
            used_gt[gi] = true;
            used_hyp[hi] = true;
            tp = tp.saturating_add(1);
            let gt_id = frame.gt[gi].id;
            let hyp_id = frame.hyp[hi].id;
            frame_matches.push((gt_id, hyp_id));
            if let Some((_, prev_hyp)) = last_match.iter().find(|(g, _)| *g == gt_id)
                && *prev_hyp != hyp_id
            {
                id_switches = id_switches.saturating_add(1);
            }
            // Identity-aware TP when the same hyp id stays with the GT.
            if last_match.iter().any(|(g, h)| *g == gt_id && *h == hyp_id)
                || last_match.iter().all(|(g, _)| *g != gt_id)
            {
                idtp = idtp.saturating_add(1);
            }
            if prev_unmatched_gt.contains(&gt_id)
                && last_match
                    .iter()
                    .find(|(g, _)| *g == gt_id)
                    .is_some_and(|(_, h)| *h != hyp_id)
            {
                fragmentations = fragmentations.saturating_add(1);
            }
        }

        for (gi, g) in frame.gt.iter().enumerate() {
            if !used_gt[gi] {
                fn_ = fn_.saturating_add(1);
                idfn = idfn.saturating_add(1);
            }
            let _ = g;
        }
        for (hi, _) in frame.hyp.iter().enumerate() {
            if !used_hyp[hi] {
                fp = fp.saturating_add(1);
                idfp = idfp.saturating_add(1);
            }
        }

        // Update last_match for matched GTs; drop GTs not present this frame.
        let present_gt: Vec<u32> = frame.gt.iter().map(|g| g.id).collect();
        last_match.retain(|(g, _)| present_gt.contains(g));
        for (gt_id, hyp_id) in &frame_matches {
            if let Some(slot) = last_match.iter_mut().find(|(g, _)| g == gt_id) {
                slot.1 = *hyp_id;
            } else {
                last_match.push((*gt_id, *hyp_id));
            }
        }
        prev_unmatched_gt = frame
            .gt
            .iter()
            .enumerate()
            .filter(|(i, _)| !used_gt[*i])
            .map(|(_, g)| g.id)
            .collect();
    }

    let frames_n = u32::try_from(frames.len()).unwrap_or(u32::MAX);
    let denom = tp.saturating_add(fn_);
    let mota = if denom == 0 {
        0.0
    } else {
        1.0 - ((fn_.saturating_add(fp).saturating_add(id_switches) as f32) / (denom as f32))
    };
    let precision = if tp.saturating_add(fp) == 0 {
        0.0
    } else {
        (tp as f32) / (tp.saturating_add(fp) as f32)
    };
    let recall = if denom == 0 {
        0.0
    } else {
        (tp as f32) / (denom as f32)
    };
    let id_denom = (2 * idtp).saturating_add(idfp).saturating_add(idfn);
    let idf1 = if id_denom == 0 {
        0.0
    } else {
        (2.0 * (idtp as f32)) / (id_denom as f32)
    };
    let det_den = tp.saturating_add(fp).saturating_add(fn_);
    let deta = if det_den == 0 {
        0.0
    } else {
        (tp as f32) / (det_den as f32)
    };
    let assa = idf1;
    let hota = (deta * assa).max(0.0).sqrt();

    BaselineMotMetrics {
        frames: frames_n,
        true_positives: tp,
        false_positives: fp,
        false_negatives: fn_,
        id_switches,
        fragmentations,
        mota,
        precision,
        recall,
        idf1,
        deta,
        assa,
        hota,
    }
}

/// Helper: wraps a track id + rect as a MOT object.
#[must_use]
pub fn mot_from_track(track_id: TrackId, bbox: Rect) -> MotObject {
    MotObject {
        id: track_id.0,
        bbox,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sightloom_core::Rect;

    #[test]
    fn perfect_tracking_scores_high_mota() {
        let box_a = Rect::new(0.0, 0.0, 10.0, 20.0).unwrap();
        let frames = [
            MotFrame {
                gt: vec![MotObject { id: 1, bbox: box_a }],
                hyp: vec![MotObject { id: 7, bbox: box_a }],
            },
            MotFrame {
                gt: vec![MotObject { id: 1, bbox: box_a }],
                hyp: vec![MotObject { id: 7, bbox: box_a }],
            },
        ];
        let m = evaluate_baseline_mot(&frames, 0.5);
        assert_eq!(m.true_positives, 2);
        assert_eq!(m.false_positives, 0);
        assert_eq!(m.false_negatives, 0);
        assert_eq!(m.id_switches, 0);
        assert!((m.mota - 1.0).abs() < 1e-5);
        assert!((m.precision - 1.0).abs() < 1e-5);
        assert!((m.recall - 1.0).abs() < 1e-5);
        assert!((m.deta - 1.0).abs() < 1e-5);
        assert!((m.hota - 1.0).abs() < 1e-5);
    }

    #[test]
    fn id_switch_is_counted() {
        let box_a = Rect::new(0.0, 0.0, 10.0, 20.0).unwrap();
        let frames = [
            MotFrame {
                gt: vec![MotObject { id: 1, bbox: box_a }],
                hyp: vec![MotObject { id: 7, bbox: box_a }],
            },
            MotFrame {
                gt: vec![MotObject { id: 1, bbox: box_a }],
                hyp: vec![MotObject { id: 9, bbox: box_a }],
            },
        ];
        let m = evaluate_baseline_mot(&frames, 0.5);
        assert_eq!(m.id_switches, 1);
        assert!(m.mota < 1.0);
    }
}
