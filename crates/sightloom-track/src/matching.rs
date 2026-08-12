//! Deterministic intersection-over-union assignment for track/detection pairs.

use sightloom_core::{ClassId, Rect, iou};

/// A candidate pair for assignment with an intersection-over-union score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatchCandidate {
    /// Track row index.
    pub track_index: usize,
    /// Detection column index.
    pub detection_index: usize,
    /// Intersection-over-union between predicted track box and detection box.
    pub iou: f32,
}

/// Sizes of the assignment outputs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AssignResult {
    /// Number of matched pairs written.
    pub match_count: usize,
    /// Number of unmatched track indices written.
    pub unmatched_track_count: usize,
    /// Number of unmatched detection indices written.
    pub unmatched_detection_count: usize,
}

/// Scratch buffers required by [`greedy_iou_assign`].
pub struct AssignScratch<'a> {
    /// Candidate pair storage (`tracks * detections` worst case).
    pub candidates: &'a mut [MatchCandidate],
    /// Per-track used flags (`track_boxes.len()`).
    pub track_used: &'a mut [bool],
    /// Per-detection used flags (`detection_boxes.len()`).
    pub detection_used: &'a mut [bool],
}

/// Greedy deterministic assignment sorted by descending score, then track index,
/// then detection index. Each track and detection is used at most once.
///
/// Pairs with intersection-over-union at or below `iou_threshold` are ignored.
#[allow(clippy::too_many_arguments)]
pub fn greedy_iou_assign(
    track_boxes: &[Rect],
    detection_boxes: &[Rect],
    track_classes: &[Option<ClassId>],
    detection_classes: &[Option<ClassId>],
    class_aware: bool,
    iou_threshold: f32,
    scratch: &mut AssignScratch<'_>,
    matches: &mut [(usize, usize)],
    unmatched_tracks: &mut [usize],
    unmatched_detections: &mut [usize],
) -> AssignResult {
    let mut candidate_count = 0_usize;
    for (ti, track_box) in track_boxes.iter().enumerate() {
        for (di, det_box) in detection_boxes.iter().enumerate() {
            if class_aware {
                let tc = track_classes.get(ti).copied().flatten();
                let dc = detection_classes.get(di).copied().flatten();
                if tc != dc {
                    continue;
                }
            }
            let score = iou(*track_box, *det_box);
            if score <= iou_threshold {
                continue;
            }
            if candidate_count < scratch.candidates.len() {
                scratch.candidates[candidate_count] = MatchCandidate {
                    track_index: ti,
                    detection_index: di,
                    iou: score,
                };
                candidate_count += 1;
            }
        }
    }

    let candidates = &mut scratch.candidates[..candidate_count];
    candidates.sort_unstable_by(|left, right| {
        right
            .iou
            .partial_cmp(&left.iou)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| left.track_index.cmp(&right.track_index))
            .then_with(|| left.detection_index.cmp(&right.detection_index))
    });

    let track_used_len = track_boxes.len().min(scratch.track_used.len());
    let det_used_len = detection_boxes.len().min(scratch.detection_used.len());
    let track_used = &mut scratch.track_used[..track_used_len];
    let det_used = &mut scratch.detection_used[..det_used_len];
    for value in track_used.iter_mut() {
        *value = false;
    }
    for value in det_used.iter_mut() {
        *value = false;
    }

    let mut match_count = 0_usize;
    for candidate in candidates.iter() {
        if candidate.track_index >= track_used.len() || candidate.detection_index >= det_used.len()
        {
            continue;
        }
        if track_used[candidate.track_index] || det_used[candidate.detection_index] {
            continue;
        }
        if match_count >= matches.len() {
            break;
        }
        matches[match_count] = (candidate.track_index, candidate.detection_index);
        match_count += 1;
        track_used[candidate.track_index] = true;
        det_used[candidate.detection_index] = true;
    }

    let mut ut = 0_usize;
    for index in 0..track_boxes.len() {
        let used = index < track_used.len() && track_used[index];
        if !used && ut < unmatched_tracks.len() {
            unmatched_tracks[ut] = index;
            ut += 1;
        }
    }
    let mut ud = 0_usize;
    for index in 0..detection_boxes.len() {
        let used = index < det_used.len() && det_used[index];
        if !used && ud < unmatched_detections.len() {
            unmatched_detections[ud] = index;
            ud += 1;
        }
    }

    AssignResult {
        match_count,
        unmatched_track_count: ut,
        unmatched_detection_count: ud,
    }
}
