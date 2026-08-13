//! Deterministic synthetic MOT scenarios for baseline regression.
//!
//! These are **not** MOT17/MOT20/DanceTrack scores. They only prove the
//! association tracker keeps identity on simple controlled sequences.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::{
    BaselineMotMetrics, ByteTrackConfig, ByteTracker, MotFrame, MotObject, TrackError,
    evaluate_baseline_mot, mot_from_track,
};
use sightloom_core::{ClassId, Detection, Rect};

/// Runs a two-object parallel walk through [`ByteTracker`] and scores CLEAR metrics.
///
/// Two non-overlapping boxes move right by 2 px each frame for `frames` steps.
/// Expected: high MOTA / recall with zero ID switches on a healthy baseline.
///
/// # Errors
///
/// Returns tracker configuration or update errors.
#[cfg(feature = "alloc")]
pub fn run_synthetic_parallel_walk(
    config: &ByteTrackConfig,
    frames: u32,
) -> Result<BaselineMotMetrics, TrackError> {
    let mut tracker = ByteTracker::new(*config)?;
    let mut mot_frames: Vec<MotFrame> = Vec::new();
    let frames_n = frames.max(1);

    for t in 0..frames_n {
        let dx = (t as f32) * 2.0;
        let gt_a =
            Rect::new(10.0 + dx, 10.0, 30.0 + dx, 50.0).map_err(|_| TrackError::NonFinite)?;
        let gt_b =
            Rect::new(80.0 + dx, 10.0, 100.0 + dx, 50.0).map_err(|_| TrackError::NonFinite)?;
        let dets = [
            Detection::new(gt_a, 0.9, Some(ClassId(0)), None).map_err(|_| TrackError::NonFinite)?,
            Detection::new(gt_b, 0.9, Some(ClassId(0)), None).map_err(|_| TrackError::NonFinite)?,
        ];
        let hyp = tracker.update(&dets)?;
        mot_frames.push(MotFrame {
            gt: vec![
                MotObject { id: 1, bbox: gt_a },
                MotObject { id: 2, bbox: gt_b },
            ],
            hyp: hyp
                .iter()
                .filter_map(|d| {
                    let tid = d.track_id()?;
                    Some(mot_from_track(tid, d.bbox()))
                })
                .collect(),
        });
    }

    Ok(evaluate_baseline_mot(&mot_frames, 0.5))
}

/// Crossing walk: two boxes swap horizontal lanes mid-sequence.
///
/// Useful as a stress smoke test; baseline MOTA may drop due to ID switches.
///
/// # Errors
///
/// Returns tracker configuration or update errors.
#[cfg(feature = "alloc")]
pub fn run_synthetic_crossing(
    config: &ByteTrackConfig,
    frames: u32,
) -> Result<BaselineMotMetrics, TrackError> {
    let mut tracker = ByteTracker::new(*config)?;
    let mut mot_frames: Vec<MotFrame> = Vec::new();
    let frames_n = frames.max(2);

    for t in 0..frames_n {
        let progress = t as f32 / (frames_n.saturating_sub(1).max(1) as f32);
        // Object A moves left→right, B right→left, crossing near mid.
        let ax = 10.0 + progress * 100.0;
        let bx = 110.0 - progress * 100.0;
        let gt_a = Rect::new(ax, 10.0, ax + 20.0, 50.0).map_err(|_| TrackError::NonFinite)?;
        let gt_b = Rect::new(bx, 10.0, bx + 20.0, 50.0).map_err(|_| TrackError::NonFinite)?;
        let dets = [
            Detection::new(gt_a, 0.9, Some(ClassId(0)), None).map_err(|_| TrackError::NonFinite)?,
            Detection::new(gt_b, 0.9, Some(ClassId(0)), None).map_err(|_| TrackError::NonFinite)?,
        ];
        let hyp = tracker.update(&dets)?;
        mot_frames.push(MotFrame {
            gt: vec![
                MotObject { id: 1, bbox: gt_a },
                MotObject { id: 2, bbox: gt_b },
            ],
            hyp: hyp
                .iter()
                .filter_map(|d| {
                    let tid = d.track_id()?;
                    Some(mot_from_track(tid, d.bbox()))
                })
                .collect(),
        });
    }

    Ok(evaluate_baseline_mot(&mot_frames, 0.5))
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::ByteTrackConfig;

    fn cfg() -> ByteTrackConfig {
        ByteTrackConfig {
            track_high_thresh: 0.5,
            track_activation_thresh: 0.5,
            track_low_thresh: 0.1,
            match_thresh: 0.3,
            max_time_lost: 30,
            class_aware: false,
        }
    }

    #[test]
    fn parallel_walk_keeps_identity() {
        let m = run_synthetic_parallel_walk(&cfg(), 20).unwrap();
        assert!(m.frames >= 20);
        assert!(m.mota > 0.9, "mota={}", m.mota);
        assert_eq!(m.id_switches, 0);
        assert!(m.recall > 0.9);
        assert!(m.precision > 0.9);
    }

    #[test]
    fn crossing_scenario_runs() {
        let m = run_synthetic_crossing(&cfg(), 16).unwrap();
        assert!(m.frames >= 16);
        // Do not require perfect MOTA — this is a stress smoke, not a claim.
        assert!(m.true_positives > 0);
    }
}
