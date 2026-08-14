//! Synthetic MOT suite report + `MOTChallenge` export for external `TrackEval`.
//!
//! **Does not** download MOT17/MOT20 or claim leaderboard scores. Hosts run
//! `TrackEval` offline with files produced by [`write_mot_challenge_sequence`].

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::format;
#[cfg(feature = "alloc")]
use alloc::string::String;
#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
#[cfg(feature = "alloc")]
use core::fmt::Write as _;

use crate::{
    BaselineMotMetrics, ByteTrackConfig, MotFrame, TrackError, evaluate_baseline_mot,
    run_synthetic_crossing, run_synthetic_parallel_walk,
};

/// Named synthetic scenario.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MotScenarioId {
    /// Two objects walking in parallel (identity should hold).
    ParallelWalk,
    /// Crossing paths (stress / possible ID switches).
    Crossing,
    /// Single object with a long gap (occlusion / re-acquire stress).
    OcclusionGap,
    /// Three objects, moderate density.
    TripleLane,
}

/// One scenario result.
#[derive(Clone, Debug, PartialEq)]
pub struct MotScenarioResult {
    /// Scenario id.
    pub id: MotScenarioId,
    /// Metrics.
    pub metrics: BaselineMotMetrics,
    /// Pass/fail vs smoke thresholds for this scenario.
    pub smoke_pass: bool,
}

/// Full suite report (synthetic only).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MotSuiteReport {
    /// Per-scenario rows.
    pub scenarios: Vec<MotScenarioResult>,
}

impl MotSuiteReport {
    /// True when every scenario that defines a smoke gate passed.
    #[must_use]
    pub fn all_smoke_pass(&self) -> bool {
        !self.scenarios.is_empty() && self.scenarios.iter().all(|s| s.smoke_pass)
    }

    /// Markdown summary for host evidence packs (not a MOT17 claim).
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut out = String::from(
            "# Synthetic MOT suite (baseline)\n\n\
             > Not MOT17/MOT20/DanceTrack. Hosts attach TrackEval numbers separately.\n\n\
             | Scenario | Frames | MOTA | Precision | Recall | IDSW | Smoke |\n\
             | --- | ---: | ---: | ---: | ---: | ---: | --- |\n",
        );
        for s in &self.scenarios {
            let name = match s.id {
                MotScenarioId::ParallelWalk => "parallel_walk",
                MotScenarioId::Crossing => "crossing",
                MotScenarioId::OcclusionGap => "occlusion_gap",
                MotScenarioId::TripleLane => "triple_lane",
            };
            let pass = if s.smoke_pass { "PASS" } else { "FAIL" };
            let _ = writeln!(
                out,
                "| {name} | {} | {:.3} | {:.3} | {:.3} | {} | {pass} |",
                s.metrics.frames,
                s.metrics.mota,
                s.metrics.precision,
                s.metrics.recall,
                s.metrics.id_switches,
            );
        }
        out
    }
}

/// Runs the default synthetic suite with the given tracker config.
///
/// # Errors
///
/// Propagates tracker errors.
pub fn run_mot_smoke_suite(config: &ByteTrackConfig) -> Result<MotSuiteReport, TrackError> {
    let mut report = MotSuiteReport::default();

    let m = run_synthetic_parallel_walk(config, 30)?;
    report.scenarios.push(MotScenarioResult {
        id: MotScenarioId::ParallelWalk,
        smoke_pass: m.mota > 0.9 && m.id_switches == 0 && m.recall > 0.9,
        metrics: m,
    });

    let m = run_synthetic_crossing(config, 24)?;
    report.scenarios.push(MotScenarioResult {
        id: MotScenarioId::Crossing,
        // Crossing may switch IDs — require only that it runs with some TP.
        smoke_pass: m.true_positives > 0 && m.frames >= 24,
        metrics: m,
    });

    let m = run_synthetic_occlusion_gap(config, 40, 10)?;
    report.scenarios.push(MotScenarioResult {
        id: MotScenarioId::OcclusionGap,
        smoke_pass: m.mota > 0.5 && m.true_positives > 0,
        metrics: m,
    });

    let m = run_synthetic_triple_lane(config, 25)?;
    report.scenarios.push(MotScenarioResult {
        id: MotScenarioId::TripleLane,
        smoke_pass: m.mota > 0.85 && m.id_switches <= 2,
        metrics: m,
    });

    Ok(report)
}

/// Single box disappears for `gap` frames then returns (same GT id).
///
/// # Errors
///
/// Tracker errors.
pub fn run_synthetic_occlusion_gap(
    config: &ByteTrackConfig,
    frames: u32,
    gap: u32,
) -> Result<BaselineMotMetrics, TrackError> {
    use crate::{ByteTracker, MotObject, mot_from_track};
    use sightloom_core::{ClassId, Detection, Rect};

    let mut tracker = ByteTracker::new(*config)?;
    let mut mot_frames: Vec<MotFrame> = Vec::new();
    let frames_n = frames.max(gap.saturating_add(4));
    let mid = frames_n / 2;
    let gap = gap.min(frames_n / 3);

    for t in 0..frames_n {
        let dx = (t as f32) * 1.5;
        let gt = Rect::new(20.0 + dx, 20.0, 40.0 + dx, 60.0).map_err(|_| TrackError::NonFinite)?;
        let in_gap = t >= mid && t < mid + gap;
        let dets = if in_gap {
            Vec::new()
        } else {
            vec![
                Detection::new(gt, 0.9, Some(ClassId(0)), None)
                    .map_err(|_| TrackError::NonFinite)?,
            ]
        };
        let hyp = tracker.update(&dets)?;
        mot_frames.push(MotFrame {
            gt: if in_gap {
                Vec::new()
            } else {
                vec![MotObject { id: 1, bbox: gt }]
            },
            hyp: hyp
                .iter()
                .filter_map(|d| d.track_id().map(|tid| mot_from_track(tid, d.bbox())))
                .collect(),
        });
    }
    Ok(evaluate_baseline_mot(&mot_frames, 0.5))
}

/// Three parallel non-overlapping lanes.
///
/// # Errors
///
/// Tracker errors.
pub fn run_synthetic_triple_lane(
    config: &ByteTrackConfig,
    frames: u32,
) -> Result<BaselineMotMetrics, TrackError> {
    use crate::{ByteTracker, MotObject, mot_from_track};
    use sightloom_core::{ClassId, Detection, Rect};

    let mut tracker = ByteTracker::new(*config)?;
    let mut mot_frames: Vec<MotFrame> = Vec::new();
    let frames_n = frames.max(1);
    for t in 0..frames_n {
        let dx = (t as f32) * 2.0;
        let boxes = [
            Rect::new(10.0 + dx, 10.0, 30.0 + dx, 40.0).map_err(|_| TrackError::NonFinite)?,
            Rect::new(50.0 + dx, 10.0, 70.0 + dx, 40.0).map_err(|_| TrackError::NonFinite)?,
            Rect::new(90.0 + dx, 10.0, 110.0 + dx, 40.0).map_err(|_| TrackError::NonFinite)?,
        ];
        let dets: Result<Vec<_>, _> = boxes
            .iter()
            .map(|b| {
                Detection::new(*b, 0.9, Some(ClassId(0)), None).map_err(|_| TrackError::NonFinite)
            })
            .collect();
        let dets = dets?;
        let hyp = tracker.update(&dets)?;
        mot_frames.push(MotFrame {
            gt: vec![
                MotObject {
                    id: 1,
                    bbox: boxes[0],
                },
                MotObject {
                    id: 2,
                    bbox: boxes[1],
                },
                MotObject {
                    id: 3,
                    bbox: boxes[2],
                },
            ],
            hyp: hyp
                .iter()
                .filter_map(|d| d.track_id().map(|tid| mot_from_track(tid, d.bbox())))
                .collect(),
        });
    }
    Ok(evaluate_baseline_mot(&mot_frames, 0.5))
}

/// `MOTChallenge` DET/GT line format helper (one sequence).
///
/// Confidence column is typically `1` for GT and the detection score for hyp.
#[must_use]
pub fn format_mot_challenge_line(
    frame_1based: u32,
    id: u32,
    left: f32,
    top: f32,
    width: f32,
    height: f32,
    conf: f32,
) -> String {
    format!("{frame_1based},{id},{left:.2},{top:.2},{width:.2},{height:.2},{conf:.2},-1,-1,-1")
}

/// Builds `MOTChallenge` text for GT or hypothesis tracks from [`MotFrame`]s.
///
/// Uses confidence `1.0` for both GT and hypothesis exports (hosts can re-score).
#[must_use]
pub fn write_mot_challenge_sequence(frames: &[MotFrame], hypotheses: bool) -> String {
    let mut out = String::new();
    for (fi, frame) in frames.iter().enumerate() {
        let frame_1 = u32::try_from(fi + 1).unwrap_or(u32::MAX);
        let objs = if hypotheses { &frame.hyp } else { &frame.gt };
        for o in objs {
            let w = o.bbox.right() - o.bbox.left();
            let h = o.bbox.bottom() - o.bbox.top();
            out.push_str(&format_mot_challenge_line(
                frame_1,
                o.id,
                o.bbox.left(),
                o.bbox.top(),
                w,
                h,
                1.0,
            ));
            out.push('\n');
        }
    }
    out
}

#[cfg(test)]
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
    fn smoke_suite_runs() {
        let report = run_mot_smoke_suite(&cfg()).unwrap();
        assert_eq!(report.scenarios.len(), 4);
        assert!(report.all_smoke_pass(), "{:?}", report.to_markdown());
        let md = report.to_markdown();
        assert!(md.contains("parallel_walk"));
        assert!(md.contains("MOTA"));
    }

    #[test]
    fn mot_challenge_export_nonempty() {
        use crate::MotObject;
        use sightloom_core::Rect;
        let frames = [MotFrame {
            gt: vec![MotObject {
                id: 1,
                bbox: Rect::new(0.0, 0.0, 10.0, 20.0).unwrap(),
            }],
            hyp: vec![MotObject {
                id: 7,
                bbox: Rect::new(0.0, 0.0, 10.0, 20.0).unwrap(),
            }],
        }];
        let gt = write_mot_challenge_sequence(&frames, false);
        assert!(gt.starts_with("1,1,"));
        let hyp = write_mot_challenge_sequence(&frames, true);
        assert!(hyp.contains(",7,"));
    }
}
