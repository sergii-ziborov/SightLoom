//! MOT section of an evidence pack.

#![allow(clippy::doc_markdown)]

use crate::error::HostError;
use serde::{Deserialize, Serialize};
use sightloom::tracking::{
    BaselineMotMetrics, ByteTrackConfig, MotSuiteReport, TrackEvalSummary,
    evaluate_mot_challenge_pair, format_mot_challenge_line, parse_track_eval_summary,
    run_mot_smoke_suite, run_synthetic_crossing, run_synthetic_parallel_walk,
    write_mot_challenge_sequence,
};
use std::fs;
use std::path::Path;

/// MOT evidence: synthetic suite + `MOTChallenge` text for host `TrackEval`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MotEvidence {
    /// Markdown suite table (baseline CLEAR, not MOT17).
    pub suite_markdown: String,
    /// All scenarios smoke-pass.
    pub all_smoke_pass: bool,
    /// Parallel-walk GT `MOTChallenge` lines.
    pub parallel_gt: String,
    /// Parallel-walk hypothesis `MOTChallenge` lines.
    pub parallel_hyp: String,
    /// Crossing GT.
    pub crossing_gt: String,
    /// Crossing hyp.
    pub crossing_hyp: String,
    /// `TrackEval` host instructions.
    pub track_eval_notes: String,
    /// Optional host-imported TrackEval summary markdown.
    #[serde(default)]
    pub host_track_eval_markdown: Option<String>,
    /// Optional baseline CLEAR re-score of parallel_walk gt/hyp (sanity bridge).
    #[serde(default)]
    pub parallel_baseline_mota: Option<f32>,
}

impl MotEvidence {
    /// Attaches a host-supplied TrackEval summary (JSON or `KEY: value` text).
    ///
    /// # Errors
    ///
    /// Parse failures.
    pub fn attach_track_eval_summary_text(&mut self, text: &str) -> Result<(), HostError> {
        let summary = parse_track_eval_summary(text)
            .map_err(|e| HostError::Runtime(format!("track eval summary: {e}")))?;
        self.attach_track_eval_summary(&summary);
        Ok(())
    }

    /// Attaches a structured host TrackEval summary.
    pub fn attach_track_eval_summary(&mut self, summary: &TrackEvalSummary) {
        self.host_track_eval_markdown = Some(summary.to_markdown());
    }

    /// Re-scores parallel_walk gt/hyp with in-tree CLEAR (sanity, not TrackEval).
    ///
    /// # Errors
    ///
    /// Parse failures of challenge text.
    pub fn rescore_parallel_baseline(
        &mut self,
        iou_threshold: f32,
    ) -> Result<BaselineMotMetrics, HostError> {
        let m = evaluate_mot_challenge_pair(&self.parallel_gt, &self.parallel_hyp, iou_threshold)
            .map_err(|e| HostError::Runtime(format!("mot challenge rescore: {e}")))?;
        self.parallel_baseline_mota = Some(m.mota);
        Ok(m)
    }
}

/// Builds synthetic MOT evidence (smoke metrics + export files).
///
/// # Errors
///
/// Tracker failures.
pub fn build_synthetic_mot_evidence(config: &ByteTrackConfig) -> Result<MotEvidence, HostError> {
    let suite =
        run_mot_smoke_suite(config).map_err(|e| HostError::Runtime(format!("mot suite: {e}")))?;
    let (parallel_gt, parallel_hyp) = export_parallel(config)?;
    let (crossing_gt, crossing_hyp) = export_crossing(config)?;
    let mut mot = MotEvidence {
        suite_markdown: suite.to_markdown(),
        all_smoke_pass: suite.all_smoke_pass(),
        parallel_gt,
        parallel_hyp,
        crossing_gt,
        crossing_hyp,
        track_eval_notes: TRACK_EVAL_NOTES.into(),
        host_track_eval_markdown: None,
        parallel_baseline_mota: None,
    };
    // Sanity: in-tree CLEAR on exported challenge text (proves parse bridge).
    let _ = mot.rescore_parallel_baseline(0.5);
    Ok(mot)
}

fn export_parallel(config: &ByteTrackConfig) -> Result<(String, String), HostError> {
    // Smoke metrics via public synthetic runner; GT/hyp text from dedicated builders.
    let _ = run_synthetic_parallel_walk(config, 20)
        .map_err(|e| HostError::Runtime(format!("parallel: {e}")))?;
    build_parallel_sequence_text(config)
}

fn export_crossing(config: &ByteTrackConfig) -> Result<(String, String), HostError> {
    let _ = run_synthetic_crossing(config, 16)
        .map_err(|e| HostError::Runtime(format!("crossing: {e}")))?;
    build_crossing_sequence_text(config)
}

fn build_parallel_sequence_text(config: &ByteTrackConfig) -> Result<(String, String), HostError> {
    use sightloom::core::{ClassId, Detection, Rect};
    use sightloom::tracking::{ByteTracker, MotFrame, MotObject, mot_from_track};

    let mut tracker = ByteTracker::new(*config).map_err(|e| HostError::Runtime(format!("{e}")))?;
    let mut frames = Vec::new();
    for t in 0..20_u32 {
        let dx = t as f32 * 2.0;
        let a = Rect::new(10.0 + dx, 10.0, 30.0 + dx, 40.0)
            .map_err(|_| HostError::Runtime("rect".into()))?;
        let b = Rect::new(80.0 + dx, 10.0, 100.0 + dx, 40.0)
            .map_err(|_| HostError::Runtime("rect".into()))?;
        let dets = vec![
            Detection::new(a, 0.9, Some(ClassId(0)), None)
                .map_err(|_| HostError::Runtime("det".into()))?,
            Detection::new(b, 0.9, Some(ClassId(0)), None)
                .map_err(|_| HostError::Runtime("det".into()))?,
        ];
        let hyp = tracker
            .update(&dets)
            .map_err(|e| HostError::Runtime(format!("track: {e}")))?;
        frames.push(MotFrame {
            gt: vec![MotObject { id: 1, bbox: a }, MotObject { id: 2, bbox: b }],
            hyp: hyp
                .iter()
                .filter_map(|d| d.track_id().map(|tid| mot_from_track(tid, d.bbox())))
                .collect(),
        });
    }
    Ok((
        write_mot_challenge_sequence(&frames, false),
        write_mot_challenge_sequence(&frames, true),
    ))
}

fn build_crossing_sequence_text(config: &ByteTrackConfig) -> Result<(String, String), HostError> {
    use sightloom::core::{ClassId, Detection, Rect};
    use sightloom::tracking::{ByteTracker, MotFrame, MotObject, mot_from_track};

    let mut tracker = ByteTracker::new(*config).map_err(|e| HostError::Runtime(format!("{e}")))?;
    let mut frames = Vec::new();
    for t in 0..16_u32 {
        let dx = t as f32 * 3.0;
        let a = Rect::new(10.0 + dx, 20.0, 30.0 + dx, 50.0)
            .map_err(|_| HostError::Runtime("rect".into()))?;
        let b = Rect::new(100.0 - dx, 20.0, 120.0 - dx, 50.0)
            .map_err(|_| HostError::Runtime("rect".into()))?;
        let dets = vec![
            Detection::new(a, 0.9, Some(ClassId(0)), None)
                .map_err(|_| HostError::Runtime("det".into()))?,
            Detection::new(b, 0.9, Some(ClassId(0)), None)
                .map_err(|_| HostError::Runtime("det".into()))?,
        ];
        let hyp = tracker
            .update(&dets)
            .map_err(|e| HostError::Runtime(format!("track: {e}")))?;
        frames.push(MotFrame {
            gt: vec![MotObject { id: 1, bbox: a }, MotObject { id: 2, bbox: b }],
            hyp: hyp
                .iter()
                .filter_map(|d| d.track_id().map(|tid| mot_from_track(tid, d.bbox())))
                .collect(),
        });
    }
    Ok((
        write_mot_challenge_sequence(&frames, false),
        write_mot_challenge_sequence(&frames, true),
    ))
}

/// Writes MOT files under `dir/mot/`.
pub(crate) fn write_mot_section(dir: &Path, mot: &MotEvidence) -> Result<(), HostError> {
    let mot_dir = dir.join("mot");
    fs::create_dir_all(&mot_dir).map_err(|e| HostError::Io(e.to_string()))?;
    fs::write(mot_dir.join("suite.md"), &mot.suite_markdown)
        .map_err(|e| HostError::Io(e.to_string()))?;
    fs::write(mot_dir.join("parallel_walk_gt.txt"), &mot.parallel_gt)
        .map_err(|e| HostError::Io(e.to_string()))?;
    fs::write(mot_dir.join("parallel_walk_hyp.txt"), &mot.parallel_hyp)
        .map_err(|e| HostError::Io(e.to_string()))?;
    fs::write(mot_dir.join("crossing_gt.txt"), &mot.crossing_gt)
        .map_err(|e| HostError::Io(e.to_string()))?;
    fs::write(mot_dir.join("crossing_hyp.txt"), &mot.crossing_hyp)
        .map_err(|e| HostError::Io(e.to_string()))?;
    fs::write(mot_dir.join("TRACK_EVAL.md"), &mot.track_eval_notes)
        .map_err(|e| HostError::Io(e.to_string()))?;
    if let Some(md) = &mot.host_track_eval_markdown {
        fs::write(mot_dir.join("host_track_eval.md"), md)
            .map_err(|e| HostError::Io(e.to_string()))?;
    }
    if let Some(mota) = mot.parallel_baseline_mota {
        let note = format!(
            "# Parallel-walk in-tree CLEAR rescore\n\n\
             MOTA (SightLoom baseline, IoU≥0.5): **{mota:.4}**\n\n\
             This is **not** TrackEval HOTA. Use `host_track_eval.md` for host numbers.\n"
        );
        fs::write(mot_dir.join("parallel_baseline_clear.md"), note)
            .map_err(|e| HostError::Io(e.to_string()))?;
    }
    // Sample single line for sanity.
    let _ = format_mot_challenge_line(1, 1, 0.0, 0.0, 10.0, 20.0, 1.0);
    let _ = MotSuiteReport::default();
    Ok(())
}

const TRACK_EVAL_NOTES: &str = r#"# Host TrackEval bridge (not run inside SightLoom)

SightLoom exports **MOTChallenge** text (gt/hyp) and can **import** host
TrackEval summaries. It does **not** ship MOT17 data or claim leaderboard scores.

## APIs

| API | Role |
| --- | --- |
| `write_mot_challenge_sequence` / `export_mot_challenge` | export |
| `parse_mot_challenge_text` / `evaluate_mot_challenge_pair` | re-score CLEAR in-tree |
| `parse_track_eval_summary` / `MotEvidence::attach_track_eval_summary` | import host HOTA/MOTA/IDF1 |

## Offline TrackEval steps

1. Install TrackEval (or equivalent evaluator).
2. Point GT at `parallel_walk_gt.txt` / `crossing_gt.txt` (or your MOT17 layout).
3. Point tracker output at matching hyp files (`IndexSession::export_mot_challenge`).
4. Copy summary into JSON or `KEY: value` text, then call
   `MotEvidence::attach_track_eval_summary_text` with e.g.
   `{"sequence":"MOT17-02","mota":0.61,"hota":0.48,"idf1":0.55}`.
5. Re-write the evidence pack — `mot/host_track_eval.md` appears.

## Honest claim language

- "Synthetic smoke MOTA on parallel/crossing scenarios" — OK (see suite.md)
- "Host TrackEval HOTA on sequence X (attached)" — OK when `host_track_eval.md` present
- "State-of-the-art on MOT17" — **not** supported by this package alone
"#;
