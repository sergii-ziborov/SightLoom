//! [`MOTChallenge`](https://motchallenge.net/) text parse / rebuild for host
//! TrackEval bridges.
//!
//! Format (comma-separated, 1-based frame index):
//! `frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z`
//!
//! SightLoom does **not** ship MOT17 data; hosts supply files.

// Clippy doc_markdown is noisy on MOTChallenge/TrackEval product names in docs.
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

extern crate alloc;

use crate::metrics::{BaselineMotMetrics, MotFrame, MotObject, evaluate_baseline_mot};
use crate::mot_report::format_mot_challenge_line;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;
use sightloom_core::Rect;

/// Parse failure for MOTChallenge / TrackEval text.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MotParseError {
    /// No usable rows.
    Empty,
    /// Line failed to parse (`1`-based line number in source text).
    BadLine {
        /// Source line number (1-based).
        line: usize,
        /// Short reason.
        reason: &'static str,
    },
}

impl fmt::Display for MotParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "empty MOTChallenge text"),
            Self::BadLine { line, reason } => {
                write!(f, "MOTChallenge line {line}: {reason}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MotParseError {}

/// One `MOTChallenge` detection / GT row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotChallengeRow {
    /// 1-based frame index.
    pub frame: u32,
    /// Track / identity id.
    pub id: u32,
    /// Box left.
    pub left: f32,
    /// Box top.
    pub top: f32,
    /// Box width.
    pub width: f32,
    /// Box height.
    pub height: f32,
    /// Confidence (GT usually `1.0`).
    pub conf: f32,
}

impl MotChallengeRow {
    /// Converts to an axis-aligned [`Rect`].
    ///
    /// # Errors
    ///
    /// Non-finite or inverted geometry.
    pub fn to_rect(self) -> Result<Rect, MotParseError> {
        Rect::new(
            self.left,
            self.top,
            self.left + self.width,
            self.top + self.height,
        )
        .map_err(|_| MotParseError::BadLine {
            line: 0,
            reason: "non-finite or inverted bbox",
        })
    }

    /// As a [`MotObject`].
    ///
    /// # Errors
    ///
    /// Geometry failures.
    pub fn to_mot_object(self) -> Result<MotObject, MotParseError> {
        Ok(MotObject {
            id: self.id,
            bbox: self.to_rect()?,
        })
    }
}

/// Parses `MOTChallenge` multi-line text into rows (skips blank / `#` comments).
///
/// # Errors
///
/// Empty input or malformed lines.
pub fn parse_mot_challenge_text(text: &str) -> Result<Vec<MotChallengeRow>, MotParseError> {
    let mut rows = Vec::new();
    for (idx, raw) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        rows.push(parse_mot_challenge_line(line, line_no)?);
    }
    if rows.is_empty() {
        return Err(MotParseError::Empty);
    }
    Ok(rows)
}

/// Parses a single `MOTChallenge` line.
///
/// # Errors
///
/// Malformed fields.
pub fn parse_mot_challenge_line(
    line: &str,
    line_no: usize,
) -> Result<MotChallengeRow, MotParseError> {
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    if parts.len() < 7 {
        return Err(MotParseError::BadLine {
            line: line_no,
            reason: "need at least 7 comma-separated fields",
        });
    }
    let frame = parse_u32(parts[0], line_no, "frame")?;
    let id = parse_u32(parts[1], line_no, "id")?;
    let left = parse_f32(parts[2], line_no, "left")?;
    let top = parse_f32(parts[3], line_no, "top")?;
    let width = parse_f32(parts[4], line_no, "width")?;
    let height = parse_f32(parts[5], line_no, "height")?;
    let conf = parse_f32(parts[6], line_no, "conf")?;
    if !(width.is_finite() && height.is_finite()) || width < 0.0 || height < 0.0 {
        return Err(MotParseError::BadLine {
            line: line_no,
            reason: "invalid width/height",
        });
    }
    Ok(MotChallengeRow {
        frame,
        id,
        left,
        top,
        width,
        height,
        conf,
    })
}

/// Serializes rows back to `MOTChallenge` text (sorted by frame, then id).
#[must_use]
pub fn write_mot_challenge_rows(rows: &[MotChallengeRow]) -> String {
    let mut sorted: Vec<MotChallengeRow> = rows.to_vec();
    sorted.sort_by(|a, b| a.frame.cmp(&b.frame).then(a.id.cmp(&b.id)));
    let mut out = String::new();
    for r in sorted {
        out.push_str(&format_mot_challenge_line(
            r.frame, r.id, r.left, r.top, r.width, r.height, r.conf,
        ));
        out.push('\n');
    }
    out
}

/// Groups GT and hypothesis rows into per-frame [`MotFrame`]s (union of frame ids).
///
/// # Errors
///
/// Invalid geometry.
pub fn mot_challenge_to_frames(
    gt: &[MotChallengeRow],
    hyp: &[MotChallengeRow],
) -> Result<Vec<MotFrame>, MotParseError> {
    let mut frame_ids: Vec<u32> = Vec::new();
    for r in gt.iter().chain(hyp.iter()) {
        if !frame_ids.contains(&r.frame) {
            frame_ids.push(r.frame);
        }
    }
    frame_ids.sort_unstable();
    let mut frames = Vec::with_capacity(frame_ids.len());
    for f in frame_ids {
        let mut g = Vec::new();
        for r in gt.iter().filter(|r| r.frame == f) {
            g.push(r.to_mot_object()?);
        }
        let mut h = Vec::new();
        for r in hyp.iter().filter(|r| r.frame == f) {
            h.push(r.to_mot_object()?);
        }
        frames.push(MotFrame { gt: g, hyp: h });
    }
    Ok(frames)
}

/// Parses GT + hyp `MOTChallenge` text and runs baseline CLEAR metrics.
///
/// # Errors
///
/// Parse / geometry failures.
pub fn evaluate_mot_challenge_pair(
    gt_text: &str,
    hyp_text: &str,
    iou_threshold: f32,
) -> Result<BaselineMotMetrics, MotParseError> {
    let gt = parse_mot_challenge_text(gt_text)?;
    let hyp = parse_mot_challenge_text(hyp_text)?;
    let frames = mot_challenge_to_frames(&gt, &hyp)?;
    Ok(evaluate_baseline_mot(&frames, iou_threshold))
}

fn parse_u32(s: &str, line: usize, field: &'static str) -> Result<u32, MotParseError> {
    s.parse::<u32>().map_err(|_| MotParseError::BadLine {
        line,
        reason: match field {
            "frame" => "bad frame",
            "id" => "bad id",
            _ => "bad integer field",
        },
    })
}

fn parse_f32(s: &str, line: usize, _field: &'static str) -> Result<f32, MotParseError> {
    let v = s.parse::<f32>().map_err(|_| MotParseError::BadLine {
        line,
        reason: "bad float field",
    })?;
    if !v.is_finite() {
        return Err(MotParseError::BadLine {
            line,
            reason: "non-finite float",
        });
    }
    Ok(v)
}

/// Host-attached TrackEval (or similar) summary numbers — **not** computed in-tree.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrackEvalSummary {
    /// Sequence / split name (`MOT17-02-FRCNN`, synthetic id, …).
    pub sequence: Option<String>,
    /// Evaluator name (`TrackEval`, custom, …).
    pub evaluator: Option<String>,
    /// HOTA when provided.
    pub hota: Option<f32>,
    /// CLEAR MOTA when provided.
    pub mota: Option<f32>,
    /// IDF1 when provided.
    pub idf1: Option<f32>,
    /// Precision when provided.
    pub precision: Option<f32>,
    /// Recall when provided.
    pub recall: Option<f32>,
    /// Identity switches when provided.
    pub id_switches: Option<u32>,
    /// DetA / AssA / LocA optional TrackEval extras (opaque keys in markdown only).
    pub extra_notes: Option<String>,
}

impl TrackEvalSummary {
    /// Markdown block for evidence packs (honest: host-supplied numbers).
    #[must_use]
    pub fn to_markdown(&self) -> String {
        use core::fmt::Write as _;
        let mut out = String::from(
            "# Host TrackEval summary (imported)\n\n\
             > Numbers below were **supplied by the host**, not computed by SightLoom.\n\n",
        );
        if let Some(s) = &self.sequence {
            let _ = writeln!(out, "- **sequence**: `{s}`");
        }
        if let Some(e) = &self.evaluator {
            let _ = writeln!(out, "- **evaluator**: `{e}`");
        }
        let _ = writeln!(out, "\n| Metric | Value |\n| --- | ---: |");
        write_opt_f32(&mut out, "HOTA", self.hota);
        write_opt_f32(&mut out, "MOTA", self.mota);
        write_opt_f32(&mut out, "IDF1", self.idf1);
        write_opt_f32(&mut out, "Precision", self.precision);
        write_opt_f32(&mut out, "Recall", self.recall);
        if let Some(v) = self.id_switches {
            let _ = writeln!(out, "| IDSW | {v} |");
        }
        if let Some(n) = &self.extra_notes {
            let _ = writeln!(out, "\n## Notes\n\n{n}\n");
        }
        out
    }
}

fn write_opt_f32(out: &mut String, name: &str, v: Option<f32>) {
    use core::fmt::Write as _;
    if let Some(x) = v {
        let _ = writeln!(out, "| {name} | {x:.4} |");
    }
}

/// Parses a compact host summary:
/// - JSON object with keys `hota`, `mota`, `idf1`, `precision`, `recall`, `id_switches`,
///   `sequence`, `evaluator`, `notes`
/// - or line-oriented `KEY: value` / `KEY=value` (case-insensitive keys)
///
/// # Errors
///
/// No recognized metrics.
pub fn parse_track_eval_summary(text: &str) -> Result<TrackEvalSummary, MotParseError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(MotParseError::Empty);
    }
    if trimmed.starts_with('{') {
        return parse_track_eval_json_lite(trimmed);
    }
    parse_track_eval_kv(trimmed)
}

/// Minimal JSON object parser for numeric TrackEval fields (no full JSON crate).
fn parse_track_eval_json_lite(text: &str) -> Result<TrackEvalSummary, MotParseError> {
    // Intentionally tiny: extract "key": number | "key": "string"
    let s = TrackEvalSummary {
        hota: extract_json_f32(text, "hota"),
        mota: extract_json_f32(text, "mota"),
        idf1: extract_json_f32(text, "idf1"),
        precision: extract_json_f32(text, "precision"),
        recall: extract_json_f32(text, "recall"),
        id_switches: extract_json_f32(text, "id_switches")
            .or_else(|| extract_json_f32(text, "idsw"))
            .and_then(|v| u32::try_from(v as i64).ok()),
        sequence: extract_json_string(text, "sequence"),
        evaluator: extract_json_string(text, "evaluator"),
        extra_notes: extract_json_string(text, "notes"),
    };
    ensure_has_metric(&s)?;
    Ok(s)
}

fn parse_track_eval_kv(text: &str) -> Result<TrackEvalSummary, MotParseError> {
    let mut hota = None;
    let mut mota = None;
    let mut idf1 = None;
    let mut precision = None;
    let mut recall = None;
    let mut id_switches = None;
    let mut sequence = None;
    let mut evaluator = None;
    let mut extra_notes = None;
    for (idx, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, val) = split_kv(line).ok_or(MotParseError::BadLine {
            line: idx + 1,
            reason: "expected KEY: value or KEY=value",
        })?;
        let k = key.trim().to_ascii_lowercase();
        let v = val.trim();
        match k.as_str() {
            "hota" => hota = Some(parse_metric_f32(v, idx + 1)?),
            "mota" => mota = Some(parse_metric_f32(v, idx + 1)?),
            "idf1" => idf1 = Some(parse_metric_f32(v, idx + 1)?),
            "precision" | "prec" => precision = Some(parse_metric_f32(v, idx + 1)?),
            "recall" | "rec" => recall = Some(parse_metric_f32(v, idx + 1)?),
            "id_switches" | "idsw" | "ids" => {
                let n = parse_metric_f32(v, idx + 1)?;
                id_switches = u32::try_from(n as i64).ok();
            }
            "sequence" | "seq" => sequence = Some(v.to_string()),
            "evaluator" | "tool" => evaluator = Some(v.to_string()),
            "notes" | "note" => extra_notes = Some(v.to_string()),
            _ => {}
        }
    }
    let s = TrackEvalSummary {
        sequence,
        evaluator,
        hota,
        mota,
        idf1,
        precision,
        recall,
        id_switches,
        extra_notes,
    };
    ensure_has_metric(&s)?;
    Ok(s)
}

fn ensure_has_metric(s: &TrackEvalSummary) -> Result<(), MotParseError> {
    if s.hota.is_none()
        && s.mota.is_none()
        && s.idf1.is_none()
        && s.precision.is_none()
        && s.recall.is_none()
        && s.id_switches.is_none()
    {
        return Err(MotParseError::BadLine {
            line: 1,
            reason: "no recognized TrackEval metrics",
        });
    }
    Ok(())
}

fn split_kv(line: &str) -> Option<(&str, &str)> {
    if let Some((a, b)) = line.split_once(':') {
        return Some((a, b));
    }
    line.split_once('=')
}

fn parse_metric_f32(s: &str, line: usize) -> Result<f32, MotParseError> {
    let cleaned = s.trim().trim_end_matches('%');
    cleaned.parse::<f32>().map_err(|_| MotParseError::BadLine {
        line,
        reason: "bad metric float",
    })
}

fn extract_json_f32(text: &str, key: &str) -> Option<f32> {
    // Match "key": <number> (prefer quoted JSON keys).
    let quoted = {
        let mut s = String::from("\"");
        s.push_str(key);
        s.push('"');
        s
    };
    let bare = key;
    for pat in [quoted.as_str(), bare] {
        if let Some(pos) = find_json_key(text, pat) {
            let rest = text[pos..].trim_start();
            let after_colon = rest.split_once(':')?.1.trim_start();
            let num: String = after_colon
                .chars()
                .take_while(|c| {
                    c.is_ascii_digit()
                        || *c == '.'
                        || *c == '-'
                        || *c == '+'
                        || *c == 'e'
                        || *c == 'E'
                })
                .collect();
            if let Ok(v) = num.parse::<f32>() {
                return Some(v);
            }
        }
    }
    None
}

fn extract_json_string(text: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let pos = find_json_key(text, &pat)?;
    let rest = text[pos..].trim_start();
    let after_colon = rest.split_once(':')?.1.trim_start();
    if !after_colon.starts_with('"') {
        return None;
    }
    let inner = &after_colon[1..];
    let end = inner.find('"')?;
    Some(inner[..end].to_string())
}

fn find_json_key(text: &str, key_token: &str) -> Option<usize> {
    text.find(key_token)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mot_report::write_mot_challenge_sequence;
    use crate::{MotFrame, MotObject};
    use alloc::vec;
    use sightloom_core::Rect;

    #[test]
    fn roundtrip_challenge_text() {
        let frames = [MotFrame {
            gt: vec![MotObject {
                id: 1,
                bbox: Rect::new(1.0, 2.0, 11.0, 22.0).unwrap(),
            }],
            hyp: vec![MotObject {
                id: 9,
                bbox: Rect::new(1.0, 2.0, 11.0, 22.0).unwrap(),
            }],
        }];
        let gt = write_mot_challenge_sequence(&frames, false);
        let hyp = write_mot_challenge_sequence(&frames, true);
        let metrics = evaluate_mot_challenge_pair(&gt, &hyp, 0.5).unwrap();
        assert_eq!(metrics.frames, 1);
        assert_eq!(metrics.true_positives, 1);
        assert!((metrics.mota - 1.0).abs() < 1e-3);
    }

    #[test]
    fn parse_rejects_bad_line() {
        let err = parse_mot_challenge_text("not,enough").unwrap_err();
        assert!(matches!(err, MotParseError::BadLine { .. }));
    }

    #[test]
    fn track_eval_kv_and_json() {
        let kv = "\
sequence: MOT17-demo
evaluator: TrackEval
MOTA: 0.612
IDF1: 0.55
HOTA: 0.48
IDSW: 3
";
        let s = parse_track_eval_summary(kv).unwrap();
        assert_eq!(s.sequence.as_deref(), Some("MOT17-demo"));
        assert!((s.mota.unwrap() - 0.612).abs() < 1e-5);
        assert_eq!(s.id_switches, Some(3));

        let json = r#"{"sequence":"x","mota":0.7,"idf1":0.6,"hota":0.5}"#;
        let s2 = parse_track_eval_summary(json).unwrap();
        assert!((s2.mota.unwrap() - 0.7).abs() < 1e-5);
        assert!(s2.to_markdown().contains("Host TrackEval"));
    }
}
