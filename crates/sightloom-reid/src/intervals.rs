//! Uncertainty / identity decision intervals for host UI and policies.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::{IdentityMatch, MatchDecision};
use sightloom_core::{MediaTime, SourceId, SubjectId, TrackId};

/// One continuous interval where an identity decision held.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdentityInterval {
    /// Local track.
    pub track_id: TrackId,
    /// Source.
    pub source_id: SourceId,
    /// Subject under consideration (`None` if rejected / unknown).
    pub subject_id: Option<SubjectId>,
    /// Decision band.
    pub decision: MatchDecision,
    /// Interval start.
    pub start: MediaTime,
    /// Interval end (inclusive of last observation time).
    pub end: MediaTime,
    /// Best fused score observed in the interval when known.
    pub peak_score: Option<f32>,
}

/// One audit-like sample used to build intervals.
pub type IdentityPoint = (
    SourceId,
    TrackId,
    Option<SubjectId>,
    MatchDecision,
    MediaTime,
    Option<f32>,
);

/// Builds intervals by coalescing consecutive audit-like samples with the same
/// `(source, track, subject, decision)`.
#[must_use]
pub fn coalesce_identity_intervals(points: &[IdentityPoint]) -> Vec<IdentityInterval> {
    let mut out: Vec<IdentityInterval> = Vec::new();
    for &(source_id, track_id, subject_id, decision, at, score) in points {
        if let Some(last) = out.last_mut()
            && last.source_id == source_id
            && last.track_id == track_id
            && last.subject_id == subject_id
            && last.decision == decision
        {
            last.end = at;
            if let Some(s) = score {
                last.peak_score = Some(last.peak_score.map_or(s, |p| p.max(s)));
            }
            continue;
        }
        out.push(IdentityInterval {
            track_id,
            source_id,
            subject_id,
            decision,
            start: at,
            end: at,
            peak_score: score,
        });
    }
    out
}

/// Filters intervals to uncertain decisions only.
#[must_use]
pub fn uncertain_only(intervals: &[IdentityInterval]) -> Vec<IdentityInterval> {
    intervals
        .iter()
        .copied()
        .filter(|i| i.decision == MatchDecision::Uncertain)
        .collect()
}

/// Coalesces points but only merges when the time gap is ≤ `max_gap_ns`.
///
/// When `max_gap_ns` is `None`, behaves like [`coalesce_identity_intervals`].
#[must_use]
pub fn coalesce_identity_intervals_gapped(
    points: &[IdentityPoint],
    max_gap_ns: Option<i64>,
) -> Vec<IdentityInterval> {
    let Some(max_gap) = max_gap_ns else {
        return coalesce_identity_intervals(points);
    };
    let mut out: Vec<IdentityInterval> = Vec::new();
    for &(source_id, track_id, subject_id, decision, at, score) in points {
        if let Some(last) = out.last_mut()
            && last.source_id == source_id
            && last.track_id == track_id
            && last.subject_id == subject_id
            && last.decision == decision
        {
            let gap = at.duration_since_ns(last.end).unsigned_abs();
            let gap = i64::try_from(gap).unwrap_or(i64::MAX);
            if gap <= max_gap {
                last.end = at;
                if let Some(s) = score {
                    last.peak_score = Some(last.peak_score.map_or(s, |p| p.max(s)));
                }
                continue;
            }
        }
        out.push(IdentityInterval {
            track_id,
            source_id,
            subject_id,
            decision,
            start: at,
            end: at,
            peak_score: score,
        });
    }
    out
}

/// Builds a synthetic interval from a single match at time `at`.
#[must_use]
pub fn interval_from_match(
    source_id: SourceId,
    track_id: TrackId,
    at: MediaTime,
    m: IdentityMatch,
) -> IdentityInterval {
    IdentityInterval {
        track_id,
        source_id,
        subject_id: Some(m.subject_id),
        decision: m.decision,
        start: at,
        end: at,
        peak_score: Some(m.score),
    }
}
