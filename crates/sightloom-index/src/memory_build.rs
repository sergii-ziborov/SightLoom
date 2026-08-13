//! Build first-class memory entities (appearances / visits / subject profiles)
//! and redaction provenance intervals from track samples.
//!
//! Track samples are the raw observation stream. Appearances, visits, and
//! subject profiles are derived video-memory records hosts can query without
//! replaying every box. Redaction intervals are exportable provenance rows
//! (no pixels).

use crate::{
    Appearance, RedactionIntent, RedactionInterval, SubjectProfile, TrackSample, VisionIndex, Visit,
};
use sightloom_core::{
    AppearanceId, MediaTime, RedactionIntervalId, SourceId, SubjectId, TrackId, VisitId,
};

/// Options for materializing appearances and visits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBuildConfig {
    /// Max gap between samples on the same `(source, track)` before a new appearance.
    pub appearance_gap_ns: i64,
    /// Max gap between consecutive appearances of a subject before a new visit.
    pub visit_gap_ns: i64,
    /// When true, only samples with `subject_id` contribute (recommended).
    pub require_subject: bool,
}

impl Default for MemoryBuildConfig {
    fn default() -> Self {
        Self {
            // ~1 second at 30 fps is too tight; 2s default for dropped frames.
            appearance_gap_ns: 2_000_000_000,
            // 5 minutes default between visit windows.
            visit_gap_ns: 300_000_000_000,
            require_subject: true,
        }
    }
}

/// One open appearance run while scanning samples.
#[derive(Clone, Debug)]
struct AppearanceRun {
    subject_id: Option<SubjectId>,
    track_id: TrackId,
    source_id: sightloom_core::SourceId,
    start: MediaTime,
    end: MediaTime,
    peak_confidence: f32,
    class_id: Option<sightloom_core::ClassId>,
}

/// Builds appearances from effective track samples.
///
/// Groups by `(source_id, local track_id)` and splits when consecutive samples
/// exceed `config.appearance_gap_ns`.
#[must_use]
pub fn build_appearances(
    samples: &[TrackSample],
    config: MemoryBuildConfig,
    next_id: &mut u64,
) -> Vec<Appearance> {
    let mut filtered: Vec<TrackSample> = samples
        .iter()
        .copied()
        .filter(|s| !config.require_subject || s.subject_id.is_some())
        .collect();
    filtered.sort_by(|a, b| {
        a.source_id
            .0
            .cmp(&b.source_id.0)
            .then_with(|| a.track_id.0.cmp(&b.track_id.0))
            .then_with(|| a.pts.as_nanos().cmp(&b.pts.as_nanos()))
            .then_with(|| a.frame_index.cmp(&b.frame_index))
    });

    let mut runs: Vec<AppearanceRun> = Vec::new();
    for sample in filtered {
        if let Some(run) = runs.last_mut() {
            let same = run.source_id == sample.source_id && run.track_id == sample.track_id;
            let gap = sample.pts.as_nanos().saturating_sub(run.end.as_nanos());
            if same && gap <= config.appearance_gap_ns {
                run.end = sample.pts;
                run.peak_confidence = run.peak_confidence.max(sample.confidence);
                if sample.class_id.is_some() {
                    run.class_id = sample.class_id;
                }
                if run.subject_id.is_none() {
                    run.subject_id = sample.subject_id;
                }
                continue;
            }
        }
        runs.push(AppearanceRun {
            subject_id: sample.subject_id,
            track_id: sample.track_id,
            source_id: sample.source_id,
            start: sample.pts,
            end: sample.pts,
            peak_confidence: sample.confidence,
            class_id: sample.class_id,
        });
    }

    runs.into_iter()
        .map(|run| {
            let id = AppearanceId(*next_id);
            *next_id = next_id.saturating_add(1);
            Appearance {
                appearance_id: id,
                subject_id: run.subject_id,
                track_id: Some(run.track_id),
                source_id: run.source_id,
                start: run.start,
                end: run.end,
                class_id: run.class_id,
                peak_confidence: run.peak_confidence,
                evidence: None,
            }
        })
        .collect()
}

/// Builds visits by merging a subject's appearances across sources.
///
/// Appearances sorted by start; merge when gap from previous visit end ≤
/// `config.visit_gap_ns` (or overlapping).
#[must_use]
pub fn build_visits(
    appearances: &[Appearance],
    config: MemoryBuildConfig,
    next_id: &mut u64,
) -> Vec<Visit> {
    let mut by_subject: Vec<(Option<SubjectId>, Vec<Appearance>)> = Vec::new();
    for appearance in appearances {
        if let Some((_, list)) = by_subject
            .iter_mut()
            .find(|(sid, _)| *sid == appearance.subject_id)
        {
            list.push(*appearance);
        } else {
            by_subject.push((appearance.subject_id, vec![*appearance]));
        }
    }

    let mut visits = Vec::new();
    for (subject_id, mut list) in by_subject {
        if config.require_subject && subject_id.is_none() {
            continue;
        }
        list.sort_by(|a, b| {
            a.start
                .as_nanos()
                .cmp(&b.start.as_nanos())
                .then_with(|| a.end.as_nanos().cmp(&b.end.as_nanos()))
        });

        let mut open_start: Option<MediaTime> = None;
        let mut open_end: Option<MediaTime> = None;
        let mut sources: Vec<u32> = Vec::new();

        for appearance in list {
            if let (Some(start), Some(end)) = (open_start, open_end) {
                let gap = appearance.start.as_nanos().saturating_sub(end.as_nanos());
                let overlaps = appearance.start.as_nanos() <= end.as_nanos();
                if overlaps || gap <= config.visit_gap_ns {
                    if appearance.end.as_nanos() > end.as_nanos() {
                        open_end = Some(appearance.end);
                    }
                    if !sources.contains(&appearance.source_id.0) {
                        sources.push(appearance.source_id.0);
                    }
                } else {
                    let id = VisitId(*next_id);
                    *next_id = next_id.saturating_add(1);
                    visits.push(Visit {
                        visit_id: id,
                        subject_id,
                        start,
                        end,
                        source_count: u32::try_from(sources.len()).unwrap_or(u32::MAX),
                        duration_ns: end.as_nanos().saturating_sub(start.as_nanos()),
                    });
                    open_start = Some(appearance.start);
                    open_end = Some(appearance.end);
                    sources = vec![appearance.source_id.0];
                }
            } else {
                open_start = Some(appearance.start);
                open_end = Some(appearance.end);
                sources = vec![appearance.source_id.0];
            }
        }
        if let (Some(start), Some(end)) = (open_start, open_end) {
            let id = VisitId(*next_id);
            *next_id = next_id.saturating_add(1);
            visits.push(Visit {
                visit_id: id,
                subject_id,
                start,
                end,
                source_count: u32::try_from(sources.len()).unwrap_or(u32::MAX),
                duration_ns: end.as_nanos().saturating_sub(start.as_nanos()),
            });
        }
    }
    visits
}

/// Rebuilds `appearances` and `visits` on the index from effective track samples.
///
/// Existing appearance/visit vectors are replaced (idempotent rebuild).
/// Returns `(appearance_count, visit_count)`.
pub fn rebuild_memory_entities(
    index: &mut VisionIndex,
    config: MemoryBuildConfig,
    next_appearance_id: &mut u64,
    next_visit_id: &mut u64,
) -> (usize, usize) {
    let samples = index.tracks.effective_samples();
    let appearances = build_appearances(&samples, config, next_appearance_id);
    let visits = build_visits(&appearances, config, next_visit_id);
    let a = appearances.len();
    let v = visits.len();
    index.appearances = appearances;
    index.visits = visits;
    (a, v)
}

/// Accumulator while aggregating a subject profile.
#[derive(Clone, Debug)]
struct ProfileAcc {
    appearance_count: u32,
    sources: Vec<u32>,
    total_duration_ns: i64,
    first_ns: i64,
    last_ns: i64,
    first: MediaTime,
    last: MediaTime,
}

fn touch_acc(acc: &mut ProfileAcc, start: MediaTime, end: MediaTime, source_id: SourceId) {
    let s = start.as_nanos();
    let e = end.as_nanos();
    acc.appearance_count = acc.appearance_count.saturating_add(1);
    if !acc.sources.contains(&source_id.0) {
        acc.sources.push(source_id.0);
    }
    acc.total_duration_ns = acc.total_duration_ns.saturating_add(e.saturating_sub(s));
    if s < acc.first_ns {
        acc.first_ns = s;
        acc.first = start;
    }
    if e > acc.last_ns {
        acc.last_ns = e;
        acc.last = end;
    }
}

/// Builds [`SubjectProfile`] rows from appearances (preferred) or track samples.
///
/// Host-supplied `label` and `embedding` on matching previous profiles are
/// preserved. Profiles are sorted by `subject_id`.
#[must_use]
pub fn build_subject_profiles(
    index: &VisionIndex,
    previous: &[SubjectProfile],
) -> Vec<SubjectProfile> {
    let mut acc: Vec<(SubjectId, ProfileAcc)> = Vec::new();

    if index.appearances.is_empty() {
        // Fallback: one "appearance unit" per effective labeled sample.
        for sample in index.tracks.effective_samples() {
            let Some(subject_id) = sample.subject_id else {
                continue;
            };
            if let Some((_, row)) = acc.iter_mut().find(|(id, _)| *id == subject_id) {
                touch_acc(row, sample.pts, sample.pts, sample.source_id);
            } else {
                let mut row = ProfileAcc {
                    appearance_count: 0,
                    sources: Vec::new(),
                    total_duration_ns: 0,
                    first_ns: sample.pts.as_nanos(),
                    last_ns: sample.pts.as_nanos(),
                    first: sample.pts,
                    last: sample.pts,
                };
                touch_acc(&mut row, sample.pts, sample.pts, sample.source_id);
                acc.push((subject_id, row));
            }
        }
    } else {
        for appearance in &index.appearances {
            let Some(subject_id) = appearance.subject_id else {
                continue;
            };
            if let Some((_, row)) = acc.iter_mut().find(|(id, _)| *id == subject_id) {
                touch_acc(row, appearance.start, appearance.end, appearance.source_id);
            } else {
                let mut row = ProfileAcc {
                    appearance_count: 0,
                    sources: Vec::new(),
                    total_duration_ns: 0,
                    first_ns: appearance.start.as_nanos(),
                    last_ns: appearance.end.as_nanos(),
                    first: appearance.start,
                    last: appearance.end,
                };
                touch_acc(
                    &mut row,
                    appearance.start,
                    appearance.end,
                    appearance.source_id,
                );
                acc.push((subject_id, row));
            }
        }
    }

    let mut profiles: Vec<SubjectProfile> = acc
        .into_iter()
        .map(|(subject_id, row)| {
            let prev = previous.iter().find(|p| p.subject_id == subject_id);
            SubjectProfile {
                subject_id,
                label: prev.and_then(|p| p.label.clone()),
                appearance_count: row.appearance_count,
                source_count: u32::try_from(row.sources.len()).unwrap_or(u32::MAX),
                total_duration_ns: row.total_duration_ns,
                first_seen: Some(row.first),
                last_seen: Some(row.last),
                embedding: prev.and_then(|p| p.embedding),
            }
        })
        .collect();

    // Keep host-only subjects that have no track/appearance evidence yet.
    for prev in previous {
        if !profiles.iter().any(|p| p.subject_id == prev.subject_id) {
            profiles.push(SubjectProfile {
                subject_id: prev.subject_id,
                label: prev.label.clone(),
                appearance_count: 0,
                source_count: 0,
                total_duration_ns: 0,
                first_seen: None,
                last_seen: None,
                embedding: prev.embedding,
            });
        }
    }

    profiles.sort_by_key(|p| p.subject_id.0);
    profiles
}

/// Rebuilds `index.subjects` from appearances (or tracks). Preserves labels /
/// embeddings. Returns profile count.
pub fn rebuild_subject_profiles(index: &mut VisionIndex) -> usize {
    let previous = index.subjects.clone();
    let profiles = build_subject_profiles(index, &previous);
    let n = profiles.len();
    index.subjects = profiles;
    n
}

/// Builds redaction provenance rows from appearances matching a filter.
///
/// - `include_subject = Some(id)`: only that subject (blur-subject path).
/// - `exclude_subject = Some(id)`: everyone except that subject (blur-others).
/// - both `None`: all labeled appearances with the given intent.
#[must_use]
pub fn build_redaction_from_appearances(
    appearances: &[Appearance],
    include_subject: Option<SubjectId>,
    exclude_subject: Option<SubjectId>,
    intent: RedactionIntent,
    tag: u32,
    next_id: &mut u64,
) -> Vec<RedactionInterval> {
    let mut out = Vec::new();
    for appearance in appearances {
        let Some(subject_id) = appearance.subject_id else {
            continue;
        };
        if let Some(want) = include_subject
            && subject_id != want
        {
            continue;
        }
        if let Some(skip) = exclude_subject
            && subject_id == skip
        {
            continue;
        }
        let id = RedactionIntervalId(*next_id);
        *next_id = next_id.saturating_add(1);
        out.push(RedactionInterval {
            interval_id: id,
            subject_id: Some(subject_id),
            source_id: appearance.source_id,
            track_id: appearance.track_id,
            start: appearance.start,
            end: appearance.end,
            intent,
            evidence: appearance.evidence,
            mask_ref: 0,
            peak_confidence: appearance.peak_confidence,
            appearance_id: Some(appearance.appearance_id),
            tag,
        });
    }
    out
}

/// Builds redaction rows from explicit host-provided interval specs.
#[must_use]
pub fn build_redaction_from_specs(
    specs: &[RedactionSpec],
    next_id: &mut u64,
) -> Vec<RedactionInterval> {
    specs
        .iter()
        .map(|spec| {
            let id = RedactionIntervalId(*next_id);
            *next_id = next_id.saturating_add(1);
            RedactionInterval {
                interval_id: id,
                subject_id: spec.subject_id,
                source_id: spec.source_id,
                track_id: spec.track_id,
                start: spec.start,
                end: spec.end,
                intent: spec.intent,
                evidence: spec.evidence,
                mask_ref: spec.mask_ref,
                peak_confidence: spec.peak_confidence,
                appearance_id: None,
                tag: spec.tag,
            }
        })
        .collect()
}

/// Host / re-id input for one provenance interval (no auto appearance link).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RedactionSpec {
    /// Subject in focus.
    pub subject_id: Option<SubjectId>,
    /// Source.
    pub source_id: SourceId,
    /// Track when known.
    pub track_id: Option<TrackId>,
    /// Start.
    pub start: MediaTime,
    /// End.
    pub end: MediaTime,
    /// Intent.
    pub intent: RedactionIntent,
    /// Evidence handle.
    pub evidence: Option<sightloom_core::EvidenceRef>,
    /// Mask handle.
    pub mask_ref: u64,
    /// Peak score / confidence.
    pub peak_confidence: f32,
    /// Host tag.
    pub tag: u32,
}

/// Replaces `index.redaction_intervals` with `rows` (idempotent assign).
pub fn set_redaction_intervals(index: &mut VisionIndex, rows: Vec<RedactionInterval>) {
    index.redaction_intervals = rows;
}

/// Appends rows to the redaction table.
pub fn append_redaction_intervals(index: &mut VisionIndex, rows: Vec<RedactionInterval>) {
    index.redaction_intervals.extend(rows);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TrackSample;
    use sightloom_core::{SourceId, SubjectId, TrackId};

    fn sample(subject: u64, source: u32, track: u32, ticks: i64, conf: f32) -> TrackSample {
        TrackSample {
            sample_id: 0,
            supersedes: None,
            revision: 0,
            idempotency_key: 0,
            source_id: SourceId(source),
            frame_index: u64::try_from(ticks).unwrap_or(0),
            pts: MediaTime::new(ticks, 30).unwrap(),
            track_id: TrackId(track),
            track_uid: None,
            subject_id: Some(SubjectId(subject)),
            class_id: None,
            left: 0.0,
            top: 0.0,
            right: 1.0,
            bottom: 1.0,
            confidence: conf,
            mask_ref: 0,
        }
    }

    #[test]
    fn gap_splits_appearances_and_merges_visits() {
        let samples = vec![
            sample(1, 1, 1, 0, 0.9),
            sample(1, 1, 1, 1, 0.95),   // same appearance (~33ms)
            sample(1, 1, 1, 100, 0.8),  // big gap → new appearance
            sample(1, 2, 2, 101, 0.85), // other source, near in time → same visit if gap large
        ];
        let mut next_a = 1;
        let config = MemoryBuildConfig {
            appearance_gap_ns: 50_000_000, // ~1.5 frames at 30fps
            visit_gap_ns: 10_000_000_000,
            require_subject: true,
        };
        let appearances = build_appearances(&samples, config, &mut next_a);
        assert_eq!(appearances.len(), 3);
        assert!((appearances[0].peak_confidence - 0.95).abs() < 1e-5);

        let mut next_v = 1;
        let visits = build_visits(&appearances, config, &mut next_v);
        // All three appearances merge into one visit (within 10s).
        assert_eq!(visits.len(), 1);
        assert_eq!(visits[0].source_count, 2);
        assert_eq!(visits[0].subject_id, Some(SubjectId(1)));
    }

    #[test]
    fn subject_profiles_and_redaction_from_appearances() {
        let mut index = VisionIndex::new("profiles");
        index.push_track(sample(7, 1, 1, 0, 0.9));
        index.push_track(sample(7, 1, 1, 1, 0.95));
        index.push_track(sample(9, 1, 2, 2, 0.8));
        let mut next_a = 1;
        let config = MemoryBuildConfig {
            appearance_gap_ns: 2_000_000_000,
            visit_gap_ns: 10_000_000_000,
            require_subject: true,
        };
        let mut next_v = 1;
        let _ = rebuild_memory_entities(&mut index, config, &mut next_a, &mut next_v);
        index.subjects.push(SubjectProfile {
            subject_id: SubjectId(7),
            label: Some("alice".into()),
            appearance_count: 0,
            source_count: 0,
            total_duration_ns: 0,
            first_seen: None,
            last_seen: None,
            embedding: None,
        });
        let n = rebuild_subject_profiles(&mut index);
        assert_eq!(n, 2);
        let alice = index
            .subjects
            .iter()
            .find(|p| p.subject_id == SubjectId(7))
            .unwrap();
        assert_eq!(alice.label.as_deref(), Some("alice"));
        assert!(alice.appearance_count >= 1);
        assert_eq!(alice.source_count, 1);

        let mut next_r = 1;
        let blur = build_redaction_from_appearances(
            &index.appearances,
            Some(SubjectId(7)),
            None,
            RedactionIntent::BlurSubject,
            0,
            &mut next_r,
        );
        assert!(!blur.is_empty());
        assert_eq!(blur[0].intent, RedactionIntent::BlurSubject);
        assert_eq!(blur[0].subject_id, Some(SubjectId(7)));

        let others = build_redaction_from_appearances(
            &index.appearances,
            None,
            Some(SubjectId(7)),
            RedactionIntent::BlurOthers,
            1,
            &mut next_r,
        );
        assert!(others.iter().all(|r| r.subject_id != Some(SubjectId(7))));
        assert!(!others.is_empty());
    }
}
