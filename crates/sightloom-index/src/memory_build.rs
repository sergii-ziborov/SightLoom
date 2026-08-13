//! Build first-class memory entities (appearances / visits) from track samples.
//!
//! Track samples are the raw observation stream. Appearances and visits are
//! derived video-memory records hosts can query without replaying every box.

use crate::{Appearance, TrackSample, VisionIndex, Visit};
use sightloom_core::{AppearanceId, MediaTime, SubjectId, TrackId, VisitId};

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
}
