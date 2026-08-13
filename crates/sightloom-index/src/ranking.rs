//! Rank subjects by frequency / visibility inside a `VisionIndex`.

use crate::VisionIndex;
use sightloom_core::SubjectId;

/// One ranked subject row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubjectRank {
    /// Subject id.
    pub subject_id: SubjectId,
    /// Number of effective track samples with this subject.
    pub sample_count: u64,
    /// Number of distinct sources observed.
    pub source_count: u32,
    /// Approximate total visible span in nanoseconds
    /// (last pts − first pts across samples; 0 if single sample).
    pub visible_span_ns: i64,
}

/// Ranks subjects by sample count (most frequent first).
///
/// Ties break by longer visible span, then lower subject id.
#[must_use]
pub fn rank_subjects_by_frequency(index: &VisionIndex) -> Vec<SubjectRank> {
    let mut rows: Vec<(SubjectId, u64, Vec<u32>, i64, i64)> = Vec::new();

    for sample in index.tracks.effective_samples() {
        let Some(subject_id) = sample.subject_id else {
            continue;
        };
        let t = sample.pts.as_nanos();
        if let Some(row) = rows.iter_mut().find(|(id, _, _, _, _)| *id == subject_id) {
            row.1 = row.1.saturating_add(1);
            if !row.2.contains(&sample.source_id.0) {
                row.2.push(sample.source_id.0);
            }
            row.3 = row.3.min(t);
            row.4 = row.4.max(t);
        } else {
            rows.push((subject_id, 1, vec![sample.source_id.0], t, t));
        }
    }

    let mut ranks: Vec<SubjectRank> = rows
        .into_iter()
        .map(
            |(subject_id, sample_count, sources, min_ns, max_ns)| SubjectRank {
                subject_id,
                sample_count,
                source_count: u32::try_from(sources.len()).unwrap_or(u32::MAX),
                visible_span_ns: max_ns.saturating_sub(min_ns),
            },
        )
        .collect();

    ranks.sort_by(|a, b| {
        b.sample_count
            .cmp(&a.sample_count)
            .then_with(|| b.visible_span_ns.cmp(&a.visible_span_ns))
            .then_with(|| a.subject_id.0.cmp(&b.subject_id.0))
    });
    ranks
}

/// Returns the single most frequent subject, if any samples are labeled.
#[must_use]
pub fn most_frequent_subject(index: &VisionIndex) -> Option<SubjectRank> {
    rank_subjects_by_frequency(index).into_iter().next()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TrackSample, VisionIndex};
    use sightloom_core::{MediaTime, SourceId, SubjectId, TrackId};

    fn sample(subject: u64, frame: u64, source: u32) -> TrackSample {
        TrackSample {
            sample_id: 0,
            supersedes: None,
            revision: 0,
            idempotency_key: 0,
            source_id: SourceId(source),
            frame_index: frame,
            pts: MediaTime::new(frame as i64, 1).unwrap(),
            track_id: TrackId(1),
            track_uid: None,
            subject_id: Some(SubjectId(subject)),
            class_id: None,
            left: 0.0,
            top: 0.0,
            right: 1.0,
            bottom: 1.0,
            confidence: 0.9,
            mask_ref: 0,
        }
    }

    #[test]
    fn ranks_most_frequent_first() {
        let mut index = VisionIndex::new("rank");
        for f in 0..5 {
            index.push_track(sample(1, f, 1));
        }
        for f in 0..2 {
            index.push_track(sample(2, f, 1));
        }
        index.push_track(sample(1, 10, 2));
        let ranks = rank_subjects_by_frequency(&index);
        assert_eq!(ranks[0].subject_id, SubjectId(1));
        assert_eq!(ranks[0].sample_count, 6);
        assert_eq!(ranks[0].source_count, 2);
        assert_eq!(ranks[1].subject_id, SubjectId(2));
        assert_eq!(
            most_frequent_subject(&index).unwrap().subject_id,
            SubjectId(1)
        );
    }
}
