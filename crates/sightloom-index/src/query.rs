//! Declarative query layer over an in-memory [`VisionIndex`].
//!
//! This is a **foundation** AST + executor (predicate composition, time range,
//! zone filter, confidence, pagination). It is not yet a full planner, spatial
//! index, or NL bridge — those build on these types.

use crate::{TrackSample, VisionIndex, ZoneStay};
use sightloom_core::{MediaTime, SourceId, SubjectId, ZoneId};

/// Sort order for query results.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum QueryOrder {
    /// Ascending by subject id.
    #[default]
    SubjectIdAsc,
    /// Descending by total sample count.
    SampleCountDesc,
}

/// Pagination cursor.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Page {
    /// Skip this many results.
    pub offset: usize,
    /// Return at most this many results (`0` = no limit).
    pub limit: usize,
}

/// Composable subject-oriented query.
#[derive(Clone, Debug, Default)]
pub struct SubjectQuery {
    /// Restrict to these subjects when non-empty.
    pub subject_ids: Vec<SubjectId>,
    /// Must have samples on this source.
    pub seen_on_source: Option<SourceId>,
    /// Must have a zone stay in this zone.
    pub seen_in_zone: Option<ZoneId>,
    /// Time window (inclusive) on track sample pts.
    pub during: Option<(MediaTime, MediaTime)>,
    /// Minimum dwell nanoseconds from zone stays.
    pub min_dwell_ns: Option<i64>,
    /// Minimum peak track confidence.
    pub min_confidence: Option<f32>,
    /// Pagination.
    pub page: Page,
    /// Sort order.
    pub order: QueryOrder,
}

impl SubjectQuery {
    /// Empty query (all subjects with any track sample).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: filter by zone.
    #[must_use]
    pub fn seen_in(mut self, zone: ZoneId) -> Self {
        self.seen_in_zone = Some(zone);
        self
    }

    /// Builder: time window.
    #[must_use]
    pub fn during(mut self, start: MediaTime, end: MediaTime) -> Self {
        self.during = Some((start, end));
        self
    }

    /// Builder: min dwell.
    #[must_use]
    pub fn with_min_dwell_ns(mut self, dwell_ns: i64) -> Self {
        self.min_dwell_ns = Some(dwell_ns);
        self
    }

    /// Builder: min confidence.
    #[must_use]
    pub fn with_min_confidence(mut self, confidence: f32) -> Self {
        self.min_confidence = Some(confidence);
        self
    }

    /// Builder: source filter.
    #[must_use]
    pub fn seen_on(mut self, source: SourceId) -> Self {
        self.seen_on_source = Some(source);
        self
    }

    /// Builder: pagination.
    #[must_use]
    pub fn page(mut self, offset: usize, limit: usize) -> Self {
        self.page = Page { offset, limit };
        self
    }
}

/// One subject row returned by [`execute_subject_query`].
#[derive(Clone, Debug, PartialEq)]
pub struct SubjectHit {
    /// Subject id.
    pub subject_id: SubjectId,
    /// Matching track samples (effective view recommended by caller).
    pub samples: Vec<TrackSample>,
    /// Matching zone stays.
    pub zone_stays: Vec<ZoneStay>,
    /// Peak confidence across samples.
    pub peak_confidence: f32,
    /// Total dwell ns from zone stays.
    pub total_dwell_ns: i64,
}

/// Executes a [`SubjectQuery`] against an in-memory index.
#[must_use]
pub fn execute_subject_query(index: &VisionIndex, query: &SubjectQuery) -> Vec<SubjectHit> {
    let mut by_subject: Vec<(SubjectId, Vec<TrackSample>)> = Vec::new();
    for sample in index.tracks.effective_samples() {
        let Some(subject_id) = sample.subject_id else {
            continue;
        };
        if !query.subject_ids.is_empty() && !query.subject_ids.contains(&subject_id) {
            continue;
        }
        if let Some(source) = query.seen_on_source
            && sample.source_id != source
        {
            continue;
        }
        if let Some((start, end)) = query.during {
            let t = sample.pts.as_nanos();
            if t < start.as_nanos() || t > end.as_nanos() {
                continue;
            }
        }
        if let Some(min_c) = query.min_confidence
            && sample.confidence < min_c
        {
            continue;
        }
        if let Some((_, samples)) = by_subject.iter_mut().find(|(id, _)| *id == subject_id) {
            samples.push(sample);
        } else {
            by_subject.push((subject_id, vec![sample]));
        }
    }

    let mut hits = Vec::new();
    for (subject_id, samples) in by_subject {
        let zone_stays: Vec<ZoneStay> = index
            .zone_stays
            .iter()
            .copied()
            .filter(|z| z.subject_id == Some(subject_id))
            .filter(|z| query.seen_in_zone.is_none_or(|zone| z.zone_id == zone))
            .collect();
        if let Some(zone) = query.seen_in_zone
            && !zone_stays.iter().any(|z| z.zone_id == zone)
        {
            continue;
        }
        let total_dwell_ns = zone_stays.iter().map(|z| z.duration_ns).sum();
        if let Some(min_dwell) = query.min_dwell_ns
            && total_dwell_ns < min_dwell
        {
            continue;
        }
        let peak_confidence = samples
            .iter()
            .map(|s| s.confidence)
            .fold(0.0_f32, f32::max);
        hits.push(SubjectHit {
            subject_id,
            samples,
            zone_stays,
            peak_confidence,
            total_dwell_ns,
        });
    }

    match query.order {
        QueryOrder::SubjectIdAsc => hits.sort_by_key(|h| h.subject_id.0),
        QueryOrder::SampleCountDesc => {
            hits.sort_by_key(|hit| core::cmp::Reverse(hit.samples.len()));
        }
    }

    let start = query.page.offset.min(hits.len());
    let end = if query.page.limit == 0 {
        hits.len()
    } else {
        start.saturating_add(query.page.limit).min(hits.len())
    };
    hits[start..end].to_vec()
}
