//! Declarative query layer over an in-memory [`VisionIndex`].
//!
//! Foundation AST + executor: predicate composition, time range, zone filters,
//! then-seen-in chains, route prefix matching, confidence, pagination.
//! Not a full planner / spatial index / NL bridge.

use crate::{Route, TrackSample, VisionIndex, ZoneStay};
use sightloom_core::{MediaTime, SourceId, SubjectId, ZoneId};

/// Sort order for query results.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum QueryOrder {
    /// Ascending by subject id.
    #[default]
    SubjectIdAsc,
    /// Descending by total sample count.
    SampleCountDesc,
    /// Descending by total dwell.
    DwellDesc,
}

/// Pagination cursor.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Page {
    /// Skip this many results.
    pub offset: usize,
    /// Return at most this many results (`0` = no limit).
    pub limit: usize,
}

/// Require zone A then zone B within a time bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThenSeenIn {
    /// First zone.
    pub first: ZoneId,
    /// Second zone.
    pub then: ZoneId,
    /// Maximum nanoseconds from end of first stay to start of second (`0` = open).
    pub within_ns: i64,
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
    /// Optional A-then-B zone chain with time bound.
    pub then_seen_in: Option<ThenSeenIn>,
    /// Route must contain this ordered zone subsequence.
    pub route_contains: Vec<ZoneId>,
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

    /// Builder: seen in `first` then `then` within `within_ns` (0 = any later).
    #[must_use]
    pub fn then_seen_in(mut self, first: ZoneId, then: ZoneId, within_ns: i64) -> Self {
        self.then_seen_in = Some(ThenSeenIn {
            first,
            then,
            within_ns,
        });
        self
    }

    /// Builder: route must contain ordered zone subsequence.
    #[must_use]
    pub fn route_contains(mut self, zones: impl Into<Vec<ZoneId>>) -> Self {
        self.route_contains = zones.into();
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

    /// Builder: order.
    #[must_use]
    pub fn order(mut self, order: QueryOrder) -> Self {
        self.order = order;
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
    /// Matching routes.
    pub routes: Vec<Route>,
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
            .collect();

        if let Some(zone) = query.seen_in_zone
            && !zone_stays.iter().any(|z| z.zone_id == zone)
        {
            continue;
        }

        if let Some(chain) = query.then_seen_in
            && !matches_then_seen_in(&zone_stays, chain)
        {
            continue;
        }

        let routes: Vec<Route> = index
            .routes
            .iter()
            .filter(|r| r.subject_id == subject_id)
            .cloned()
            .collect();

        if !query.route_contains.is_empty()
            && !routes
                .iter()
                .any(|r| route_contains_subsequence(&r.zones, &query.route_contains))
        {
            continue;
        }

        let total_dwell_ns = zone_stays.iter().map(|z| z.duration_ns).sum();
        if let Some(min_dwell) = query.min_dwell_ns
            && total_dwell_ns < min_dwell
        {
            continue;
        }
        let peak_confidence = samples.iter().map(|s| s.confidence).fold(0.0_f32, f32::max);
        hits.push(SubjectHit {
            subject_id,
            samples,
            zone_stays,
            routes,
            peak_confidence,
            total_dwell_ns,
        });
    }

    match query.order {
        QueryOrder::SubjectIdAsc => hits.sort_by_key(|h| h.subject_id.0),
        QueryOrder::SampleCountDesc => {
            hits.sort_by_key(|hit| core::cmp::Reverse(hit.samples.len()));
        }
        QueryOrder::DwellDesc => {
            hits.sort_by_key(|hit| core::cmp::Reverse(hit.total_dwell_ns));
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

fn matches_then_seen_in(stays: &[ZoneStay], chain: ThenSeenIn) -> bool {
    let mut first_ends: Vec<i64> = stays
        .iter()
        .filter(|z| z.zone_id == chain.first)
        .map(|z| z.end.as_nanos())
        .collect();
    first_ends.sort_unstable();
    let mut second_starts: Vec<i64> = stays
        .iter()
        .filter(|z| z.zone_id == chain.then)
        .map(|z| z.start.as_nanos())
        .collect();
    second_starts.sort_unstable();

    for end in first_ends {
        for &start in &second_starts {
            if start < end {
                continue;
            }
            if chain.within_ns == 0 || start.saturating_sub(end) <= chain.within_ns {
                return true;
            }
        }
    }
    false
}

fn route_contains_subsequence(route: &[ZoneId], needle: &[ZoneId]) -> bool {
    if needle.is_empty() {
        return true;
    }
    if needle.len() > route.len() {
        return false;
    }
    route.windows(needle.len()).any(|window| window == needle)
}

/// Axis-aligned region query over effective track samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialQuery {
    /// Region left.
    pub left: f32,
    /// Region top.
    pub top: f32,
    /// Region right.
    pub right: f32,
    /// Region bottom.
    pub bottom: f32,
    /// Optional source filter.
    pub source_id: Option<SourceId>,
    /// Optional time window.
    pub during: Option<(MediaTime, MediaTime)>,
    /// Minimum sample confidence.
    pub min_confidence: Option<f32>,
    /// When true, only labeled subjects are returned.
    pub require_subject: bool,
    /// Pagination.
    pub page: Page,
}

impl SpatialQuery {
    /// Creates a spatial region query.
    #[must_use]
    pub fn new(left: f32, top: f32, right: f32, bottom: f32) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
            source_id: None,
            during: None,
            min_confidence: None,
            require_subject: false,
            page: Page::default(),
        }
    }

    /// Builder: source filter.
    #[must_use]
    pub fn on_source(mut self, source: SourceId) -> Self {
        self.source_id = Some(source);
        self
    }

    /// Builder: time window.
    #[must_use]
    pub fn during(mut self, start: MediaTime, end: MediaTime) -> Self {
        self.during = Some((start, end));
        self
    }

    /// Builder: require subject labels.
    #[must_use]
    pub fn with_subject(mut self) -> Self {
        self.require_subject = true;
        self
    }

    /// Builder: min confidence.
    #[must_use]
    pub fn with_min_confidence(mut self, confidence: f32) -> Self {
        self.min_confidence = Some(confidence);
        self
    }

    /// Builder: pagination.
    #[must_use]
    pub fn page(mut self, offset: usize, limit: usize) -> Self {
        self.page = Page { offset, limit };
        self
    }
}

/// One spatial hit (track sample intersecting the region).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialHit {
    /// Matching sample (effective view).
    pub sample: TrackSample,
    /// Subject when labeled.
    pub subject_id: Option<SubjectId>,
    /// Intersection-over-union of sample box vs query region (0 when degenerate).
    pub iou: f32,
}

/// Returns effective track samples whose boxes intersect the spatial region.
#[must_use]
pub fn execute_spatial_query(index: &VisionIndex, query: &SpatialQuery) -> Vec<SpatialHit> {
    let mut hits = Vec::new();
    for sample in index.tracks.effective_samples() {
        if let Some(source) = query.source_id
            && sample.source_id != source
        {
            continue;
        }
        if query.require_subject && sample.subject_id.is_none() {
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
        if !boxes_intersect(
            sample.left,
            sample.top,
            sample.right,
            sample.bottom,
            query.left,
            query.top,
            query.right,
            query.bottom,
        ) {
            continue;
        }
        let iou = box_iou(
            sample.left,
            sample.top,
            sample.right,
            sample.bottom,
            query.left,
            query.top,
            query.right,
            query.bottom,
        );
        hits.push(SpatialHit {
            sample,
            subject_id: sample.subject_id,
            iou,
        });
    }
    hits.sort_by(|a, b| {
        b.iou
            .partial_cmp(&a.iou)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| a.sample.sample_id.cmp(&b.sample.sample_id))
    });
    let start = query.page.offset.min(hits.len());
    let end = if query.page.limit == 0 {
        hits.len()
    } else {
        start.saturating_add(query.page.limit).min(hits.len())
    };
    hits[start..end].to_vec()
}

#[allow(clippy::too_many_arguments)]
fn boxes_intersect(
    a_l: f32,
    a_t: f32,
    a_r: f32,
    a_b: f32,
    b_l: f32,
    b_t: f32,
    b_r: f32,
    b_b: f32,
) -> bool {
    a_l < b_r && a_r > b_l && a_t < b_b && a_b > b_t
}

#[allow(clippy::too_many_arguments)]
fn box_iou(a_l: f32, a_t: f32, a_r: f32, a_b: f32, b_l: f32, b_t: f32, b_r: f32, b_b: f32) -> f32 {
    let inter_l = a_l.max(b_l);
    let inter_t = a_t.max(b_t);
    let inter_r = a_r.min(b_r);
    let inter_b = a_b.min(b_b);
    let inter_w = (inter_r - inter_l).max(0.0);
    let inter_h = (inter_b - inter_t).max(0.0);
    let inter = inter_w * inter_h;
    if inter <= 0.0 {
        return 0.0;
    }
    let area_a = ((a_r - a_l) * (a_b - a_t)).max(0.0);
    let area_b = ((b_r - b_l) * (b_b - b_t)).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TrackSample, VisionIndex, ZoneStay};
    use sightloom_core::{MediaTime, SourceId, SubjectId, TrackId, ZoneId};

    fn sample(subject: u64, frame: u64) -> TrackSample {
        TrackSample {
            sample_id: 0,
            supersedes: None,
            revision: 0,
            idempotency_key: 0,
            source_id: SourceId(1),
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
    fn then_seen_in_filters_subjects() {
        let mut index = VisionIndex::new("q");
        index.push_track(sample(1, 0));
        index.push_track(sample(2, 0));
        index.zone_stays.push(ZoneStay {
            zone_id: ZoneId(1),
            subject_id: Some(SubjectId(1)),
            track_id: None,
            start: MediaTime::new(0, 1).unwrap(),
            end: MediaTime::new(1, 1).unwrap(),
            duration_ns: 1,
        });
        index.zone_stays.push(ZoneStay {
            zone_id: ZoneId(2),
            subject_id: Some(SubjectId(1)),
            track_id: None,
            start: MediaTime::new(2, 1).unwrap(),
            end: MediaTime::new(3, 1).unwrap(),
            duration_ns: 1,
        });
        // Subject 2 only entrance
        index.zone_stays.push(ZoneStay {
            zone_id: ZoneId(1),
            subject_id: Some(SubjectId(2)),
            track_id: None,
            start: MediaTime::new(0, 1).unwrap(),
            end: MediaTime::new(1, 1).unwrap(),
            duration_ns: 1,
        });

        let hits = execute_subject_query(
            &index,
            &SubjectQuery::new().then_seen_in(ZoneId(1), ZoneId(2), 10_000_000_000),
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].subject_id, SubjectId(1));
    }

    #[test]
    fn route_contains_subsequence_works() {
        assert!(route_contains_subsequence(
            &[ZoneId(1), ZoneId(2), ZoneId(3)],
            &[ZoneId(2), ZoneId(3)]
        ));
        assert!(!route_contains_subsequence(
            &[ZoneId(1), ZoneId(3)],
            &[ZoneId(1), ZoneId(2)]
        ));
    }
}
