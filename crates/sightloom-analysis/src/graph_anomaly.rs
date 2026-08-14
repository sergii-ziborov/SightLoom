//! Graph / multi-camera **relational** anomaly backend.
//!
//! Pure-Rust rules over host-prepared [`AnalysisSeries`] + optional
//! [`CameraGraph`] travel constraints. Complements statistical / iForest /
//! OCSVM backends without depending on `sightloom-reid` (facade maps topology).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

extern crate alloc;

use crate::anomaly::{AnomalyEvent, AnomalyReason, Severity};
use crate::anomaly_backend::AnomalyDetector;
use crate::input::{AnalysisSeries, PairSample, RouteSample, TimedSubjectEvent};
use alloc::{vec, vec::Vec};
use sightloom_core::{AnomalyId, EventId, MediaTime, SourceId, SubjectId, ZoneId};

/// Directed camera hop constraint (analysis-local; mirrors re-id topology shape).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphEdge {
    /// From camera.
    pub from: SourceId,
    /// To camera.
    pub to: SourceId,
    /// Minimum plausible travel time (nanoseconds).
    pub min_travel_ns: i64,
    /// Optional maximum travel time.
    pub max_travel_ns: Option<i64>,
}

/// Sparse camera graph used for impossible-hop checks.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CameraGraph {
    edges: Vec<GraphEdge>,
    /// When true, unknown edges are impossible (strict topology).
    pub strict_unknown: bool,
}

impl CameraGraph {
    /// Empty graph.
    #[must_use]
    pub const fn new(strict_unknown: bool) -> Self {
        Self {
            edges: Vec::new(),
            strict_unknown,
        }
    }

    /// Adds or replaces a directed edge.
    pub fn set_edge_window(
        &mut self,
        from: SourceId,
        to: SourceId,
        min_travel_ns: i64,
        max_travel_ns: Option<i64>,
    ) {
        if let Some(edge) = self.edges.iter_mut().find(|e| e.from == from && e.to == to) {
            edge.min_travel_ns = min_travel_ns;
            edge.max_travel_ns = max_travel_ns;
        } else {
            self.edges.push(GraphEdge {
                from,
                to,
                min_travel_ns,
                max_travel_ns,
            });
        }
    }

    /// Bidirectional edges with the same window.
    pub fn set_bidirectional_window(
        &mut self,
        a: SourceId,
        b: SourceId,
        min_travel_ns: i64,
        max_travel_ns: Option<i64>,
    ) {
        self.set_edge_window(a, b, min_travel_ns, max_travel_ns);
        self.set_edge_window(b, a, min_travel_ns, max_travel_ns);
    }

    /// Whether a hop is physically allowed.
    #[must_use]
    pub fn allows_hop(&self, from: SourceId, to: SourceId, elapsed_ns: i64) -> bool {
        if from == to {
            return true;
        }
        match self.edges.iter().find(|e| e.from == from && e.to == to) {
            Some(edge) if elapsed_ns < edge.min_travel_ns => false,
            Some(edge) if edge.max_travel_ns.is_some_and(|max| elapsed_ns > max) => false,
            None if self.strict_unknown => false,
            Some(_) | None => true,
        }
    }

    /// Edges in insertion order.
    #[must_use]
    pub fn edges(&self) -> &[GraphEdge] {
        &self.edges
    }
}

/// Configuration for relational / graph anomaly detection.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct GraphAnomalyConfig {
    /// Minimum hop samples before rare-hop scoring.
    pub min_hop_samples: usize,
    /// Baseline hop fraction below this is "rare".
    pub rare_hop_fraction: f32,
    /// Minimum pair samples before rare-pair scoring.
    pub min_pair_samples: usize,
    /// Baseline pair fraction below this is rare.
    pub rare_pair_fraction: f32,
    /// Minimum route samples before rare-route scoring.
    pub min_route_samples: usize,
    /// Baseline route fraction below this is rare.
    pub rare_route_fraction: f32,
    /// Flag topology-impossible cross-camera hops.
    pub flag_impossible_hops: bool,
    /// Flag rare camera transitions vs baseline.
    pub flag_rare_hops: bool,
    /// Flag rare co-occurrence pairs.
    pub flag_rare_pairs: bool,
    /// Flag rare zone routes.
    pub flag_rare_routes: bool,
}

impl Default for GraphAnomalyConfig {
    fn default() -> Self {
        Self {
            min_hop_samples: 8,
            rare_hop_fraction: 0.05,
            min_pair_samples: 6,
            rare_pair_fraction: 0.05,
            min_route_samples: 6,
            rare_route_fraction: 0.05,
            flag_impossible_hops: true,
            flag_rare_hops: true,
            flag_rare_pairs: true,
            flag_rare_routes: true,
        }
    }
}

/// One observed subject hop between cameras.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CameraHop {
    /// Subject when known.
    pub subject_id: Option<SubjectId>,
    /// From camera.
    pub from: SourceId,
    /// To camera.
    pub to: SourceId,
    /// Elapsed nanoseconds between sightings.
    pub elapsed_ns: i64,
    /// Arrival time.
    pub at_ns: i64,
    /// Optional evidence at destination.
    pub event_id: Option<EventId>,
}

/// Relational baseline histograms.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GraphBaseline {
    /// `(from, to, count)` directed hop histogram.
    pub hop_counts: Vec<(SourceId, SourceId, u32)>,
    /// Total hops.
    pub hop_n: usize,
    /// Normalized co-occurrence pairs `(a, b, count)` with `a.0 <= b.0`.
    pub pair_counts: Vec<(SubjectId, SubjectId, u32)>,
    /// Total pairs.
    pub pair_n: usize,
    /// Zone route keys as sorted zone id lists + count.
    pub route_counts: Vec<(Vec<ZoneId>, u32)>,
    /// Total routes.
    pub route_n: usize,
}

impl GraphBaseline {
    /// Fraction of hops on `from → to`.
    #[must_use]
    pub fn hop_fraction(&self, from: SourceId, to: SourceId) -> f32 {
        if self.hop_n == 0 {
            return 0.0;
        }
        let c = self
            .hop_counts
            .iter()
            .find(|(a, b, _)| *a == from && *b == to)
            .map_or(0, |(_, _, n)| *n);
        c as f32 / self.hop_n as f32
    }

    /// Fraction of co-occurrence pairs.
    #[must_use]
    pub fn pair_fraction(&self, a: SubjectId, b: SubjectId) -> f32 {
        if self.pair_n == 0 {
            return 0.0;
        }
        let (lo, hi) = order_subjects(a, b);
        let c = self
            .pair_counts
            .iter()
            .find(|(x, y, _)| *x == lo && *y == hi)
            .map_or(0, |(_, _, n)| *n);
        c as f32 / self.pair_n as f32
    }

    /// Fraction of identical zone routes.
    #[must_use]
    pub fn route_fraction(&self, zones: &[ZoneId]) -> f32 {
        if self.route_n == 0 {
            return 0.0;
        }
        let c = self
            .route_counts
            .iter()
            .find(|(z, _)| z.as_slice() == zones)
            .map_or(0, |(_, n)| *n);
        c as f32 / self.route_n as f32
    }
}

/// Extracts consecutive cross-camera hops from timed subject events.
#[must_use]
pub fn extract_camera_hops(events: &[TimedSubjectEvent]) -> Vec<CameraHop> {
    let mut by_subject: Vec<(Option<SubjectId>, Vec<TimedSubjectEvent>)> = Vec::new();
    for event in events {
        if event.source_id.is_none() {
            continue;
        }
        if let Some((_, list)) = by_subject
            .iter_mut()
            .find(|(sid, _)| *sid == event.subject_id)
        {
            list.push(*event);
        } else {
            by_subject.push((event.subject_id, vec![*event]));
        }
    }

    let mut hops = Vec::new();
    for (subject_id, mut list) in by_subject {
        list.sort_by_key(|e| e.at_ns);
        for window in list.windows(2) {
            let a = window[0];
            let b = window[1];
            let (Some(from), Some(to)) = (a.source_id, b.source_id) else {
                continue;
            };
            if from == to {
                continue;
            }
            let elapsed = b.at_ns.saturating_sub(a.at_ns);
            if elapsed <= 0 {
                continue;
            }
            hops.push(CameraHop {
                subject_id,
                from,
                to,
                elapsed_ns: elapsed,
                at_ns: b.at_ns,
                event_id: b.event_id,
            });
        }
    }
    hops
}

/// Builds relational histograms from a history series.
#[must_use]
pub fn build_graph_baseline(series: &AnalysisSeries) -> GraphBaseline {
    let mut baseline = GraphBaseline::default();

    for hop in extract_camera_hops(&series.timed) {
        bump_hop(&mut baseline.hop_counts, hop.from, hop.to);
        baseline.hop_n = baseline.hop_n.saturating_add(1);
    }

    for pair in &series.pairs {
        let (lo, hi) = order_subjects(pair.subject_a, pair.subject_b);
        bump_pair(&mut baseline.pair_counts, lo, hi);
        baseline.pair_n = baseline.pair_n.saturating_add(1);
    }

    for route in &series.routes {
        bump_route(&mut baseline.route_counts, &route.zones);
        baseline.route_n = baseline.route_n.saturating_add(1);
    }

    baseline
}

/// Detects relational anomalies in `live` against `baseline` + optional graph.
#[must_use]
pub fn detect_graph_anomalies(
    live: &AnalysisSeries,
    baseline: &GraphBaseline,
    graph: &CameraGraph,
    config: GraphAnomalyConfig,
    next_id: &mut u64,
) -> Vec<AnomalyEvent> {
    let mut out = Vec::new();
    if config.flag_impossible_hops || config.flag_rare_hops {
        out.extend(detect_hop_anomalies(
            &extract_camera_hops(&live.timed),
            baseline,
            graph,
            config,
            next_id,
        ));
    }
    if config.flag_rare_pairs {
        out.extend(detect_pair_anomalies(
            &live.pairs,
            baseline,
            config,
            next_id,
        ));
    }
    if config.flag_rare_routes {
        out.extend(detect_route_anomalies(
            &live.routes,
            baseline,
            config,
            next_id,
        ));
    }
    out
}

fn detect_hop_anomalies(
    hops: &[CameraHop],
    baseline: &GraphBaseline,
    graph: &CameraGraph,
    config: GraphAnomalyConfig,
    next_id: &mut u64,
) -> Vec<AnomalyEvent> {
    let mut out = Vec::new();
    let rare = config.rare_hop_fraction.clamp(0.0, 0.5);
    for hop in hops {
        if config.flag_impossible_hops && !graph.allows_hop(hop.from, hop.to, hop.elapsed_ns) {
            out.push(make_event(
                next_id,
                8.0,
                AnomalyReason::ImpossibleCrossCameraHop,
                hop.subject_id,
                Some(hop.to),
                hop.event_id,
                hop.at_ns,
            ));
            continue;
        }
        if !config.flag_rare_hops || baseline.hop_n < config.min_hop_samples {
            continue;
        }
        let frac = baseline.hop_fraction(hop.from, hop.to);
        if frac > rare {
            continue;
        }
        // Never-seen or rare hop: score inverse rarity.
        let score = (1.0 / (frac + 1e-3)).min(20.0);
        out.push(make_event(
            next_id,
            score,
            AnomalyReason::UnusualCameraTransition,
            hop.subject_id,
            Some(hop.to),
            hop.event_id,
            hop.at_ns,
        ));
    }
    out
}

fn detect_pair_anomalies(
    pairs: &[PairSample],
    baseline: &GraphBaseline,
    config: GraphAnomalyConfig,
    next_id: &mut u64,
) -> Vec<AnomalyEvent> {
    if baseline.pair_n < config.min_pair_samples {
        return Vec::new();
    }
    let rare = config.rare_pair_fraction.clamp(0.0, 0.5);
    let mut out = Vec::new();
    for pair in pairs {
        let frac = baseline.pair_fraction(pair.subject_a, pair.subject_b);
        if frac > rare {
            continue;
        }
        let score = (1.0 / (frac + 1e-3)).min(20.0);
        out.push(make_event(
            next_id,
            score,
            AnomalyReason::UnusualCoOccurrence,
            Some(pair.subject_a),
            pair.source_id,
            pair.event_id,
            pair.at_ns,
        ));
    }
    out
}

fn detect_route_anomalies(
    routes: &[RouteSample],
    baseline: &GraphBaseline,
    config: GraphAnomalyConfig,
    next_id: &mut u64,
) -> Vec<AnomalyEvent> {
    if baseline.route_n < config.min_route_samples {
        return Vec::new();
    }
    let rare = config.rare_route_fraction.clamp(0.0, 0.5);
    let mut out = Vec::new();
    for route in routes {
        if route.zones.len() < 2 {
            continue;
        }
        let frac = baseline.route_fraction(&route.zones);
        if frac > rare {
            continue;
        }
        let score = (1.0 / (frac + 1e-3)).min(20.0);
        out.push(make_event(
            next_id,
            score,
            AnomalyReason::UnusualRoute,
            Some(route.subject_id),
            None,
            route.event_id,
            route.at_ns,
        ));
    }
    out
}

fn make_event(
    next_id: &mut u64,
    score: f32,
    reason: AnomalyReason,
    subject_id: Option<SubjectId>,
    source_id: Option<SourceId>,
    event_id: Option<EventId>,
    at_ns: i64,
) -> AnomalyEvent {
    let id = AnomalyId(*next_id);
    *next_id = next_id.saturating_add(1);
    let mut evidence = Vec::new();
    if let Some(event_id) = event_id {
        evidence.push(event_id);
    }
    let severity = if score >= 10.0 {
        Severity::Critical
    } else if score >= 6.0 {
        Severity::High
    } else if score >= 3.5 {
        Severity::Medium
    } else {
        Severity::Low
    };
    AnomalyEvent {
        anomaly_id: id,
        score: if score.is_finite() { score } else { 100.0 },
        severity,
        reasons: vec![reason],
        evidence,
        subject_id,
        source_id,
        at: MediaTime::new(at_ns, 1_000_000_000).unwrap_or_default(),
    }
}

fn order_subjects(a: SubjectId, b: SubjectId) -> (SubjectId, SubjectId) {
    if a.0 <= b.0 {
        (a, b)
    } else {
        (b, a)
    }
}

fn bump_hop(counts: &mut Vec<(SourceId, SourceId, u32)>, from: SourceId, to: SourceId) {
    if let Some(slot) = counts.iter_mut().find(|(a, b, _)| *a == from && *b == to) {
        slot.2 = slot.2.saturating_add(1);
    } else {
        counts.push((from, to, 1));
    }
}

fn bump_pair(counts: &mut Vec<(SubjectId, SubjectId, u32)>, a: SubjectId, b: SubjectId) {
    if let Some(slot) = counts.iter_mut().find(|(x, y, _)| *x == a && *y == b) {
        slot.2 = slot.2.saturating_add(1);
    } else {
        counts.push((a, b, 1));
    }
}

fn bump_route(counts: &mut Vec<(Vec<ZoneId>, u32)>, zones: &[ZoneId]) {
    if let Some(slot) = counts.iter_mut().find(|(z, _)| z.as_slice() == zones) {
        slot.1 = slot.1.saturating_add(1);
    } else {
        counts.push((zones.to_vec(), 1));
    }
}

/// Graph / relational detector implementing [`AnomalyDetector`].
#[derive(Clone, Debug, Default)]
pub struct GraphRelationalDetector {
    /// Detection config.
    pub config: GraphAnomalyConfig,
    /// Camera travel graph (impossible hop checks).
    pub graph: CameraGraph,
    /// Fitted baseline.
    pub baseline: Option<GraphBaseline>,
}

impl GraphRelationalDetector {
    /// Creates with config and empty graph.
    #[must_use]
    pub fn new(config: GraphAnomalyConfig) -> Self {
        Self {
            config,
            graph: CameraGraph::new(true),
            baseline: None,
        }
    }

    /// Creates with config and camera graph.
    #[must_use]
    pub fn with_graph(config: GraphAnomalyConfig, graph: CameraGraph) -> Self {
        Self {
            config,
            graph,
            baseline: None,
        }
    }

    /// Replaces the camera graph.
    pub fn set_graph(&mut self, graph: CameraGraph) {
        self.graph = graph;
    }
}

impl AnomalyDetector for GraphRelationalDetector {
    type Error = &'static str;

    fn fit(&mut self, history: &AnalysisSeries) -> Result<(), Self::Error> {
        self.baseline = Some(build_graph_baseline(history));
        Ok(())
    }

    fn detect(
        &mut self,
        live: &AnalysisSeries,
        next_id: &mut u64,
    ) -> Result<Vec<AnomalyEvent>, Self::Error> {
        let baseline = self
            .baseline
            .clone()
            .unwrap_or_else(|| build_graph_baseline(live));
        Ok(detect_graph_anomalies(
            live,
            &baseline,
            &self.graph,
            self.config,
            next_id,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::AnalysisSeries;

    fn cam(id: u32) -> SourceId {
        SourceId(id)
    }

    fn sub(id: u64) -> SubjectId {
        SubjectId(id)
    }

    fn timed(subject: u64, source: u32, at_ns: i64) -> TimedSubjectEvent {
        TimedSubjectEvent {
            subject_id: Some(sub(subject)),
            source_id: Some(cam(source)),
            at_ns,
            event_id: Some(EventId(at_ns as u64)),
            kind_tag: 0,
        }
    }

    #[test]
    fn extract_hops_skips_same_camera() {
        let events = [
            timed(1, 1, 1_000),
            timed(1, 1, 2_000),
            timed(1, 2, 5_000),
        ];
        let hops = extract_camera_hops(&events);
        assert_eq!(hops.len(), 1);
        assert_eq!(hops[0].from, cam(1));
        assert_eq!(hops[0].to, cam(2));
        assert_eq!(hops[0].elapsed_ns, 3_000);
    }

    #[test]
    fn impossible_hop_flagged() {
        let mut graph = CameraGraph::new(true);
        // Cam1 → Cam2 requires at least 10s.
        graph.set_edge_window(cam(1), cam(2), 10_000_000_000, None);

        let mut history = AnalysisSeries::default();
        for i in 0..10 {
            history.timed.push(timed(1, 1, i * 20_000_000_000));
            history.timed.push(timed(1, 2, i * 20_000_000_000 + 15_000_000_000));
        }
        let baseline = build_graph_baseline(&history);

        let mut live = AnalysisSeries::default();
        // Too fast: 1s travel.
        live.timed.push(timed(1, 1, 100_000_000_000));
        live.timed.push(timed(1, 2, 101_000_000_000));

        let mut next_id = 1;
        let found = detect_graph_anomalies(
            &live,
            &baseline,
            &graph,
            GraphAnomalyConfig {
                flag_rare_hops: false,
                flag_rare_pairs: false,
                flag_rare_routes: false,
                ..GraphAnomalyConfig::default()
            },
            &mut next_id,
        );
        assert!(
            found
                .iter()
                .any(|a| a.reasons.contains(&AnomalyReason::ImpossibleCrossCameraHop)),
            "{found:?}"
        );
    }

    #[test]
    fn rare_hop_flagged_against_baseline() {
        let mut history = AnalysisSeries::default();
        // Baseline: only 1→2 hops (distinct subjects so no reverse hop).
        for i in 0_i64..12 {
            history.timed.push(timed(i as u64 + 1, 1, i * 10_000_000_000));
            history
                .timed
                .push(timed(i as u64 + 1, 2, i * 10_000_000_000 + 5_000_000_000));
        }
        let baseline = build_graph_baseline(&history);
        assert_eq!(baseline.hop_n, 12);
        assert!((baseline.hop_fraction(cam(1), cam(2)) - 1.0).abs() < 1e-5);

        let mut live = AnalysisSeries::default();
        // Novel 1→3 hop.
        live.timed.push(timed(1, 1, 200_000_000_000));
        live.timed.push(timed(1, 3, 205_000_000_000));

        let mut next_id = 1;
        let found = detect_graph_anomalies(
            &live,
            &baseline,
            &CameraGraph::new(false),
            GraphAnomalyConfig {
                flag_impossible_hops: false,
                flag_rare_pairs: false,
                flag_rare_routes: false,
                ..GraphAnomalyConfig::default()
            },
            &mut next_id,
        );
        assert!(
            found
                .iter()
                .any(|a| a.reasons.contains(&AnomalyReason::UnusualCameraTransition)),
            "{found:?}"
        );
    }

    #[test]
    fn rare_pair_and_route() {
        let mut history = AnalysisSeries::default();
        for i in 0..10 {
            history.pairs.push(PairSample {
                subject_a: sub(1),
                subject_b: sub(2),
                source_id: Some(cam(1)),
                at_ns: i * 1_000,
                event_id: None,
            });
            history.routes.push(RouteSample {
                subject_id: sub(1),
                zones: vec![ZoneId(1), ZoneId(2)],
                at_ns: i * 1_000,
                event_id: None,
            });
        }
        let baseline = build_graph_baseline(&history);

        let mut live = AnalysisSeries::default();
        live.pairs.push(PairSample {
            subject_a: sub(1),
            subject_b: sub(9),
            source_id: Some(cam(1)),
            at_ns: 99_000,
            event_id: None,
        });
        live.routes.push(RouteSample {
            subject_id: sub(1),
            zones: vec![ZoneId(9), ZoneId(8)],
            at_ns: 99_000,
            event_id: None,
        });

        let mut next_id = 1;
        let found = detect_graph_anomalies(
            &live,
            &baseline,
            &CameraGraph::new(false),
            GraphAnomalyConfig {
                flag_impossible_hops: false,
                flag_rare_hops: false,
                ..GraphAnomalyConfig::default()
            },
            &mut next_id,
        );
        assert!(
            found
                .iter()
                .any(|a| a.reasons.contains(&AnomalyReason::UnusualCoOccurrence))
        );
        assert!(
            found
                .iter()
                .any(|a| a.reasons.contains(&AnomalyReason::UnusualRoute))
        );
    }

    #[test]
    fn detector_trait_fit_detect() {
        let mut det = GraphRelationalDetector::new(GraphAnomalyConfig {
            flag_impossible_hops: false,
            flag_rare_pairs: false,
            flag_rare_routes: false,
            ..GraphAnomalyConfig::default()
        });
        let mut history = AnalysisSeries::default();
        for i in 0_i64..12 {
            history.timed.push(timed(i as u64 + 1, 1, i * 10_000_000_000));
            history
                .timed
                .push(timed(i as u64 + 1, 2, i * 10_000_000_000 + 5_000_000_000));
        }
        det.fit(&history).unwrap();
        let mut live = AnalysisSeries::default();
        live.timed.push(timed(1, 1, 500_000_000_000));
        live.timed.push(timed(1, 3, 505_000_000_000));
        let mut next_id = 1;
        let found = det.detect(&live, &mut next_id).unwrap();
        assert!(!found.is_empty());
    }
}
