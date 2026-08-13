//! Composable boolean query AST over subjects (foundation).
//!
//! Compiles to [`crate::SubjectQuery`] where possible; supports AND/OR/NOT trees
//! for hosts that need boolean composition without a full planner/NL bridge.

use crate::{
    Page, QueryOrder, SpatialQuery, SubjectHit, SubjectQuery, ThenSeenIn, VisionIndex,
    execute_spatial_query, execute_subject_query,
};
use sightloom_core::{MediaTime, SourceId, SubjectId, ZoneId};

/// Atomic subject predicate.
#[derive(Clone, Debug, PartialEq)]
pub enum SubjectPredicate {
    /// Restrict to subject ids.
    Subjects(Vec<SubjectId>),
    /// Seen on source.
    SeenOn(SourceId),
    /// Zone stay.
    SeenInZone(ZoneId),
    /// Then-seen-in chain.
    ThenSeenIn(ThenSeenIn),
    /// Route contains zone subsequence.
    RouteContains(Vec<ZoneId>),
    /// Time window.
    During {
        /// Start.
        start: MediaTime,
        /// End.
        end: MediaTime,
    },
    /// Min dwell.
    MinDwellNs(i64),
    /// Min confidence.
    MinConfidence(f32),
    /// Spatial region (intersects any sample box).
    Spatial(SpatialQuery),
}

/// Boolean query tree.
#[derive(Clone, Debug, PartialEq)]
pub enum QueryNode {
    /// All children must match.
    And(Vec<QueryNode>),
    /// Any child matches.
    Or(Vec<QueryNode>),
    /// Negation (subjects that do **not** match the child).
    Not(Box<QueryNode>),
    /// Leaf predicate.
    Pred(SubjectPredicate),
}

impl QueryNode {
    /// Leaf helper.
    #[must_use]
    pub fn pred(p: SubjectPredicate) -> Self {
        Self::Pred(p)
    }

    /// AND helper.
    #[must_use]
    pub fn and(nodes: impl Into<Vec<QueryNode>>) -> Self {
        Self::And(nodes.into())
    }

    /// OR helper.
    #[must_use]
    pub fn or(nodes: impl Into<Vec<QueryNode>>) -> Self {
        Self::Or(nodes.into())
    }

    /// NOT helper (named to avoid clashing with `std::ops::Not`).
    #[must_use]
    pub fn negate(node: QueryNode) -> Self {
        Self::Not(Box::new(node))
    }
}

/// Executes a boolean query AST, returning subject hits.
///
/// Implementation: evaluates leaves to subject-id sets, then composes with
/// set algebra, and finally materializes hits via a subject-id filter query.
#[must_use]
pub fn execute_query_ast(index: &VisionIndex, root: &QueryNode) -> Vec<SubjectHit> {
    let universe = all_subject_ids(index);
    let matched = eval_node(index, root, &universe);
    if matched.is_empty() {
        return Vec::new();
    }
    let mut q = SubjectQuery::new();
    q.subject_ids = matched;
    q.order = QueryOrder::SubjectIdAsc;
    q.page = Page::default();
    execute_subject_query(index, &q)
}

fn all_subject_ids(index: &VisionIndex) -> Vec<SubjectId> {
    let mut ids = Vec::new();
    for sample in index.tracks.effective_samples() {
        if let Some(id) = sample.subject_id
            && !ids.contains(&id)
        {
            ids.push(id);
        }
    }
    ids.sort_by_key(|id| id.0);
    ids
}

fn eval_node(index: &VisionIndex, node: &QueryNode, universe: &[SubjectId]) -> Vec<SubjectId> {
    match node {
        QueryNode::And(children) => {
            let mut acc: Option<Vec<SubjectId>> = None;
            for child in children {
                let set = eval_node(index, child, universe);
                acc = Some(match acc {
                    None => set,
                    Some(prev) => intersect(&prev, &set),
                });
            }
            acc.unwrap_or_else(|| universe.to_vec())
        }
        QueryNode::Or(children) => {
            let mut acc = Vec::new();
            for child in children {
                for id in eval_node(index, child, universe) {
                    if !acc.contains(&id) {
                        acc.push(id);
                    }
                }
            }
            acc.sort_by_key(|id| id.0);
            acc
        }
        QueryNode::Not(child) => {
            let inner = eval_node(index, child, universe);
            universe
                .iter()
                .copied()
                .filter(|id| !inner.contains(id))
                .collect()
        }
        QueryNode::Pred(p) => eval_pred(index, p),
    }
}

fn eval_pred(index: &VisionIndex, pred: &SubjectPredicate) -> Vec<SubjectId> {
    match pred {
        SubjectPredicate::Subjects(ids) => {
            let mut out = ids.clone();
            out.sort_by_key(|id| id.0);
            out.dedup();
            out
        }
        SubjectPredicate::SeenOn(source) => subject_ids_from_hits(execute_subject_query(
            index,
            &SubjectQuery::new().seen_on(*source),
        )),
        SubjectPredicate::SeenInZone(zone) => subject_ids_from_hits(execute_subject_query(
            index,
            &SubjectQuery::new().seen_in(*zone),
        )),
        SubjectPredicate::ThenSeenIn(chain) => subject_ids_from_hits(execute_subject_query(
            index,
            &SubjectQuery::new().then_seen_in(chain.first, chain.then, chain.within_ns),
        )),
        SubjectPredicate::RouteContains(zones) => subject_ids_from_hits(execute_subject_query(
            index,
            &SubjectQuery::new().route_contains(zones.clone()),
        )),
        SubjectPredicate::During { start, end } => subject_ids_from_hits(execute_subject_query(
            index,
            &SubjectQuery::new().during(*start, *end),
        )),
        SubjectPredicate::MinDwellNs(ns) => subject_ids_from_hits(execute_subject_query(
            index,
            &SubjectQuery::new().with_min_dwell_ns(*ns),
        )),
        SubjectPredicate::MinConfidence(c) => subject_ids_from_hits(execute_subject_query(
            index,
            &SubjectQuery::new().with_min_confidence(*c),
        )),
        SubjectPredicate::Spatial(spatial) => {
            let hits = execute_spatial_query(index, spatial);
            let mut ids = Vec::new();
            for h in hits {
                if let Some(id) = h.subject_id
                    && !ids.contains(&id)
                {
                    ids.push(id);
                }
            }
            ids.sort_by_key(|id| id.0);
            ids
        }
    }
}

fn subject_ids_from_hits(hits: Vec<SubjectHit>) -> Vec<SubjectId> {
    let mut ids: Vec<SubjectId> = hits.into_iter().map(|h| h.subject_id).collect();
    ids.sort_by_key(|id| id.0);
    ids.dedup();
    ids
}

fn intersect(a: &[SubjectId], b: &[SubjectId]) -> Vec<SubjectId> {
    a.iter().copied().filter(|id| b.contains(id)).collect()
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
    fn and_or_not_compose() {
        let mut index = VisionIndex::new("ast");
        index.push_track(sample(1, 0, 1));
        index.push_track(sample(2, 0, 1));
        index.push_track(sample(3, 0, 2));

        let on1 = QueryNode::pred(SubjectPredicate::SeenOn(SourceId(1)));
        let only1 = QueryNode::and(vec![
            on1.clone(),
            QueryNode::pred(SubjectPredicate::Subjects(vec![SubjectId(1)])),
        ]);
        let hits = execute_query_ast(&index, &only1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].subject_id, SubjectId(1));

        let not_on1 = QueryNode::negate(on1);
        let hits2 = execute_query_ast(&index, &not_on1);
        assert_eq!(hits2.len(), 1);
        assert_eq!(hits2[0].subject_id, SubjectId(3));
    }
}
