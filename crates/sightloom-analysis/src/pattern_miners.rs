//! Pattern mining algorithms over timed / dwell / route / pair series.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

extern crate alloc;

use crate::input::{AnalysisSeries, DurationSample, PairSample, RouteSample, TimedSubjectEvent};
use crate::pattern::{PatternKind, PatternRecord};
use crate::stats::{day_of_week_ns, hour_of_day_ns, mean, median, stddev};
use alloc::{vec, vec::Vec};
use sightloom_core::{EventId, PatternId, SubjectId, ZoneId};

/// Mines all built-in patterns and returns them with sequential ids from `next_id`.
#[must_use]
pub fn mine_patterns(series: &AnalysisSeries, next_id: &mut u64) -> Vec<PatternRecord> {
    let mut out = Vec::new();
    out.extend(mine_time_of_day(&series.timed, next_id));
    out.extend(mine_day_of_week(&series.timed, next_id));
    out.extend(mine_visit_periodicity(&series.timed, next_id));
    out.extend(mine_dwell_distribution(&series.durations, next_id));
    out.extend(mine_route_sequences(&series.routes, next_id));
    out.extend(mine_co_occurrence(&series.pairs, next_id));
    out.extend(mine_expected_absence(&series.timed, next_id));
    out.extend(mine_group_formation(&series.pairs, next_id));
    out
}

/// Time-of-day concentration per subject (and global when subject is missing).
#[must_use]
pub fn mine_time_of_day(events: &[TimedSubjectEvent], next_id: &mut u64) -> Vec<PatternRecord> {
    mine_bucketed(events, next_id, PatternKind::TimeOfDay, |e| {
        u32::from(hour_of_day_ns(e.at_ns))
    })
}

/// Day-of-week concentration per subject.
#[must_use]
pub fn mine_day_of_week(events: &[TimedSubjectEvent], next_id: &mut u64) -> Vec<PatternRecord> {
    mine_bucketed(events, next_id, PatternKind::DayOfWeek, |e| {
        u32::from(day_of_week_ns(e.at_ns))
    })
}

/// Visit periodicity from inter-arrival gaps (median gap + regularity).
#[must_use]
pub fn mine_visit_periodicity(
    events: &[TimedSubjectEvent],
    next_id: &mut u64,
) -> Vec<PatternRecord> {
    let mut out = Vec::new();
    for (subject, times) in group_times(events) {
        if times.len() < 3 {
            continue;
        }
        let mut gaps = Vec::with_capacity(times.len() - 1);
        for window in times.windows(2) {
            let gap = (window[1] - window[0]) as f32;
            if gap > 0.0 {
                gaps.push(gap);
            }
        }
        if gaps.len() < 2 {
            continue;
        }
        let mut scratch = vec![0.0_f32; gaps.len()];
        let Some(med) = median(&gaps, &mut scratch) else {
            continue;
        };
        let Some(sd) = stddev(&gaps) else {
            continue;
        };
        // High confidence when gaps are tight relative to the median period.
        let confidence = if med <= 0.0 {
            0.0
        } else {
            (1.0 - (sd / med).clamp(0.0, 1.0)).clamp(0.0, 1.0)
        };
        if confidence < 0.35 {
            continue;
        }
        out.push(push_pattern(
            next_id,
            PatternKind::VisitPeriodicity,
            subject,
            confidence,
            evidence_from_timed(events, subject),
            med as u32,
        ));
    }
    out
}

/// Dwell distribution summary (mean duration) as a pattern per subject.
#[must_use]
pub fn mine_dwell_distribution(
    samples: &[DurationSample],
    next_id: &mut u64,
) -> Vec<PatternRecord> {
    let mut out = Vec::new();
    for (subject, values) in group_durations(samples) {
        if values.len() < 2 {
            continue;
        }
        let Some(mu) = mean(&values) else {
            continue;
        };
        let Some(sd) = stddev(&values) else {
            continue;
        };
        let confidence = if mu <= 0.0 {
            0.0
        } else {
            (1.0 - (sd / mu).clamp(0.0, 1.0)).clamp(0.0, 1.0)
        };
        out.push(push_pattern(
            next_id,
            PatternKind::DwellDistribution,
            subject,
            confidence,
            evidence_from_durations(samples, subject),
            mu as u32,
        ));
    }
    out
}

/// Most frequent route n-gram (full sequence hash as tag).
#[must_use]
pub fn mine_route_sequences(routes: &[RouteSample], next_id: &mut u64) -> Vec<PatternRecord> {
    let mut out = Vec::new();
    // Count (subject, sequence key) frequency.
    let mut keys: Vec<(Option<SubjectId>, u32, usize, Option<EventId>)> = Vec::new();
    for route in routes {
        if route.zones.is_empty() {
            continue;
        }
        let key = route_key(&route.zones);
        keys.push((Some(route.subject_id), key, 1, route.event_id));
    }
    // Aggregate counts
    let mut agg: Vec<(Option<SubjectId>, u32, u32, Vec<EventId>)> = Vec::new();
    for (subject, key, _, event_id) in keys {
        if let Some(slot) = agg
            .iter_mut()
            .find(|(s, k, _, _)| *s == subject && *k == key)
        {
            slot.2 = slot.2.saturating_add(1);
            if let Some(id) = event_id {
                slot.3.push(id);
            }
        } else {
            let mut evidence = Vec::new();
            if let Some(id) = event_id {
                evidence.push(id);
            }
            agg.push((subject, key, 1, evidence));
        }
    }
    let total = routes.len().max(1) as f32;
    for (subject, key, count, evidence) in agg {
        if count < 2 {
            continue;
        }
        let confidence = (count as f32 / total).clamp(0.0, 1.0);
        out.push(push_pattern(
            next_id,
            PatternKind::RouteSequence,
            subject,
            confidence,
            evidence,
            key,
        ));
    }
    out
}

/// Frequent co-occurring subject pairs.
#[must_use]
pub fn mine_co_occurrence(pairs: &[PairSample], next_id: &mut u64) -> Vec<PatternRecord> {
    let mut out = Vec::new();
    let mut agg: Vec<((SubjectId, SubjectId), u32, Vec<EventId>)> = Vec::new();
    for pair in pairs {
        let key = ordered_pair(pair.subject_a, pair.subject_b);
        if let Some(slot) = agg.iter_mut().find(|(k, _, _)| *k == key) {
            slot.1 = slot.1.saturating_add(1);
            if let Some(id) = pair.event_id {
                slot.2.push(id);
            }
        } else {
            let mut evidence = Vec::new();
            if let Some(id) = pair.event_id {
                evidence.push(id);
            }
            agg.push((key, 1, evidence));
        }
    }
    let total = pairs.len().max(1) as f32;
    for ((a, b), count, evidence) in agg {
        if count < 2 {
            continue;
        }
        let confidence = (count as f32 / total).clamp(0.0, 1.0);
        // Tag packs both subject ids into a u32 when they fit; otherwise hash mix.
        let tag = pack_pair_tag(a, b);
        out.push(push_pattern(
            next_id,
            PatternKind::CoOccurrence,
            Some(a),
            confidence,
            evidence,
            tag,
        ));
        let _ = b;
    }
    out
}

/// Expected absence: long median gap is stored as the expected quiet period.
#[must_use]
pub fn mine_expected_absence(
    events: &[TimedSubjectEvent],
    next_id: &mut u64,
) -> Vec<PatternRecord> {
    // Reuse periodicity gaps; high median gap with moderate regularity.
    let mut out = Vec::new();
    for (subject, times) in group_times(events) {
        if times.len() < 3 {
            continue;
        }
        let mut gaps = Vec::new();
        for window in times.windows(2) {
            let gap = (window[1] - window[0]) as f32;
            if gap > 0.0 {
                gaps.push(gap);
            }
        }
        if gaps.len() < 2 {
            continue;
        }
        let mut scratch = vec![0.0_f32; gaps.len()];
        let Some(med) = median(&gaps, &mut scratch) else {
            continue;
        };
        // Only emit when quiet periods are meaningful (> 1 hour in ns).
        if med < 3_600_000_000_000.0 {
            continue;
        }
        let Some(sd) = stddev(&gaps) else {
            continue;
        };
        let confidence = if med <= 0.0 {
            0.0
        } else {
            (1.0 - (sd / med).clamp(0.0, 1.0)).clamp(0.0, 1.0)
        };
        out.push(push_pattern(
            next_id,
            PatternKind::ExpectedAbsence,
            subject,
            confidence,
            evidence_from_timed(events, subject),
            (med / 1_000_000_000.0) as u32, // seconds
        ));
    }
    out
}

/// Group formation: subjects that co-occur with many peers.
#[must_use]
pub fn mine_group_formation(pairs: &[PairSample], next_id: &mut u64) -> Vec<PatternRecord> {
    let mut out = Vec::new();
    let mut degree: Vec<(SubjectId, u32, Vec<EventId>)> = Vec::new();
    for pair in pairs {
        for subject in [pair.subject_a, pair.subject_b] {
            if let Some(slot) = degree.iter_mut().find(|(s, _, _)| *s == subject) {
                slot.1 = slot.1.saturating_add(1);
                if let Some(id) = pair.event_id {
                    slot.2.push(id);
                }
            } else {
                let mut evidence = Vec::new();
                if let Some(id) = pair.event_id {
                    evidence.push(id);
                }
                degree.push((subject, 1, evidence));
            }
        }
    }
    for (subject, count, evidence) in degree {
        if count < 3 {
            continue;
        }
        let confidence = (count as f32 / 10.0).clamp(0.0, 1.0);
        out.push(push_pattern(
            next_id,
            PatternKind::GroupFormation,
            Some(subject),
            confidence,
            evidence,
            count,
        ));
    }
    out
}

fn mine_bucketed(
    events: &[TimedSubjectEvent],
    next_id: &mut u64,
    kind: PatternKind,
    bucket_fn: impl Fn(&TimedSubjectEvent) -> u32,
) -> Vec<PatternRecord> {
    let mut out = Vec::new();
    // (subject, bucket) counts
    let mut counts: Vec<(Option<SubjectId>, u32, u32, Vec<EventId>)> = Vec::new();
    for event in events {
        let bucket = bucket_fn(event);
        if let Some(slot) = counts
            .iter_mut()
            .find(|(s, b, _, _)| *s == event.subject_id && *b == bucket)
        {
            slot.2 = slot.2.saturating_add(1);
            if let Some(id) = event.event_id {
                slot.3.push(id);
            }
        } else {
            let mut evidence = Vec::new();
            if let Some(id) = event.event_id {
                evidence.push(id);
            }
            counts.push((event.subject_id, bucket, 1, evidence));
        }
    }
    // Per subject pick the dominant bucket.
    let subjects: Vec<Option<SubjectId>> = {
        let mut s = Vec::new();
        for (subject, _, _, _) in &counts {
            if !s.contains(subject) {
                s.push(*subject);
            }
        }
        s
    };
    for subject in subjects {
        let mut best: Option<(u32, u32, Vec<EventId>)> = None;
        let mut total = 0_u32;
        for (s, bucket, count, evidence) in &counts {
            if *s != subject {
                continue;
            }
            total = total.saturating_add(*count);
            match &best {
                Some((_, bc, _)) if *count > *bc => {
                    best = Some((*bucket, *count, evidence.clone()))
                }
                None => best = Some((*bucket, *count, evidence.clone())),
                _ => {}
            }
        }
        let Some((bucket, count, evidence)) = best else {
            continue;
        };
        if total < 3 || count < 2 {
            continue;
        }
        let confidence = count as f32 / total as f32;
        out.push(push_pattern(
            next_id, kind, subject, confidence, evidence, bucket,
        ));
    }
    out
}

fn group_times(events: &[TimedSubjectEvent]) -> Vec<(Option<SubjectId>, Vec<i64>)> {
    let mut groups: Vec<(Option<SubjectId>, Vec<i64>)> = Vec::new();
    for event in events {
        if let Some(slot) = groups.iter_mut().find(|(s, _)| *s == event.subject_id) {
            slot.1.push(event.at_ns);
        } else {
            groups.push((event.subject_id, alloc::vec![event.at_ns]));
        }
    }
    for (_, times) in &mut groups {
        times.sort_unstable();
    }
    groups
}

fn group_durations(samples: &[DurationSample]) -> Vec<(Option<SubjectId>, Vec<f32>)> {
    let mut groups: Vec<(Option<SubjectId>, Vec<f32>)> = Vec::new();
    for sample in samples {
        let value = sample.duration_ns as f32;
        if value <= 0.0 {
            continue;
        }
        if let Some(slot) = groups.iter_mut().find(|(s, _)| *s == sample.subject_id) {
            slot.1.push(value);
        } else {
            groups.push((sample.subject_id, alloc::vec![value]));
        }
    }
    groups
}

fn evidence_from_timed(events: &[TimedSubjectEvent], subject: Option<SubjectId>) -> Vec<EventId> {
    events
        .iter()
        .filter(|e| e.subject_id == subject)
        .filter_map(|e| e.event_id)
        .collect()
}

fn evidence_from_durations(samples: &[DurationSample], subject: Option<SubjectId>) -> Vec<EventId> {
    samples
        .iter()
        .filter(|s| s.subject_id == subject)
        .filter_map(|s| s.event_id)
        .collect()
}

fn push_pattern(
    next_id: &mut u64,
    kind: PatternKind,
    subject_id: Option<SubjectId>,
    confidence: f32,
    evidence_events: Vec<EventId>,
    tag: u32,
) -> PatternRecord {
    let id = PatternId(*next_id);
    *next_id = next_id.saturating_add(1);
    PatternRecord {
        pattern_id: id,
        kind,
        subject_id,
        confidence,
        evidence_events,
        tag,
    }
}

fn route_key(zones: &[ZoneId]) -> u32 {
    let mut hash = 2_166_136_261_u32;
    for zone in zones {
        hash ^= u32::from(zone.0);
        hash = hash.wrapping_mul(16_777_619);
    }
    hash
}

fn ordered_pair(a: SubjectId, b: SubjectId) -> (SubjectId, SubjectId) {
    if a.0 <= b.0 { (a, b) } else { (b, a) }
}

fn pack_pair_tag(a: SubjectId, b: SubjectId) -> u32 {
    let lo = (a.0 & 0xffff) as u32;
    let hi = (b.0 & 0xffff) as u32;
    (hi << 16) | lo
}
