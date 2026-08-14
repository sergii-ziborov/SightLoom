//! Statistical anomaly backend (z-score rules).
//!
//! Hosts must treat outputs as backend-neutral [`AnomalyEvent`] values.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

extern crate alloc;

use crate::anomaly::{AnomalyEvent, AnomalyReason, Severity};
use crate::input::{AnalysisSeries, DurationSample, TimedSubjectEvent};
use crate::stats::{
    change_point_cusum, day_of_week_ns, hour_of_day_ns, mad, mean, median, robust_z_score, stddev,
    z_score,
};
use alloc::{vec, vec::Vec};
use sightloom_core::{AnomalyId, EventId, MediaTime, SubjectId};

/// Configuration for the statistical anomaly detector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StatAnomalyConfig {
    /// Absolute z-score threshold to flag an anomaly.
    pub z_threshold: f32,
    /// Minimum samples required to build a baseline.
    pub min_samples: usize,
    /// When true, also run robust MAD scoring and CUSUM change-points.
    pub use_robust: bool,
    /// CUSUM change-point score threshold (unitless; smoke default).
    pub change_point_threshold: f32,
    /// When true, flag appearances on weekdays that were rare in baseline.
    pub use_day_of_week: bool,
    /// Baseline day fraction below this is "rare" (e.g. `0.08` = 8%).
    pub rare_day_fraction: f32,
}

impl Default for StatAnomalyConfig {
    fn default() -> Self {
        Self {
            z_threshold: 2.5,
            min_samples: 5,
            use_robust: true,
            change_point_threshold: 8.0,
            use_day_of_week: true,
            rare_day_fraction: 0.08,
        }
    }
}

/// Baseline statistics learned from historical series.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BaselineStats {
    /// Mean dwell duration nanoseconds.
    pub dwell_mean: Option<f32>,
    /// Stddev dwell duration nanoseconds.
    pub dwell_std: Option<f32>,
    /// Mean inter-arrival nanoseconds.
    pub gap_mean: Option<f32>,
    /// Stddev inter-arrival nanoseconds.
    pub gap_std: Option<f32>,
    /// Mean hour-of-day (0..24) for appearances.
    pub hour_mean: Option<f32>,
    /// Stddev hour-of-day.
    pub hour_std: Option<f32>,
    /// Sample counts used.
    pub dwell_n: usize,
    /// Gap sample count.
    pub gap_n: usize,
    /// Timed sample count.
    pub timed_n: usize,
    /// Robust dwell median (when enough samples).
    pub dwell_median: Option<f32>,
    /// Robust dwell MAD.
    pub dwell_mad: Option<f32>,
    /// Fraction of timed events per weekday `0..7` (epoch-aligned).
    pub day_fraction: [f32; 7],
    /// Timed events used for day-of-week fractions.
    pub day_n: usize,
}

/// Builds baseline stats from historical series.
#[must_use]
pub fn build_baseline(series: &AnalysisSeries, config: StatAnomalyConfig) -> BaselineStats {
    let mut stats = BaselineStats::default();

    let dwells: Vec<f32> = series
        .durations
        .iter()
        .map(|d| d.duration_ns as f32)
        .filter(|v| *v > 0.0 && v.is_finite())
        .collect();
    stats.dwell_n = dwells.len();
    if dwells.len() >= config.min_samples {
        stats.dwell_mean = mean(&dwells);
        stats.dwell_std = stddev(&dwells);
        let mut scratch = vec![0.0_f32; dwells.len()];
        stats.dwell_median = median(&dwells, &mut scratch);
        stats.dwell_mad = mad(&dwells, &mut scratch);
    }

    let mut gaps = Vec::new();
    for (_, times) in group_subject_times(&series.timed) {
        for window in times.windows(2) {
            let gap = (window[1] - window[0]) as f32;
            if gap > 0.0 {
                gaps.push(gap);
            }
        }
    }
    stats.gap_n = gaps.len();
    if gaps.len() >= config.min_samples {
        stats.gap_mean = mean(&gaps);
        stats.gap_std = stddev(&gaps);
    }

    let hours: Vec<f32> = series
        .timed
        .iter()
        .map(|e| f32::from(hour_of_day_ns(e.at_ns)))
        .collect();
    stats.timed_n = hours.len();
    if hours.len() >= config.min_samples {
        stats.hour_mean = mean(&hours);
        stats.hour_std = stddev(&hours);
    }

    // Day-of-week seasonality histogram.
    let mut day_counts = [0_u32; 7];
    for e in &series.timed {
        let d = day_of_week_ns(e.at_ns) as usize % 7;
        day_counts[d] = day_counts[d].saturating_add(1);
    }
    let day_n = day_counts.iter().sum::<u32>() as usize;
    stats.day_n = day_n;
    if day_n > 0 {
        let inv = 1.0 / day_n as f32;
        for (frac, count) in stats.day_fraction.iter_mut().zip(day_counts.iter()) {
            *frac = *count as f32 * inv;
        }
    }

    stats
}

/// Detects statistical anomalies in `series` against `baseline`.
#[must_use]
pub fn detect_statistical(
    series: &AnalysisSeries,
    baseline: &BaselineStats,
    config: StatAnomalyConfig,
    next_id: &mut u64,
) -> Vec<AnomalyEvent> {
    let mut out = Vec::new();
    out.extend(detect_dwell_anomalies(
        &series.durations,
        baseline,
        config,
        next_id,
    ));
    out.extend(detect_frequency_anomalies(
        &series.timed,
        baseline,
        config,
        next_id,
    ));
    out.extend(detect_time_anomalies(
        &series.timed,
        baseline,
        config,
        next_id,
    ));
    out.extend(detect_missing_expected(
        &series.timed,
        baseline,
        config,
        next_id,
    ));
    out.extend(detect_sudden_change(
        &series.durations,
        baseline,
        config,
        next_id,
    ));
    if config.use_robust {
        out.extend(detect_robust_dwell(
            &series.durations,
            baseline,
            config,
            next_id,
        ));
        out.extend(detect_sequence_change_points(
            &series.durations,
            config,
            next_id,
        ));
        out.extend(detect_subject_specific_gaps(&series.timed, config, next_id));
    }
    if config.use_day_of_week {
        out.extend(detect_day_of_week_anomalies(
            &series.timed,
            baseline,
            config,
            next_id,
        ));
    }
    out
}

/// Flags timed events on weekdays that were rare in the baseline seasonality.
fn detect_day_of_week_anomalies(
    events: &[TimedSubjectEvent],
    baseline: &BaselineStats,
    config: StatAnomalyConfig,
    next_id: &mut u64,
) -> Vec<AnomalyEvent> {
    if baseline.day_n < config.min_samples.saturating_mul(2) {
        return Vec::new();
    }
    let rare = config.rare_day_fraction.clamp(0.0, 0.5);
    let mut out = Vec::new();
    for event in events {
        let d = day_of_week_ns(event.at_ns) as usize % 7;
        let frac = baseline.day_fraction[d];
        if frac > rare {
            continue;
        }
        // Score: inverse rarity (higher when day was almost never seen).
        let score = 1.0 / (frac + 1e-3);
        // Map into z-like scale for severity helpers.
        let z = (score * 0.5).min(10.0);
        if z < config.z_threshold * 0.5 {
            // Only flag clearly rare days (fraction well below rare threshold).
            continue;
        }
        out.push(make_event(
            next_id,
            z,
            AnomalyReason::UnusualAppearanceTime,
            event.subject_id,
            event.event_id,
            event.at_ns,
        ));
    }
    out
}

fn detect_robust_dwell(
    samples: &[DurationSample],
    baseline: &BaselineStats,
    config: StatAnomalyConfig,
    next_id: &mut u64,
) -> Vec<AnomalyEvent> {
    let (Some(med), Some(mad_v)) = (baseline.dwell_median, baseline.dwell_mad) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for sample in samples {
        let value = sample.duration_ns as f32;
        let Some(z) = robust_z_score(value, med, mad_v) else {
            continue;
        };
        if z < config.z_threshold {
            continue;
        }
        out.push(make_event(
            next_id,
            z,
            AnomalyReason::UnusualDwell,
            sample.subject_id,
            sample.event_id,
            sample.at_ns,
        ));
    }
    out
}

fn detect_sequence_change_points(
    samples: &[DurationSample],
    config: StatAnomalyConfig,
    next_id: &mut u64,
) -> Vec<AnomalyEvent> {
    if samples.len() < config.min_samples.saturating_mul(2) {
        return Vec::new();
    }
    let values: Vec<f32> = samples.iter().map(|s| s.duration_ns as f32).collect();
    let Some((idx, score)) = change_point_cusum(&values) else {
        return Vec::new();
    };
    if score < config.change_point_threshold {
        return Vec::new();
    }
    let sample = samples[idx.min(samples.len().saturating_sub(1))];
    vec![make_event(
        next_id,
        score,
        AnomalyReason::SuddenBehaviourChange,
        sample.subject_id,
        sample.event_id,
        sample.at_ns,
    )]
}

/// Per-subject gap baselines (subject-specific frequency anomalies).
fn detect_subject_specific_gaps(
    events: &[TimedSubjectEvent],
    config: StatAnomalyConfig,
    next_id: &mut u64,
) -> Vec<AnomalyEvent> {
    let mut out = Vec::new();
    for (subject, times) in group_subject_times(events) {
        if times.len() < config.min_samples.saturating_add(1) {
            continue;
        }
        let mut gaps: Vec<f32> = Vec::new();
        for w in times.windows(2) {
            let g = (w[1] - w[0]) as f32;
            if g > 0.0 {
                gaps.push(g);
            }
        }
        if gaps.len() < config.min_samples {
            continue;
        }
        let mut scratch = vec![0.0_f32; gaps.len()];
        let Some(med) = median(&gaps, &mut scratch) else {
            continue;
        };
        let Some(mad_v) = mad(&gaps, &mut scratch) else {
            continue;
        };
        // Flag the latest gap if robust-outlier vs this subject's own history.
        if let Some(last) = gaps.last().copied() {
            let Some(z) = robust_z_score(last, med, mad_v) else {
                continue;
            };
            if z >= config.z_threshold {
                let at = *times.last().unwrap_or(&0);
                out.push(make_event(
                    next_id,
                    z,
                    AnomalyReason::UnusualFrequency,
                    subject,
                    None,
                    at,
                ));
            }
        }
    }
    out
}

fn detect_dwell_anomalies(
    samples: &[DurationSample],
    baseline: &BaselineStats,
    config: StatAnomalyConfig,
    next_id: &mut u64,
) -> Vec<AnomalyEvent> {
    let (Some(mu), Some(sd)) = (baseline.dwell_mean, baseline.dwell_std) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for sample in samples {
        let value = sample.duration_ns as f32;
        let Some(z) = z_score(value, mu, sd) else {
            continue;
        };
        if z < config.z_threshold {
            continue;
        }
        out.push(make_event(
            next_id,
            z,
            AnomalyReason::UnusualDwell,
            sample.subject_id,
            sample.event_id,
            sample.at_ns,
        ));
    }
    out
}

fn detect_frequency_anomalies(
    events: &[TimedSubjectEvent],
    baseline: &BaselineStats,
    config: StatAnomalyConfig,
    next_id: &mut u64,
) -> Vec<AnomalyEvent> {
    let (Some(mu), Some(sd)) = (baseline.gap_mean, baseline.gap_std) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (subject, times) in group_subject_times(events) {
        for window in times.windows(2) {
            let gap = (window[1] - window[0]) as f32;
            let Some(z) = z_score(gap, mu, sd) else {
                continue;
            };
            if z < config.z_threshold {
                continue;
            }
            out.push(make_event(
                next_id,
                z,
                AnomalyReason::UnusualFrequency,
                subject,
                None,
                window[1],
            ));
        }
    }
    out
}

fn detect_time_anomalies(
    events: &[TimedSubjectEvent],
    baseline: &BaselineStats,
    config: StatAnomalyConfig,
    next_id: &mut u64,
) -> Vec<AnomalyEvent> {
    let (Some(mu), Some(sd)) = (baseline.hour_mean, baseline.hour_std) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for event in events {
        let hour = f32::from(hour_of_day_ns(event.at_ns));
        // Circular hour distance approx via min wrap.
        let delta = hour_delta(hour, mu);
        let Some(z) = z_score(mu + delta, mu, sd.max(0.5)) else {
            continue;
        };
        if z < config.z_threshold {
            continue;
        }
        out.push(make_event(
            next_id,
            z,
            AnomalyReason::UnusualAppearanceTime,
            event.subject_id,
            event.event_id,
            event.at_ns,
        ));
    }
    out
}

fn detect_missing_expected(
    events: &[TimedSubjectEvent],
    baseline: &BaselineStats,
    config: StatAnomalyConfig,
    next_id: &mut u64,
) -> Vec<AnomalyEvent> {
    let (Some(mu), Some(sd)) = (baseline.gap_mean, baseline.gap_std) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (subject, times) in group_subject_times(events) {
        if times.len() < 2 {
            continue;
        }
        let last = *times.last().unwrap();
        let prev = times[times.len() - 2];
        let gap = (last - prev) as f32;
        // Missing expected: gap much larger than baseline mean.
        let Some(z) = z_score(gap, mu, sd) else {
            continue;
        };
        if gap > mu && z >= config.z_threshold {
            out.push(make_event(
                next_id,
                z,
                AnomalyReason::MissingExpectedAppearance,
                subject,
                None,
                last,
            ));
        }
    }
    out
}

fn detect_sudden_change(
    samples: &[DurationSample],
    baseline: &BaselineStats,
    config: StatAnomalyConfig,
    next_id: &mut u64,
) -> Vec<AnomalyEvent> {
    let (Some(mu), Some(sd)) = (baseline.dwell_mean, baseline.dwell_std) else {
        return Vec::new();
    };
    if samples.len() < config.min_samples {
        return Vec::new();
    }
    // Compare recent window mean vs baseline mean.
    let recent_n = config.min_samples.min(samples.len());
    let recent: Vec<f32> = samples[samples.len() - recent_n..]
        .iter()
        .map(|s| s.duration_ns as f32)
        .collect();
    let Some(recent_mu) = mean(&recent) else {
        return Vec::new();
    };
    let Some(z) = z_score(recent_mu, mu, sd) else {
        return Vec::new();
    };
    if z < config.z_threshold {
        return Vec::new();
    }
    let last = samples.last();
    let at_ns = last.map_or(0, |s| s.at_ns);
    let subject = last.and_then(|s| s.subject_id);
    let event_id = last.and_then(|s| s.event_id);
    vec![make_event(
        next_id,
        z,
        AnomalyReason::SuddenBehaviourChange,
        subject,
        event_id,
        at_ns,
    )]
}

fn make_event(
    next_id: &mut u64,
    z: f32,
    reason: AnomalyReason,
    subject_id: Option<SubjectId>,
    event_id: Option<EventId>,
    at_ns: i64,
) -> AnomalyEvent {
    let id = AnomalyId(*next_id);
    *next_id = next_id.saturating_add(1);
    let mut evidence = Vec::new();
    if let Some(event_id) = event_id {
        evidence.push(event_id);
    }
    AnomalyEvent {
        anomaly_id: id,
        score: if z.is_finite() { z } else { 100.0 },
        severity: severity_from_z(z),
        reasons: alloc::vec![reason],
        evidence,
        subject_id,
        source_id: None,
        at: MediaTime::new(at_ns, 1_000_000_000).unwrap_or_default(),
    }
}

fn severity_from_z(z: f32) -> Severity {
    if !z.is_finite() {
        return Severity::Critical;
    }
    if z >= 6.0 {
        Severity::Critical
    } else if z >= 4.5 {
        Severity::High
    } else if z >= 3.5 {
        Severity::Medium
    } else {
        Severity::Low
    }
}

fn hour_delta(a: f32, b: f32) -> f32 {
    let d = (a - b).abs();
    d.min(24.0 - d)
}

fn group_subject_times(events: &[TimedSubjectEvent]) -> Vec<(Option<SubjectId>, Vec<i64>)> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::AnalysisSeries;

    const DAY_NS: i64 = 86_400_i64 * 1_000_000_000;

    fn timed(day: i64, subject: u64) -> TimedSubjectEvent {
        TimedSubjectEvent {
            subject_id: Some(SubjectId(subject)),
            source_id: None,
            at_ns: day * DAY_NS + 10 * 3_600_000_000_000, // 10:00
            event_id: Some(EventId(day as u64 + 1)),
            kind_tag: 0,
        }
    }

    #[test]
    fn baseline_day_fraction_concentrates_on_seen_weekdays() {
        let mut series = AnalysisSeries::default();
        // All events on epoch weekday 1 (days 1, 8, 15, …).
        for k in 0..12 {
            series.timed.push(timed(1 + k * 7, 1));
        }
        let cfg = StatAnomalyConfig {
            min_samples: 5,
            use_day_of_week: true,
            ..StatAnomalyConfig::default()
        };
        let baseline = build_baseline(&series, cfg);
        assert_eq!(baseline.day_n, 12);
        assert!((baseline.day_fraction[1] - 1.0).abs() < 1e-5);
        for d in [0, 2, 3, 4, 5, 6] {
            assert!(baseline.day_fraction[d].abs() < 1e-6);
        }
    }

    #[test]
    fn rare_weekday_flags_unusual_appearance_time() {
        let mut history = AnalysisSeries::default();
        for k in 0..14 {
            history.timed.push(timed(1 + k * 7, 1)); // weekday 1 only
        }
        let cfg = StatAnomalyConfig {
            min_samples: 5,
            use_day_of_week: true,
            rare_day_fraction: 0.08,
            use_robust: false,
            z_threshold: 2.5,
            change_point_threshold: 100.0,
        };
        let baseline = build_baseline(&history, cfg);

        let mut live = AnalysisSeries::default();
        // Weekday 3 never seen in baseline → rare.
        live.timed.push(timed(3, 1));

        let mut next_id = 1;
        let anomalies = detect_statistical(&live, &baseline, cfg, &mut next_id);
        assert!(
            anomalies
                .iter()
                .any(|a| a.reasons.contains(&AnomalyReason::UnusualAppearanceTime)),
            "expected day-of-week anomaly, got {anomalies:?}"
        );
    }

    #[test]
    fn day_of_week_can_be_disabled() {
        let mut history = AnalysisSeries::default();
        for k in 0..14 {
            history.timed.push(timed(1 + k * 7, 1));
        }
        let cfg = StatAnomalyConfig {
            use_day_of_week: false,
            use_robust: false,
            ..StatAnomalyConfig::default()
        };
        let baseline = build_baseline(&history, cfg);
        let mut live = AnalysisSeries::default();
        live.timed.push(timed(3, 1));
        let mut next_id = 1;
        let with = detect_statistical(
            &live,
            &baseline,
            StatAnomalyConfig {
                use_day_of_week: true,
                use_robust: false,
                ..StatAnomalyConfig::default()
            },
            &mut next_id,
        );
        let mut next_id = 1;
        let without = detect_statistical(&live, &baseline, cfg, &mut next_id);
        assert!(
            with.iter()
                .any(|a| a.reasons.contains(&AnomalyReason::UnusualAppearanceTime))
        );
        assert!(
            without
                .iter()
                .all(|a| !a.reasons.contains(&AnomalyReason::UnusualAppearanceTime)),
            "use_day_of_week=false must not emit weekday seasonality hits"
        );
    }
}
