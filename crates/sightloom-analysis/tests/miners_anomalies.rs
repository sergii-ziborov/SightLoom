//! Pattern miners and statistical anomaly backend tests.

use sightloom_analysis::{
    AnalysisSeries, AnomalyReason, DurationSample, PairSample, PatternKind, RouteSample,
    StatAnomalyConfig, TimedSubjectEvent, build_baseline, detect_statistical, mine_patterns,
};
use sightloom_core::{EventId, SubjectId, ZoneId};

fn hour_ns(hour: i64) -> i64 {
    hour * 3_600_000_000_000
}

#[test]
fn mines_time_of_day_and_co_occurrence_patterns() {
    let mut series = AnalysisSeries::default();
    let subject = SubjectId(1);
    // Many events around hour 9
    for i in 0..6 {
        series.timed.push(TimedSubjectEvent {
            subject_id: Some(subject),
            source_id: None,
            at_ns: hour_ns(9) + i * 60_000_000_000,
            event_id: Some(EventId(u64::try_from(i).unwrap() + 1)),
        });
    }
    // One outlier hour
    series.timed.push(TimedSubjectEvent {
        subject_id: Some(subject),
        source_id: None,
        at_ns: hour_ns(21),
        event_id: Some(EventId(99)),
    });

    for _ in 0..3 {
        series.pairs.push(PairSample {
            subject_a: SubjectId(1),
            subject_b: SubjectId(2),
            source_id: None,
            at_ns: hour_ns(9),
            event_id: Some(EventId(50)),
        });
    }

    series.routes.push(RouteSample {
        subject_id: subject,
        zones: vec![ZoneId(1), ZoneId(2), ZoneId(3)],
        at_ns: hour_ns(9),
        event_id: None,
    });
    series.routes.push(RouteSample {
        subject_id: subject,
        zones: vec![ZoneId(1), ZoneId(2), ZoneId(3)],
        at_ns: hour_ns(10),
        event_id: None,
    });

    for d in [1_000_000_000_i64, 1_100_000_000, 900_000_000, 1_050_000_000] {
        series.durations.push(DurationSample {
            subject_id: Some(subject),
            zone_id: Some(ZoneId(1)),
            duration_ns: d,
            at_ns: hour_ns(9),
            event_id: None,
        });
    }

    let mut next_id = 1_u64;
    let patterns = mine_patterns(&series, &mut next_id);
    assert!(
        patterns
            .iter()
            .any(|p| p.kind == PatternKind::TimeOfDay && p.subject_id == Some(subject))
    );
    assert!(patterns.iter().any(|p| p.kind == PatternKind::CoOccurrence));
    assert!(
        patterns
            .iter()
            .any(|p| p.kind == PatternKind::RouteSequence)
    );
    assert!(
        patterns
            .iter()
            .any(|p| p.kind == PatternKind::DwellDistribution)
    );
}

#[test]
fn statistical_detector_flags_unusual_dwell_and_time() {
    let mut history = AnalysisSeries::default();
    let subject = SubjectId(7);
    // Baseline: 1s dwells, morning hours, regular gaps of 1 day
    for day in 0..10 {
        let at = day * 86_400_000_000_000 + hour_ns(10);
        history.timed.push(TimedSubjectEvent {
            subject_id: Some(subject),
            source_id: None,
            at_ns: at,
            event_id: Some(EventId(u64::try_from(day).unwrap() + 1)),
        });
        history.durations.push(DurationSample {
            subject_id: Some(subject),
            zone_id: None,
            duration_ns: 1_000_000_000,
            at_ns: at,
            event_id: Some(EventId(u64::try_from(day).unwrap() + 100)),
        });
    }

    let config = StatAnomalyConfig {
        z_threshold: 2.5,
        min_samples: 5,
    };
    let baseline = build_baseline(&history, config);
    assert!(baseline.dwell_mean.is_some());
    assert!(baseline.gap_mean.is_some());
    assert!(baseline.hour_mean.is_some());

    let mut live = AnalysisSeries::default();
    // Huge dwell
    live.durations.push(DurationSample {
        subject_id: Some(subject),
        zone_id: None,
        duration_ns: 50_000_000_000,
        at_ns: 10 * 86_400_000_000_000 + hour_ns(10),
        event_id: Some(EventId(999)),
    });
    // Odd hour
    live.timed.push(TimedSubjectEvent {
        subject_id: Some(subject),
        source_id: None,
        at_ns: 10 * 86_400_000_000_000 + hour_ns(3),
        event_id: Some(EventId(1000)),
    });
    // Also keep a normal timed event to form a gap series with history not needed

    let mut next_id = 1_u64;
    let anomalies = detect_statistical(&live, &baseline, config, &mut next_id);
    assert!(
        anomalies
            .iter()
            .any(|a| a.reasons.contains(&AnomalyReason::UnusualDwell))
    );
    assert!(
        anomalies
            .iter()
            .any(|a| a.reasons.contains(&AnomalyReason::UnusualAppearanceTime))
    );
    assert!(anomalies.iter().all(|a| a.score >= config.z_threshold));
}
