//! Public zone-monitor event contract tests.

use sightloom_core::{
    CoreError, Direction, LineSegment, LineZoneMonitor, Point, Polygon, PolygonZoneMonitor,
    TrackId, VisionEvent, ZoneId,
};

fn point(x: f32, y: f32) -> Point {
    Point::new(x, y).unwrap()
}

fn horizontal_segment() -> LineSegment {
    LineSegment::new(point(0.0, 0.0), point(4.0, 0.0)).unwrap()
}

fn square_points() -> [Point; 4] {
    [
        point(0.0, 0.0),
        point(4.0, 0.0),
        point(4.0, 4.0),
        point(0.0, 4.0),
    ]
}

#[test]
fn polygon_emits_membership_transitions_and_includes_boundary() {
    let points = square_points();
    let mut monitor = PolygonZoneMonitor::<2>::new(ZoneId(7), Polygon::new(&points).unwrap());
    let mut output = [VisionEvent::Entered {
        track_id: TrackId(99),
        zone_id: ZoneId(99),
    }];

    assert_eq!(
        monitor.update(TrackId(1), point(5.0, 2.0), &mut output),
        Ok(0)
    );
    assert_eq!(
        monitor.update(TrackId(1), point(2.0, 2.0), &mut output),
        Ok(1)
    );
    assert_eq!(
        output[0],
        VisionEvent::Entered {
            track_id: TrackId(1),
            zone_id: ZoneId(7),
        }
    );
    assert_eq!(
        monitor.update(TrackId(1), point(0.0, 2.0), &mut output),
        Ok(0)
    );
    assert_eq!(
        monitor.update(TrackId(1), point(5.0, 2.0), &mut output),
        Ok(1)
    );
    assert_eq!(
        output[0],
        VisionEvent::Exited {
            track_id: TrackId(1),
            zone_id: ZoneId(7),
        }
    );
    assert_eq!(
        monitor.update(TrackId(2), point(0.0, 0.0), &mut output),
        Ok(1)
    );
}

#[test]
fn line_emits_directions_for_crossings() {
    let mut monitor = LineZoneMonitor::<2>::new(ZoneId(3), horizontal_segment());
    let mut output = [VisionEvent::Entered {
        track_id: TrackId(99),
        zone_id: ZoneId(99),
    }];

    assert_eq!(
        monitor.update(TrackId(1), point(2.0, 1.0), &mut output),
        Ok(0)
    );
    assert_eq!(
        monitor.update(TrackId(1), point(2.0, -1.0), &mut output),
        Ok(1)
    );
    assert_eq!(
        output[0],
        VisionEvent::Crossed {
            track_id: TrackId(1),
            zone_id: ZoneId(3),
            direction: Direction::LeftToRight,
        }
    );
    assert_eq!(
        monitor.update(TrackId(2), point(2.0, -1.0), &mut output),
        Ok(0)
    );
    assert_eq!(
        monitor.update(TrackId(2), point(2.0, 1.0), &mut output),
        Ok(1)
    );
    assert_eq!(
        output[0],
        VisionEvent::Crossed {
            track_id: TrackId(2),
            zone_id: ZoneId(3),
            direction: Direction::RightToLeft,
        }
    );
}

#[test]
fn line_ignores_touches_and_extension_only_crossings() {
    let mut monitor = LineZoneMonitor::<2>::new(ZoneId(3), horizontal_segment());
    let mut output = [VisionEvent::Entered {
        track_id: TrackId(99),
        zone_id: ZoneId(99),
    }];

    assert_eq!(
        monitor.update(TrackId(1), point(2.0, 1.0), &mut output),
        Ok(0)
    );
    assert_eq!(
        monitor.update(TrackId(1), point(2.0, 0.0), &mut output),
        Ok(0)
    );
    assert_eq!(
        monitor.update(TrackId(1), point(2.0, 1.0), &mut output),
        Ok(0)
    );
    assert_eq!(
        monitor.update(TrackId(2), point(6.0, 1.0), &mut output),
        Ok(0)
    );
    assert_eq!(
        monitor.update(TrackId(2), point(6.0, -1.0), &mut output),
        Ok(0)
    );
}

#[test]
fn line_preserves_the_non_on_side_across_on_samples() {
    let mut monitor = LineZoneMonitor::<2>::new(ZoneId(3), horizontal_segment());
    let mut output = [VisionEvent::Entered {
        track_id: TrackId(99),
        zone_id: ZoneId(99),
    }];

    assert_eq!(
        monitor.update(TrackId(1), point(2.0, 1.0), &mut output),
        Ok(0)
    );
    assert_eq!(
        monitor.update(TrackId(1), point(2.0, 0.0), &mut output),
        Ok(0)
    );
    assert_eq!(
        monitor.update(TrackId(1), point(2.0, -1.0), &mut output),
        Ok(1)
    );
    assert_eq!(
        output[0],
        VisionEvent::Crossed {
            track_id: TrackId(1),
            zone_id: ZoneId(3),
            direction: Direction::LeftToRight,
        }
    );
    assert_eq!(
        monitor.update(TrackId(2), point(2.0, -1.0), &mut output),
        Ok(0)
    );
    assert_eq!(
        monitor.update(TrackId(2), point(2.0, 0.0), &mut output),
        Ok(0)
    );
    assert_eq!(
        monitor.update(TrackId(2), point(2.0, 1.0), &mut output),
        Ok(1)
    );
    assert_eq!(
        output[0],
        VisionEvent::Crossed {
            track_id: TrackId(2),
            zone_id: ZoneId(3),
            direction: Direction::RightToLeft,
        }
    );
}

#[test]
fn tracks_are_independent_and_forgetting_makes_an_observation_fresh() {
    let points = square_points();
    let mut monitor = PolygonZoneMonitor::<2>::new(ZoneId(7), Polygon::new(&points).unwrap());
    let mut output = [VisionEvent::Entered {
        track_id: TrackId(99),
        zone_id: ZoneId(99),
    }];

    assert_eq!(
        monitor.update(TrackId(1), point(2.0, 2.0), &mut output),
        Ok(1)
    );
    assert_eq!(
        output[0],
        VisionEvent::Entered {
            track_id: TrackId(1),
            zone_id: ZoneId(7)
        }
    );
    assert_eq!(
        monitor.update(TrackId(2), point(2.0, 2.0), &mut output),
        Ok(1)
    );
    assert_eq!(
        output[0],
        VisionEvent::Entered {
            track_id: TrackId(2),
            zone_id: ZoneId(7)
        }
    );
    assert!(monitor.forget_track(TrackId(1)));
    assert!(!monitor.forget_track(TrackId(1)));
    assert_eq!(
        monitor.update(TrackId(1), point(5.0, 2.0), &mut output),
        Ok(0)
    );
}

#[test]
fn capacity_is_reclaimed_after_forgetting_and_zero_capacity_rejects_observations() {
    let points = square_points();
    let mut monitor = PolygonZoneMonitor::<1>::new(ZoneId(7), Polygon::new(&points).unwrap());
    let mut output = [VisionEvent::Entered {
        track_id: TrackId(99),
        zone_id: ZoneId(99),
    }];

    assert_eq!(
        monitor.update(TrackId(1), point(5.0, 2.0), &mut output),
        Ok(0)
    );
    assert_eq!(
        monitor.update(TrackId(2), point(5.0, 2.0), &mut output),
        Err(CoreError::InsufficientCapacity)
    );
    assert!(monitor.forget_track(TrackId(1)));
    assert_eq!(
        monitor.update(TrackId(2), point(5.0, 2.0), &mut output),
        Ok(0)
    );

    let mut zero = PolygonZoneMonitor::<0>::new(ZoneId(7), Polygon::new(&points).unwrap());
    assert_eq!(
        zero.update(TrackId(1), point(5.0, 2.0), &mut output),
        Err(CoreError::InsufficientCapacity)
    );
}

#[test]
fn a_missing_output_slot_preserves_state_for_retry() {
    let points = square_points();
    let mut monitor = PolygonZoneMonitor::<1>::new(ZoneId(7), Polygon::new(&points).unwrap());
    let mut output = [VisionEvent::Entered {
        track_id: TrackId(99),
        zone_id: ZoneId(99),
    }];

    assert_eq!(
        monitor.update(TrackId(1), point(5.0, 2.0), &mut output),
        Ok(0)
    );
    assert_eq!(
        monitor.update(TrackId(1), point(2.0, 2.0), &mut []),
        Err(CoreError::InsufficientCapacity)
    );
    assert_eq!(
        monitor.update(TrackId(1), point(2.0, 2.0), &mut output),
        Ok(1)
    );
    assert_eq!(
        output[0],
        VisionEvent::Entered {
            track_id: TrackId(1),
            zone_id: ZoneId(7),
        }
    );
}
