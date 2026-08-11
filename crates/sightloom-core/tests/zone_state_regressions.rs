//! Public state-machine regression tests for zone monitors.

use sightloom_core::{
    CoreError, Direction, LineSegment, LineZoneMonitor, Point, Polygon, PolygonZoneMonitor,
    TrackId, VisionEvent, ZoneId,
};

fn point(x: f32, y: f32) -> Point {
    Point::new(x, y).unwrap()
}

fn square_points() -> [Point; 4] {
    [
        point(0.0, 0.0),
        point(4.0, 0.0),
        point(4.0, 4.0),
        point(0.0, 4.0),
    ]
}

fn horizontal_segment() -> LineSegment {
    LineSegment::new(point(0.0, 0.0), point(4.0, 0.0)).unwrap()
}

#[test]
fn existing_track_after_a_freed_slot_emits_its_exit() {
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
        monitor.update(TrackId(2), point(2.0, 2.0), &mut output),
        Ok(1)
    );
    assert!(monitor.forget_track(TrackId(1)));
    assert_eq!(
        monitor.update(TrackId(2), point(5.0, 2.0), &mut output),
        Ok(1)
    );
    assert_eq!(
        output[0],
        VisionEvent::Exited {
            track_id: TrackId(2),
            zone_id: ZoneId(7),
        }
    );
}

#[test]
fn line_retry_after_missing_output_preserves_the_crossing() {
    let mut monitor = LineZoneMonitor::<1>::new(ZoneId(3), horizontal_segment());
    let mut output = [VisionEvent::Entered {
        track_id: TrackId(99),
        zone_id: ZoneId(99),
    }];

    assert_eq!(
        monitor.update(TrackId(1), point(2.0, 1.0), &mut output),
        Ok(0)
    );
    assert_eq!(
        monitor.update(TrackId(1), point(2.0, -1.0), &mut []),
        Err(CoreError::InsufficientCapacity)
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
}
