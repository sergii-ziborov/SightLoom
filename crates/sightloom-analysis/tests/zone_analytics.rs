//! Zone analytics tests.

use sightloom_analysis::{AnalyticsEvent, AnchorPolicy, ZoneAnalytics, ZoneAnalyticsConfig};
use sightloom_core::{MediaTime, Point, Polygon, Rect, TrackId, ZoneId};

fn with_zone<F: FnOnce(&mut ZoneAnalytics<'_, 8>)>(f: F) {
    let points = [
        Point::new(0.0, 0.0).unwrap(),
        Point::new(100.0, 0.0).unwrap(),
        Point::new(100.0, 100.0).unwrap(),
        Point::new(0.0, 100.0).unwrap(),
    ];
    let polygon = Polygon::new(&points).unwrap();
    let config = ZoneAnalyticsConfig {
        anchor: AnchorPolicy::Center,
        enter_hysteresis: 2,
        exit_hysteresis: 2,
        missed_frame_tolerance: 1,
        class_filter: None,
        dwell_start_debounce_ns: 0,
    };
    let mut zone = ZoneAnalytics::new(ZoneId(1), polygon, config).unwrap();
    f(&mut zone);
}

#[test]
fn hysteresis_delays_enter_and_exit() {
    with_zone(|zone| {
        let mut out = [AnalyticsEvent::OccupancyChanged {
            zone_id: ZoneId(0),
            occupancy: 0,
        }; 8];
        let now = MediaTime::new(0, 30).unwrap();
        let inside = Rect::new(40.0, 40.0, 60.0, 60.0).unwrap();
        let n = zone
            .update(TrackId(1), inside, None, None, 0, now, &mut out)
            .unwrap();
        assert_eq!(n, 0, "first inside sample should not confirm");

        let n = zone
            .update(
                TrackId(1),
                inside,
                None,
                None,
                1,
                MediaTime::new(1, 30).unwrap(),
                &mut out,
            )
            .unwrap();
        assert!(n >= 1);
        assert!(matches!(out[0], AnalyticsEvent::Entered { .. }));
        assert_eq!(zone.occupancy(), 1);

        let outside = Rect::new(200.0, 200.0, 220.0, 220.0).unwrap();
        let n = zone
            .update(
                TrackId(1),
                outside,
                None,
                None,
                2,
                MediaTime::new(2, 30).unwrap(),
                &mut out,
            )
            .unwrap();
        assert_eq!(n, 0, "first outside should not exit yet");

        let n = zone
            .update(
                TrackId(1),
                outside,
                None,
                None,
                3,
                MediaTime::new(3, 30).unwrap(),
                &mut out,
            )
            .unwrap();
        assert!(n >= 1);
        assert!(matches!(out[0], AnalyticsEvent::Exited { .. }));
        assert_eq!(zone.occupancy(), 0);
        assert_eq!(zone.visit_count(TrackId(1)), Some(1));
    });
}

#[test]
fn bottom_center_anchor_can_miss_when_only_top_overlaps() {
    let points = [
        Point::new(0.0, 50.0).unwrap(),
        Point::new(100.0, 50.0).unwrap(),
        Point::new(100.0, 100.0).unwrap(),
        Point::new(0.0, 100.0).unwrap(),
    ];
    let polygon = Polygon::new(&points).unwrap();
    let config = ZoneAnalyticsConfig {
        anchor: AnchorPolicy::BottomCenter,
        enter_hysteresis: 1,
        exit_hysteresis: 1,
        ..ZoneAnalyticsConfig::default()
    };
    let mut zone: ZoneAnalytics<'_, 8> = ZoneAnalytics::new(ZoneId(2), polygon, config).unwrap();
    let mut out = [AnalyticsEvent::OccupancyChanged {
        zone_id: ZoneId(0),
        occupancy: 0,
    }; 4];
    let bbox = Rect::new(40.0, 20.0, 60.0, 40.0).unwrap();
    let n = zone
        .update(
            TrackId(1),
            bbox,
            None,
            None,
            0,
            MediaTime::new(0, 1).unwrap(),
            &mut out,
        )
        .unwrap();
    assert_eq!(n, 0);
    let bbox = Rect::new(40.0, 40.0, 60.0, 80.0).unwrap();
    let n = zone
        .update(
            TrackId(1),
            bbox,
            None,
            None,
            1,
            MediaTime::new(1, 1).unwrap(),
            &mut out,
        )
        .unwrap();
    assert!(n >= 1);
}
