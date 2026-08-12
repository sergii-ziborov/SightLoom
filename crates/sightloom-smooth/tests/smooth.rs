//! Smoothing tests.

use sightloom_core::{Detection, Rect, TrackId};
use sightloom_smooth::{
    DetectionSmoother, SmoothConfig, TrajectoryHistory, TrajectorySample, interpolate_bbox,
};

fn tracked(id: u32, left: f32, score: f32) -> Detection {
    Detection::new(
        Rect::new(left, 0.0, left + 10.0, 20.0).unwrap(),
        score,
        None,
        Some(TrackId(id)),
    )
    .unwrap()
}

#[test]
fn smoother_blends_bbox() {
    let mut smoother = DetectionSmoother::<8>::new(SmoothConfig {
        alpha: 0.5,
        max_missed: 2,
    })
    .unwrap();
    let mut out = [Detection::default(); 8];
    let n = smoother.update(&[tracked(1, 0.0, 0.9)], &mut out).unwrap();
    assert_eq!(n, 1);
    let n = smoother.update(&[tracked(1, 10.0, 0.9)], &mut out).unwrap();
    assert_eq!(n, 1);
    // 0.5 blend from 0 toward 10 => left at 5
    assert!((out[0].bbox().left() - 5.0).abs() < 1e-5);
}

#[test]
fn interpolate_midpoint() {
    let a = Rect::new(0.0, 0.0, 10.0, 10.0).unwrap();
    let b = Rect::new(10.0, 0.0, 20.0, 10.0).unwrap();
    let mid = interpolate_bbox(a, b, 0.5).unwrap();
    assert!((mid.left() - 5.0).abs() < 1e-5);
    assert!((mid.right() - 15.0).abs() < 1e-5);
}

#[test]
fn trajectory_velocity_and_jitter() {
    let mut hist = TrajectoryHistory::<8>::new(TrackId(7));
    hist.push(TrajectorySample {
        frame_index: 0,
        bbox: Rect::new(0.0, 0.0, 10.0, 10.0).unwrap(),
        confidence: 0.9,
    })
    .unwrap();
    hist.push(TrajectorySample {
        frame_index: 1,
        bbox: Rect::new(4.0, 0.0, 14.0, 10.0).unwrap(),
        confidence: 0.9,
    })
    .unwrap();
    let v = hist.velocity().expect("velocity");
    assert!((v.x() - 4.0).abs() < 1e-4);
    assert!(hist.jitter() > 0.0);
}
