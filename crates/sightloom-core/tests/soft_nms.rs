//! Soft-NMS and merge-NMS smoke tests.

use sightloom_core::{
    Detection, NmsConfig, NmsMode, OverlapMetric, Rect, SoftNmsConfig, SoftNmsMethod,
    merge_nms_in_place, soft_nms_in_place,
};

fn det(l: f32, score: f32) -> Detection {
    Detection::new(
        Rect::new(l, 0.0, l + 10.0, 10.0).unwrap(),
        score,
        None,
        None,
    )
    .unwrap()
}

#[test]
fn soft_nms_decays_overlap_keeps_both() {
    let mut dets = [det(0.0, 0.9), det(2.0, 0.85)];
    let mut order = [0_usize; 2];
    let mut scores = [0.0_f32; 2];
    let n = soft_nms_in_place(
        &mut dets,
        &mut order,
        &mut scores,
        SoftNmsConfig {
            method: SoftNmsMethod::Gaussian,
            sigma: 0.5,
            score_threshold: 0.01,
            ..SoftNmsConfig::default()
        },
    )
    .unwrap();
    assert_eq!(n, 2);
    assert!(dets[0].score() >= dets[1].score());
}

#[test]
fn merge_nms_merges_close_boxes() {
    let mut dets = [det(0.0, 0.9), det(1.0, 0.8)];
    let mut order = [0_usize; 2];
    let mut parent = [0_usize; 2];
    let n = merge_nms_in_place(
        &mut dets,
        &mut order,
        &mut parent,
        NmsConfig {
            threshold: 0.1,
            mode: NmsMode::ClassAgnostic,
            metric: OverlapMetric::IoU,
        },
    )
    .unwrap();
    assert_eq!(n, 1);
    assert!(dets[0].score() >= 0.9);
}
