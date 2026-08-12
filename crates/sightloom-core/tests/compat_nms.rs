//! Geometry-reference NMS fixture tests.

use serde::Deserialize;
use sightloom_core::{
    ClassId, Detection, NmsConfig, NmsMode, OverlapMetric, Rect, TrackId, nms_in_place,
};

const NMS: &str = include_str!("../../../fixtures/geometry-reference/nms.json");

#[derive(Debug, Deserialize)]
struct NmsFixture {
    cases: Vec<NmsCase>,
}

#[derive(Debug, Deserialize)]
struct NmsCase {
    expected_keep: Vec<bool>,
    metric: String,
    name: String,
    predictions: Vec<Vec<f32>>,
    threshold: f32,
}

fn detection(row: &[f32], original_index: usize) -> Detection {
    assert_eq!(
        row.len(),
        6,
        "reference prediction row must contain six values"
    );
    let bbox = Rect::new(row[0], row[1], row[2], row[3])
        .expect("reference prediction bounds must be valid");
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let class = ClassId(row[5] as u16);
    let track_id = TrackId(u32::try_from(original_index).expect("fixture index must fit in u32"));

    Detection::new(bbox, row[4], Some(class), Some(track_id))
        .expect("reference prediction score must be finite")
}

fn metric(name: &str) -> OverlapMetric {
    match name {
        "IOU" => OverlapMetric::IoU,
        "IOS" => OverlapMetric::IoS,
        value => panic!("unsupported fixture metric: {value}"),
    }
}

#[test]
fn class_aware_nms_matches_geometry_reference() {
    let fixture: NmsFixture = blazingly_json::from_str(NMS).expect("fixture must be valid JSON");

    for case in fixture.cases {
        let mut detections = case
            .predictions
            .into_iter()
            .enumerate()
            .map(|(index, row)| detection(&row, index))
            .collect::<Vec<_>>();
        let mut order = vec![0; detections.len()];
        let mut suppressed = vec![false; detections.len()];
        let kept = nms_in_place(
            &mut detections,
            &mut order,
            &mut suppressed,
            NmsConfig {
                threshold: case.threshold,
                mode: NmsMode::ClassAware,
                metric: metric(&case.metric),
            },
        )
        .expect("fixture NMS must succeed");
        let mut actual = vec![false; suppressed.len()];
        for detection in &detections[..kept] {
            let track_id = detection
                .track_id()
                .expect("fixture detection must have a track ID");
            let index = usize::try_from(track_id.0).expect("track ID must fit in usize");
            actual[index] = true;
        }

        // SightLoom equal-score tie-break keeps the lower original index.
        if case.name == "equal_scores" {
            assert_eq!(case.expected_keep, [false, true]);
            assert_eq!(actual, [true, false]);
        } else {
            assert_eq!(actual, case.expected_keep, "case {}", case.name);
        }
    }
}
