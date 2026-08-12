//! Geometry-reference overlap fixture tests.

use serde::Deserialize;
use sightloom_core::{Rect, ios, iou};

const MANIFEST: &str = include_str!("../../../fixtures/geometry-reference/manifest.json");
const OVERLAP: &str = include_str!("../../../fixtures/geometry-reference/overlap.json");

#[derive(Debug, Deserialize)]
struct Manifest {
    numeric_contract: NumericContract,
}

#[derive(Debug, Deserialize)]
struct NumericContract {
    absolute_tolerance: f64,
    relative_tolerance: f64,
}

#[derive(Debug, Deserialize)]
struct OverlapFixture {
    cases: Vec<OverlapCase>,
}

#[derive(Debug, Deserialize)]
struct OverlapCase {
    expected: f64,
    first: Vec<f32>,
    metric: String,
    name: String,
    second: Vec<f32>,
}

fn rect(bounds: Vec<f32>) -> Rect {
    let bounds: [f32; 4] = bounds
        .try_into()
        .expect("reference fixture rectangle must contain four coordinates");
    Rect::new(bounds[0], bounds[1], bounds[2], bounds[3])
        .expect("reference fixture rectangle must be valid")
}

#[test]
fn overlap_matches_geometry_reference() {
    let manifest: Manifest =
        blazingly_json::from_str(MANIFEST).expect("manifest fixture must be valid JSON");
    let fixture: OverlapFixture =
        blazingly_json::from_str(OVERLAP).expect("overlap fixture must be valid JSON");

    for case in fixture.cases {
        let first = rect(case.first);
        let second = rect(case.second);
        let actual = match case.metric.as_str() {
            "IOU" => iou(first, second),
            "IOS" => ios(first, second),
            metric => panic!("unsupported fixture metric: {metric}"),
        };
        let actual = f64::from(actual);
        let difference = (actual - case.expected).abs();
        let scale = actual.abs().max(case.expected.abs());
        let tolerance = manifest.numeric_contract.absolute_tolerance
            + manifest.numeric_contract.relative_tolerance * scale;

        assert!(
            difference <= tolerance,
            "case {}: expected {}, got {actual}, tolerance {tolerance}",
            case.name,
            case.expected
        );
    }
}
