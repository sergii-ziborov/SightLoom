//! Deterministic host benchmarks for the portable geometry and NMS core.

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use sightloom_core::{
    ClassId, Detection, NmsConfig, NmsMode, OverlapMetric, Rect, iou, nms_in_place,
};

const SIZES: [usize; 3] = [16, 64, 256];
const CONFIG: NmsConfig = NmsConfig {
    threshold: 0.5,
    mode: NmsMode::ClassAgnostic,
    metric: OverlapMetric::IoU,
};

struct NmsCase {
    detections: Vec<Detection>,
    order: Vec<usize>,
    suppressed: Vec<bool>,
}

fn deterministic_detections(count: usize) -> Vec<Detection> {
    let count = u16::try_from(count).expect("benchmark count fits in u16");
    (0..count)
        .map(|index| {
            let column = f32::from(index % 16);
            let row = f32::from(index / 16);
            let left = column * 5.0;
            let top = row * 5.0;
            let bbox = Rect::new(left, top, left + 8.0, top + 8.0)
                .expect("deterministic benchmark bounds are valid");
            Detection::new(
                bbox,
                f32::from(count - index) / f32::from(count),
                Some(ClassId(index % 4)),
                None,
            )
            .expect("deterministic benchmark score is finite")
        })
        .collect()
}

fn pairwise_iou(detections: &[Detection]) -> f32 {
    let mut total = 0.0;
    for (index, detection) in detections.iter().enumerate() {
        for other in detections.iter().skip(index + 1) {
            total += iou(detection.bbox(), other.bbox());
        }
    }
    total
}

fn benchmark_core(c: &mut Criterion) {
    let mut group = c.benchmark_group("core");
    for count in SIZES {
        let input = deterministic_detections(count);
        group.bench_with_input(
            BenchmarkId::new("pairwise_iou", count),
            &input,
            |b, input| {
                b.iter(|| black_box(pairwise_iou(black_box(input))));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("nms_in_place", count),
            &input,
            |b, input| {
                b.iter_batched_ref(
                    || NmsCase {
                        detections: input.clone(),
                        order: vec![0; input.len()],
                        suppressed: vec![false; input.len()],
                    },
                    |case| {
                        let kept = nms_in_place(
                            &mut case.detections,
                            &mut case.order,
                            &mut case.suppressed,
                            CONFIG,
                        )
                        .expect("fixed benchmark configuration and scratch must succeed");
                        black_box(kept);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, benchmark_core);
criterion_main!(benches);
