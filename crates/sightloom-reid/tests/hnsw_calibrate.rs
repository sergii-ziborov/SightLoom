//! HNSW ANN + ROC/EER calibration integration.

#![allow(clippy::cast_precision_loss)]

use sightloom_reid::{
    AnnBackend, AnnIndex, AnnKind, EmbeddingStore, LabeledScore, ResolveConfig, SubjectGallery,
    SubjectModality, compute_roc, resolve_config_from_calibration,
};

#[test]
fn hnsw_backend_search_and_calibration_pipeline() {
    let mut backend = AnnBackend::new(AnnKind::hnsw_default());
    for i in 0..30_u64 {
        let t = i as f32 * 0.1;
        backend.upsert(i, &[t.cos(), t.sin(), 0.0]).unwrap();
    }
    let hits = backend.search(&[1.0, 0.0, 0.0], 3).unwrap();
    assert!(!hits.is_empty());
    assert!(hits[0].score > 0.85);

    // Calibration: well-separated genuine / impostor cosines.
    let mut scores = Vec::new();
    for i in 0..25 {
        scores.push(LabeledScore {
            score: 0.88 + (i as f32) * 0.002,
            genuine: true,
        });
        scores.push(LabeledScore {
            score: 0.15 + (i as f32) * 0.005,
            genuine: false,
        });
    }
    let report = compute_roc(&scores, 40).unwrap();
    assert!(report.eer < 0.1, "eer={}", report.eer);

    let mut gallery = SubjectGallery::new();
    gallery
        .apply_calibration(&report)
        .expect("config from calibration");
    let cfg = gallery.resolve_config();
    assert!(cfg.accept_threshold >= cfg.reject_threshold);

    // From pairs through store.
    let mut store = EmbeddingStore::new();
    let a = store.insert([1.0_f32, 0.0]).unwrap();
    let b = store.insert([0.99_f32, 0.01]).unwrap();
    let c = store.insert([0.0_f32, 1.0]).unwrap();
    let pairs = [(a, b, true), (a, c, false), (b, c, false)];
    // Need more pairs for both classes - expand
    let mut labeled = Vec::new();
    for _ in 0..10 {
        labeled.push((a, b, true));
        labeled.push((a, c, false));
    }
    let scores2 = sightloom_reid::labeled_scores_from_pairs(&store, &labeled).unwrap();
    let report2 = compute_roc(&scores2, 16).unwrap();
    let cfg2 = resolve_config_from_calibration(ResolveConfig::default(), &report2);
    assert!(cfg2.validate().is_ok());
    let _ = pairs;
    let _ = SubjectModality::PersonAppearance;
}
