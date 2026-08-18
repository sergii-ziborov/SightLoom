# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
within the `0.1.x` alpha line (API may still evolve).

## [Unreleased]

### Added
- **YOLO-style ONNX detection**: `PreprocessConfig::yolo_detect` + letterbox,
  `decode_detector_output` for YOLOv8/v11 `[1,4+C,N]`, YOLOv5 `[1,N,5+C]`,
  and flat `N×6`, plus class-aware NMS (`OnnxDetector::nms_thresh`).
  Example: `onnx_detect`.
- **TrackEval host bridge**: parse/write `MOTChallenge` text
  (`parse_mot_challenge_text`, `evaluate_mot_challenge_pair`), import host
  HOTA/MOTA/IDF1 (`parse_track_eval_summary` / `TrackEvalSummary`), session
  `export_mot_challenge`, evidence `MotEvidence::attach_track_eval_summary`
  (+ `host_track_eval.md` / in-tree CLEAR rescore). Still **not** published
  MOT17 leaderboard numbers.

## [0.1.6] - 2026-08-15

### Added
- **Quantum anomaly stub**: `QuantumStubDetector` / `QuantumStubConfig` behind
  `AnomalyDetector` (classical placeholder; real quantum solvers stay host-side).
- **Host model download** (feature `download`): `HttpModelFetcher` GETs
  `ModelSpec.uri` (http/https) into the model cache (atomic `.part` rename).
- **Encoded photo decode** (feature `image-decode`): JPEG/PNG → RGB for
  `PhotoView::encoded`; wired into `ReferenceEmbedder` / `OnnxEmbedder`.
  Convenience feature `full` = `onnx` + `download` + `image-decode`.
- **Day-of-week seasonality anomalies**: `StatAnomalyConfig::use_day_of_week` /
  `rare_day_fraction`; `BaselineStats::day_fraction` / `day_n`.
- **Graph / multi-camera relational anomalies**: `CameraGraph`,
  `GraphRelationalDetector`, `build_graph_baseline` / `detect_graph_anomalies`
  (impossible hops, rare camera transitions, rare co-occurrence pairs, rare
  zone routes). New reasons: `ImpossibleCrossCameraHop`,
  `UnusualCameraTransition`. Session:
  `detect_and_store_graph_anomalies` + `camera_graph_from_topology`.
- **Host weights cookbook**: `ModelManifest` JSON inventory, optional
  `ModelSpec.sha256` verification (`file_sha256_hex` / `HostError::Integrity`),
  `resolve_manifest`, example `weights_cookbook`, `COOKBOOK.md`.

## [0.1.5] - 2026-08-15

### Added
- **`sightloom-host` crate** (host model package, steps 1–4):
  - config / preprocess / reference detectors+embedders / `HostPipeline`
  - feature `onnx`: pure-Rust **tract** `OnnxEmbedder` / `OnnxDetector` (weights on disk only)
  - evidence packs: MOT + MOTChallenge, re-id ROC/EER, redaction pixels, anomaly FAR
  - examples: `photo_to_subject`, `onnx_photo_search`, `write_evidence_pack`
- **Anomaly FAR + scoped baselines**: `calibrate_far_threshold` / `calibrate_far_from_series`,
  `ScopedBaselineStore` (subject/camera), session `calibrate_anomaly_far` /
  `apply_anomaly_far` / `detect_and_store_anomalies_scoped`
- **Identity hypothesis lifecycle**: `open_identity_cases`, `accept_identity_hypothesis`,
  `dismiss_identity_case`, `assigned_identity_view` / `identity_audit_view`
- **Negative evidence policy**: `NegativeEvidencePolicy` on `ResolveConfig`
- **Cross-camera travel window**: `set_edge_window` / `allows_hop`
- **Continuous track embeddings**: `TrackEmbeddingAdapter`, `detect_ingest_and_embed_tracks`
- **Per-camera thresholds + topology session API**
- **Gapped uncertainty intervals**: `uncertain_intervals_gapped`
- **Polygon OBB IoU**: `oriented_iou`
- **Audit vs effective track views** on session

## [0.1.4] - 2026-08-14

### Added
- **Evidence reels** stored on `VisionIndex` and persisted in package `entities.json` (`store_subject_reel`, `evidence_reels`).
- Track sample **host revisions**: `IndexSession::revise_latest_track_sample` (supersedes + revision).
- Synthetic MOT regression scenarios: `run_synthetic_parallel_walk`, `run_synthetic_crossing` (baseline smoke, not MOT17 scores).
- **Observation** supersedes/revision/idempotency + `VisionIndex` observation table (package-backed).
- Host **idempotency keys**: `ingest_detections_keyed` / `SessionError::DuplicateIdempotencyKey`.
- **`DetectorAdapter`** + `FrameView` / `detect_and_ingest` (host-owned models).
- **Spatial query**: `SpatialQuery` / `execute_spatial_query` / `query_spatial`.
- **EventBeforeEvent** pattern miner (`kind_tag` on timed events).
- Identity hypotheses helpers: `latest_identity_audit`, `identity_hypotheses`.
- **ANN foundation**: `AnnIndex` / `BruteForceAnn` / `LshAnn` / `AnnBackend`; `EmbeddingStore::search_top_k` / `build_ann`; session `set_track_ann_kind` for track embedding search.
- **Query AST**: `QueryNode` / `SubjectPredicate` / `execute_query_ast` / `query_ast` (AND/OR/NOT).
- **Retention policy**: `RetentionPolicy` / `apply_retention` (track samples, observations, audit).
- **Prometheus text metrics**: `prometheus_text` / `IndexSession::prometheus_metrics` (no network I/O).
- **HNSW ANN** (pure Rust): `HnswAnn` / `AnnKind::Hnsw` / `AnnKind::hnsw_default`; host FAISS hook via `HostAnnAdapter` (no FAISS link).
- **ROC/EER calibration**: `compute_roc`, `CalibrationReport`, `resolve_config_from_calibration`, gallery/session apply helpers.
- **Streaming subject query**: `StreamingSubjectQuery` / `stream_subjects` / `stream_next_page` / `stream_poll_new`.
- **Deterministic NL query bridge** (no LLM): `parse_nl_query` / `query_nl` → `QueryNode`.
- **Privacy/retention product**: legal holds (subject/source), per-source TTL, `forget_subject`, `RetentionReport`.
- **Photo embedding adapter**: `PhotoEmbeddingAdapter` / `search_photo_with_adapter` (photo→vector stays host-side).
- **Quality reports**: tracking/re-id/redaction pixel metrics helpers (host-filled evidence).
- **Anomaly backends**: `AnomalyDetector` trait + robust MAD / CUSUM change-point / subject-specific gaps.
- **Soft-NMS / merge-NMS** (`soft_nms_in_place`, `merge_nms_in_place`).
- **Oriented boxes**: corners, AABB, approximate oriented NMS; **keypoints** store.
- **Inference tiling**: `generate_tiles` / `tile_to_global` for 4K/8K host detectors.
- Example **`host_model_stub`**: fake detector + photo embedder → enroll/search/memory.
- **Telemetry adapters**: `MetricsExporter` / `SpanExporter`, OTLP-shaped JSON metrics, span helpers (no OTel SDK dep).
- **Isolation Forest** anomaly backend (`IsolationForestDetector`) behind `AnomalyDetector`.
- **Moore contour tracing**: `dense_to_contour` / `dense_to_contours` (outer boundary from dense masks).
- **Synthetic MOT suite report**: `run_mot_smoke_suite` / `MotSuiteReport` + `MOTChallenge` text export for host-side TrackEval (not published MOT17 scores).
- **One-Class SVM** (RBF) anomaly backend: `OcSvmDetector` / `OcSvmConfig` behind `AnomalyDetector` (pure Rust baseline, not libsvm SMO parity).
- **Arrow-shaped track stream**: `TrackArrowBatch`, `encode_track_arrow` / `decode_track_arrow` (`SLARROW1` columnar codec; package default remains CBOR).
- **Source lifecycle hardening**: `MultiSourceTracker::reset_source` / `remove_source`; session `apply_source_lifecycle` clears motion + watermarks on Reset/Removed.
- Topology helpers: `CameraTopology::set_bidirectional` / `remove_edge` / `edge_count`.

### Improved
- `Display` + `std::error::Error` on core/session error types for host `?` ergonomics.
- Expanded crate READMEs for docs.rs; facade quick-start example; fixed broken intra-doc links.

## [0.1.3] - 2026-08-13

### Added
- Auto **`SubjectProfile`** fill from appearances (or track samples): `rebuild_subject_profiles`, `rebuild_memory_from_tracks`, `set_subject_label` (preserves labels/embeddings; optional gallery embedding enrich).
- First-class **redaction provenance intervals** (`RedactionInterval` / `RedactionIntent`) in `VisionIndex` + package `entities.json` (backward-compatible default empty).
- Session planners: `plan_redaction_subject`, `plan_redaction_blur_others`, `plan_redaction_uncertain`, `export_redaction_intervals_json`, `clear_redaction_intervals`.
- Opt-in **auto memory rebuild** during ingest: `MemoryAutoRebuild` / `set_memory_auto_rebuild` (every N accepted frames).
- **`ingest_detection_batch`** / **`ingest_detection_batch_soft`** for multi-frame host ingest loops.
- Bounded host **`FrameQueue`** (`DropOldest` / `DropNewest` / `RejectNew`) + `drain_frame_queue`.

## [0.1.2] - 2026-08-13

### Added
- Package **`gallery.json`** sidecar (subjects, embeddings, track→subject map, track embedding index) written by `IndexSession::save_package` and restored by `load_package`.
- **Track embedding search index**: `note_track_embedding` records the latest handle per `TrackKey`; `search_tracks_by_embedding` ranks unlabeled tracks by cosine similarity.
- Auto **appearances / visits** materialization from effective track samples (`MemoryBuildConfig`, `rebuild_appearances_and_visits`).

### Fixed
- CI-facing issues for package/sqlite paths and pedantic clippy in tests (follow-up commits after tag).

## [0.1.1] - 2026-08-13

### Added
- Reference-photo enrollment and multi-factor gallery search: `enroll_subject_photos`, `add_subject_photo`, `search_by_photo`, `search_photo_with_reels`.
- Evidence reel builders and subject ranking (`rank_subjects`, `most_frequent_subject_reel`).
- Demo export helpers (spans / uncertain intervals JSON), seed click APIs.

### Fixed
- `no_std` package verification (`vec!` / `Vec` imports in tracking and re-id).

## [0.1.0] - 2026-08-13

### Added
- First crates.io release of the workspace:
  - `sightloom-core`
  - `sightloom-tracking`
  - `sightloom-analysis`
  - `sightloom-reid`
  - `sightloom-index`
  - `sightloom` (facade)
- Multi-source tracking (`TrackKey`, `TrackUid`, per-source tracker pool).
- Session checkpoint + transactional package generations with checksums.
- Multi-factor re-id foundation (similarity × quality × temporal × topology × class × prior).
- Subject query foundation, validation, baseline MOT helpers, host sketch example.

### Notes
- Tracker is a **baseline** association implementation, not a published MOT benchmark winner.
- Publish uses GitHub Actions secret `CARGO_REGISTRY_TOKEN` (never committed).

[0.1.3]: https://github.com/sergii-ziborov/SightLoom/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/sergii-ziborov/SightLoom/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/sergii-ziborov/SightLoom/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/sergii-ziborov/SightLoom/releases/tag/v0.1.0
