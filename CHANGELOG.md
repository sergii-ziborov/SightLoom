# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
within the `0.1.x` alpha line (API may still evolve).

## [Unreleased]

### Added
- **Evidence reels** stored on `VisionIndex` and persisted in package `entities.json` (`store_subject_reel`, `evidence_reels`).
- Track sample **host revisions**: `IndexSession::revise_latest_track_sample` (supersedes + revision).
- Synthetic MOT regression scenarios: `run_synthetic_parallel_walk`, `run_synthetic_crossing` (baseline smoke, not MOT17 scores).
- README refreshed for memory, redaction, reels, queue, and MOT helpers.
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
