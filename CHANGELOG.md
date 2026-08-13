# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
within the `0.1.x` alpha line (API may still evolve).

## [Unreleased]

### Added
- Auto **`SubjectProfile`** fill from appearances (or track samples): `rebuild_subject_profiles`, `rebuild_memory_from_tracks`, `set_subject_label` (preserves labels/embeddings; optional gallery embedding enrich).
- First-class **redaction provenance intervals** (`RedactionInterval` / `RedactionIntent`) in `VisionIndex` + package `entities.json` (backward-compatible default empty).
- Session planners: `plan_redaction_subject`, `plan_redaction_blur_others`, `plan_redaction_uncertain`, `export_redaction_intervals_json`, `clear_redaction_intervals`.
- Opt-in **auto memory rebuild** during ingest: `MemoryAutoRebuild` / `set_memory_auto_rebuild` (every N accepted frames).
- **`ingest_detection_batch`** for multi-frame host ingest loops.

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

[0.1.2]: https://github.com/sergii-ziborov/SightLoom/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/sergii-ziborov/SightLoom/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/sergii-ziborov/SightLoom/releases/tag/v0.1.0
