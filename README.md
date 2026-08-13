<div align="center">

# SightLoom

### Model-neutral video understanding and memory library

[![Status](https://img.shields.io/badge/status-VisionIndex%20alpha-2563eb)](https://github.com/sergii-ziborov/SightLoom)
[![CI](https://github.com/sergii-ziborov/SightLoom/actions/workflows/ci.yml/badge.svg)](https://github.com/sergii-ziborov/SightLoom/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/sightloom.svg)](https://crates.io/crates/sightloom)
[![docs.rs](https://docs.rs/sightloom/badge.svg)](https://docs.rs/sightloom)
[![Rust](https://img.shields.io/badge/Rust-1.97%2B-000000?logo=rust)](https://www.rust-lang.org/)
[![Target](https://img.shields.io/badge/target-no__std%20core-7c3aed)](https://docs.rust-embedded.org/book/intro/no-std.html)
[![License](https://img.shields.io/badge/license-MIT-2563eb)](LICENSE)

</div>

SightLoom turns **model-neutral detections** (or frames through an optional
host-side detector adapter) into **tracks, identities, events, and queryable
video memory**. It returns structured data — not drawn pixels, not decoded
video, and not a capture or render product.

Typical consumers are media and intelligence hosts that own capture, edit
plans, and rendering. SightLoom owns the **VisionIndex** document only.

## Document ownership

Do not mix product documents:

| Document | Owner | Contents |
| --- | --- | --- |
| **VisionIndex** | SightLoom | detections, tracks, masks, identities, appearances, visits, events, patterns, anomalies, evidence |
| **CaptureProject** | Capture product | media, audio, event streams, non-destructive edits, autosave, render targets |
| **SemanticEditPlan** | Intelligence product | intent, selectors, queries, privacy/uncertainty policy, target output |
| **RenderGraph** | Media product | deterministic executable media model |
| **ExecutionPlan** | Executor | FFmpeg / Rust / SightLoom materialization / GPU / encode stages |

## Workspace

```text
crates/
  sightloom-core        # geometry, Detection, NMS, zones, FrameStamp, EventEnvelope
  sightloom-tracking    # multi-object tracking, smoothers, trajectories
  sightloom-index       # Observation, masks, VisionIndex, JSON/CBOR package, optional SQLite
  sightloom-analysis    # zone analytics, pattern miners, statistical anomalies
  sightloom-reid        # subject gallery, embeddings, threshold resolver, audit
  sightloom             # facade: IndexSession end-to-end pipeline
```

| Crate | Role |
| --- | --- |
| `sightloom-core` | Portable geometry, compact detections, NMS, line/polygon zones, stamps, event envelopes |
| `sightloom-tracking` | Kalman association tracking, detection smoothing, trajectory history |
| `sightloom-index` | Rich observations, compact masks, VisionIndex memory, on-disk package |
| `sightloom-analysis` | Zone dwell/occupancy analytics, pattern mining, z-score anomaly backend |
| `sightloom-reid` | Subject references, embedding store, accept/reject/uncertain matching, merge/split, audit |
| `sightloom` | Host facade wiring track + re-id + zones into VisionIndex JSON/packages |

## Current capabilities

**Core**
- Finite points/rects, IoU/IoS, deterministic in-place NMS
- Compact `Detection` batches (caller-owned + optional owned)
- `FrameStamp` / `MediaTime` multi-source time
- Line and polygon zone monitors (`Entered` / `Exited` / `Crossed`)
- Portable `EventEnvelope` vocabulary

**Tracking (baseline association tracker)**
- Multi-object tracker with high/low confidence association (ByteTrack-style baseline)
- Kalman box filter, stable **local** track ids, lost buffer
- **Multi-source pool**: independent motion state per `SourceId`
- Composite `TrackKey { source_id, local_track_id }` and global `TrackUid`
- Exponential bbox smoothing, trajectory velocity/jitter
- Baseline CLEAR metrics helper (`MOTA`, precision/recall, ID switches, IDF1 approx)
  — not a full TrackEval/HOTA publish path yet

**Index / VisionIndex**
- Rich `Observation` above compact detections
- Dense / cropped / RLE / polygon masks, morphology, mask IoU
- In-memory VisionIndex: tracks, masks, events, appearances, visits, routes,
  zone stays, co-occurrences, source transitions, subjects, patterns, anomalies
- Track sample **revisions**: `sample_id`, `supersedes`, `revision`, effective view
- Subject query foundation: `SubjectQuery` + `execute_subject_query` (zone/time/dwell/confidence/page)
- JSON snapshot (`VisionIndexSnapshot`)
- On-disk package (`VisionIndexPackage`) with **transactional generations**:
  - `CURRENT` pointer → `gen-XXXXXXXX/`
  - per-generation `manifest.json`, `checksums.json` (FNV-1a),
    `tracks.cbor`, `masks.bin`, `events.cbor`, `entities.json`
  - optional `events.sqlite` (feature `sqlite`) for subject/track queries
  - legacy flat layouts still load
- Validation: `validate_fast` / `validate_full` / `repair_plan` with object paths

**Re-identification (P1 multi-factor baseline)**
- Subject modalities and reference samples (positive / negative / unlabeled)
- Embedding store with optional model name/version separation
- Multi-factor identity score: similarity × quality × temporal × topology × class × prior
- Camera topology gating (impossible hops cannot Accept)
- Per-source accept thresholds, reference eviction, multiple hypotheses in audit
- Uncertainty intervals from audit trail
- Gallery merge/split and manual confirmation
- `IndexSession` maps **TrackKey** → `SubjectId` (source-safe)
- Note: ANN backends, ROC/EER calibration, retention policy — later

**Analysis**
- Zone analytics: hysteresis, dwell, occupancy, anchors, class filter
- Pattern miners: time-of-day, day-of-week, visit periodicity, dwell distribution,
  route sequences, co-occurrence, expected absence, group formation
- Statistical anomaly backend: baseline stats + z-score detectors emitting
  backend-neutral `AnomalyEvent` values

## Pipeline

```text
external detections  ──┐
                       ├──► Observation / Detection
optional detector  ────┘
         │
         ▼
   multi-source tracking
   (per-SourceId ByteTrack baseline → TrackKey + TrackUid)
         │
         ├──► re-id (SubjectId, audit)
         ├──► smoothing / trajectory
         ├──► zone analytics (dwell, occupancy)
         ├──► pattern miners / statistical anomalies
         └──► VisionIndex  →  JSON / transactional package / session checkpoint
```

Facade entry point: `sightloom::IndexSession`

```text
ingest_detections (FrameStamp.source_id selects tracker; ingest policy)
seed_click / seed_subject_from_box / assign_subject / accept_host_track
note_track_embedding(TrackKey) → resolve_track_identity / resolve_pending_identities
uncertain_intervals / export_uncertain_intervals_json
export_track_spans / export_track_spans_json
mine_and_store_patterns / freeze_anomaly_baseline / detect_and_store_anomalies
query_subjects(SubjectQuery) / then_seen_in / route_contains
build_subject_reel / build_subject_reel_samples  # evidence handles only
enroll_subject_photos / search_by_photo / search_photo_with_reels  # reference photos
rank_subjects / most_frequent_subject_reel
rebuild_appearances_and_visits  # video memory from tracks
search_tracks_by_embedding  # unlabeled track embedding index
ingest_zone_updates
materialize_json / save_package / load_package  # package includes gallery.json
save_checkpoint / load_checkpoint   # full live-session resume
IngestPolicy + SourceWatermark + IngestMetrics  # streaming lifecycle contracts
```

Thin host sketch (fake detector, no render):

```bash
cargo run -p sightloom --example host_sketch
```

## Install (crates.io)

```toml
[dependencies]
sightloom = "0.1"
# or individual crates:
# sightloom-core = "0.1"
# sightloom-tracking = "0.1"
# sightloom-index = "0.1"
# sightloom-reid = "0.1"
# sightloom-analysis = "0.1"
```

```rust
use sightloom::IndexSession;
```

Workspace crates are versioned together as **0.1.x** (alpha API; expect evolution).

## Out of scope for this library

- video decode / encode / capture stacks
- pixel annotators, blur, overlays, GUI, notebooks
- model-specific inference SDKs
- CaptureProject / SemanticEditPlan / RenderGraph / ExecutionPlan ownership

SightLoom returns data. Host products render and capture.

## Verification

```powershell
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo test -p sightloom-core --no-default-features
cargo test -p sightloom-core --features alloc
cargo test -p sightloom-core --features std
cargo check -p sightloom-core --no-default-features --target riscv32imac-unknown-none-elf
cargo check -p sightloom-core --features alloc --target riscv32imac-unknown-none-elf
cargo check -p sightloom-tracking --no-default-features --target riscv32imac-unknown-none-elf
cargo check -p sightloom-analysis --no-default-features --target riscv32imac-unknown-none-elf
cargo check -p sightloom-reid --no-default-features --target riscv32imac-unknown-none-elf
cargo bench -p sightloom-core --bench core --no-run
cargo doc --workspace --all-features --no-deps
git diff --check
```

Geometry fixtures: [fixtures/geometry-reference](fixtures/geometry-reference)  
Provenance: [evidence/fixture-generation.md](evidence/fixture-generation.md)

## Maintainer publish notes

Releases are maintainer-only. **Never commit registry API tokens** or put them in
this README or any tracked file. Tokens live only as:

- a local shell environment variable for manual `cargo publish`, or
- a **GitHub Actions repository secret** used by `.github/workflows/publish.yml`
  on version tags (`v0.1.0`, …).

Scripts under `scripts/publish-crates.*` check that the environment secret is set
and refuse to run otherwise.

## License

SightLoom is licensed under the [MIT License](LICENSE).
