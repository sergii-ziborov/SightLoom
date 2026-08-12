<div align="center">

# SightLoom

### Model-neutral video understanding and memory library

[![Status](https://img.shields.io/badge/status-VisionIndex%20alpha-2563eb)](https://github.com/sergii-ziborov/SightLoom)
[![CI](https://github.com/sergii-ziborov/SightLoom/actions/workflows/ci.yml/badge.svg)](https://github.com/sergii-ziborov/SightLoom/actions/workflows/ci.yml)
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

**Tracking**
- Multi-object tracker with high/low confidence association
- Kalman box filter, stable track ids, lost buffer
- Exponential bbox smoothing, trajectory velocity/jitter

**Index / VisionIndex**
- Rich `Observation` above compact detections
- Dense / cropped / RLE / polygon masks, morphology, mask IoU
- In-memory VisionIndex: tracks, masks, events, appearances, visits, routes,
  zone stays, co-occurrences, source transitions, subjects, patterns, anomalies
- JSON snapshot (`VisionIndexSnapshot`)
- On-disk package (`VisionIndexPackage`):
  - `manifest.json`, `tracks.cbor`, `masks.bin`, `events.cbor`, `entities.json`
  - optional `events.sqlite` (feature `sqlite`) for subject/track queries

**Re-identification**
- Subject modalities and reference samples (positive / negative / unlabeled)
- Embedding store, cosine similarity, fragment aggregation
- Threshold resolver with uncertain band
- Gallery merge/split and manual confirmation audit trail
- `IndexSession` maps track ids to subjects and stamps samples/events

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
   tracking (stable TrackId)
         │
         ├──► re-id (SubjectId, audit)
         ├──► smoothing / trajectory
         ├──► zone analytics (dwell, occupancy)
         ├──► pattern miners / statistical anomalies
         └──► VisionIndex  →  JSON snapshot / on-disk package
```

Facade entry point: `sightloom::IndexSession`

```text
ingest_detections
note_track_embedding → resolve_track_identity / resolve_pending_identities
ingest_zone_updates
materialize_json / save_package / load_package
```

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

## License

SightLoom is licensed under the [MIT License](LICENSE).
