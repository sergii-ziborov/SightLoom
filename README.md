<div align="center">

# SightLoom

### Model-neutral video understanding and memory library

[![Status](https://img.shields.io/badge/status-crate%20consolidation-2563eb)](https://github.com/sergii-ziborov/SightLoom)
[![CI](https://github.com/sergii-ziborov/SightLoom/actions/workflows/ci.yml/badge.svg)](https://github.com/sergii-ziborov/SightLoom/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/Rust-1.97%2B-000000?logo=rust)](https://www.rust-lang.org/)
[![Target](https://img.shields.io/badge/target-no__std%20core-7c3aed)](https://docs.rust-embedded.org/book/intro/no-std.html)
[![License](https://img.shields.io/badge/license-MIT-2563eb)](LICENSE)

</div>

SightLoom is a **model-neutral video understanding and memory library**.
It accepts external detections (or frames through an optional detector adapter),
builds tracks and identities, emits events, and stores a queryable **VisionIndex**.

SightLoom returns **data** — subjects, track samples, masks, appearances,
confidence, events, patterns, anomalies, and evidence handles. Host products
draw pixels, run capture, and schedule render stages.

## Document ownership

Do not mix these documents:

| Document | Owner | Contents |
| --- | --- | --- |
| **VisionIndex** | SightLoom | detections, tracks, masks, identities, appearances, visits, events, patterns, anomalies, evidence |
| **CaptureProject** | Capture product | media, audio, event streams, non-destructive edits, autosave, render targets |
| **SemanticEditPlan** | Intelligence product | intent, selectors, queries, privacy/uncertainty policy, target output |
| **RenderGraph** | Media product | deterministic executable media model |
| **ExecutionPlan** | Executor | FFmpeg / Rust / SightLoom materialization / GPU / encode stages |

## Workspace crates

| Crate | Role |
| --- | --- |
| `sightloom-core` | Geometry, detections, NMS, zones, stamps, event envelopes |
| `sightloom-tracking` | Multi-object tracking, smoothers, trajectories |
| `sightloom-index` | Observations, masks, VisionIndex storage, JSON/CBOR package, optional SQLite |
| `sightloom-analysis` | Zone analytics, patterns, backend-neutral anomalies |
| `sightloom-reid` | Subject gallery, embeddings, threshold resolver, merge/split, audit |
| `sightloom` | Facade: `IndexSession` (track + re-id + zones → VisionIndex JSON) |

## Pipeline shape

```text
external detections  ──┐
                       ├──► Observation / Detection
optional detector  ────┘
         │
         ▼
   tracking (stable TrackId)
         │
         ├──► smoothing / trajectory
         ├──► zone analytics
         └──► VisionIndex (tracks, masks, events, identities, …)
```

Materialization exit path:

```text
detector results → stable tracks → re-id (subject_id) → masks → zone events → serialized VisionIndex
```

Use `sightloom::IndexSession`:
- `ingest_detections` / `ingest_zone_updates`
- `note_track_embedding` + `resolve_track_identity` / `resolve_pending_identities`
- `materialize_json()` / `save_package` / `load_package`

On-disk package layout (`VisionIndexPackage`):

```text
manifest.json   tracks.cbor   masks.bin   events.cbor   entities.json
events.sqlite   # when built with feature sqlite
```

Out of scope for SightLoom: video decode/encode, pixel annotators, GUI,
notebook helpers, and model-specific SDKs.

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
cargo bench -p sightloom-core --bench core --no-run
cargo doc --workspace --all-features --no-deps
git diff --check
```

Pinned geometry fixtures live under
[fixtures/geometry-reference](fixtures/geometry-reference) with provenance in
[evidence/fixture-generation.md](evidence/fixture-generation.md).

## License

SightLoom is licensed under the [MIT License](LICENSE).
