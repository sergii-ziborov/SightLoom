<div align="center">

# SightLoom

### Model-neutral video understanding and memory library

[![Status](https://img.shields.io/badge/status-M0%20contracts-2563eb)](https://github.com/sergii-ziborov/SightLoom)
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

## M0 contracts (this release)

Portable contracts already in tree:

- `FrameStamp`, `MediaTime`, `SourceId`
- compact `Detection` and rich `Observation`
- `MaskRef`, `TrackSample`, `SubjectId`, `EventId`
- `EventEnvelope` / `EventKind` / `EventPayload`
- **VisionIndex** header + in-memory document with appearances, visits, routes,
  zone stays, co-occurrences, source transitions, subject profiles, patterns,
  and backend-neutral `AnomalyEvent`

## Workspace crates

| Crate | Role |
| --- | --- |
| `sightloom-core` | Geometry, detections, NMS, zones, stamps, event envelopes |
| `sightloom-obs` | Rich `Observation` |
| `sightloom-mask` | Compact masks and morphology |
| `sightloom-track` | Multi-object tracking (Kalman + ByteTrack-style association) |
| `sightloom-smooth` | Smoothing and trajectories |
| `sightloom-analytics` | Zone dwell / occupancy analytics |
| `sightloom-memory` | VisionIndex storage, track/mask/event stores |

Target consolidation (not all renamed yet): `core`, `tracking`, `index`,
`reid`, `analysis`, facade `sightloom`.

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
