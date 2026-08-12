<div align="center">

# SightLoom

### Model-neutral video understanding and memory library

[![Status](https://img.shields.io/badge/status-P0%20video%20understanding-2563eb)](https://github.com/sergii-ziborov/SightLoom)
[![CI](https://github.com/sergii-ziborov/SightLoom/actions/workflows/ci.yml/badge.svg)](https://github.com/sergii-ziborov/SightLoom/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/Rust-1.97%2B-000000?logo=rust)](https://www.rust-lang.org/)
[![Target](https://img.shields.io/badge/target-no__std%20core-7c3aed)](https://docs.rust-embedded.org/book/intro/no-std.html)
[![License](https://img.shields.io/badge/license-MIT-2563eb)](LICENSE)

</div>

SightLoom is a **model-neutral video understanding and memory library**.
It turns detections (or frames via an optional detector adapter) into tracks,
identities, events, and queryable video memory. It is not a full port of
Roboflow Supervision, and it does not own video I/O or pixel drawing.

**SightLoom returns data** — subjects, track samples, masks, appearances,
confidence, and evidence handles. A host compositor (for example ReelForge)
draws boxes, blur, and overlays.

## Workspace crates

| Crate | Role |
| --- | --- |
| `sightloom-core` | Portable geometry, compact `Detection`, NMS, enter/exit/cross zones |
| `sightloom-obs` | Rich `Observation` above compact detections |
| `sightloom-mask` | Dense / cropped / RLE / polygon masks, IoU, morphology, convert |
| `sightloom-track` | Kalman filter, greedy IoU matching, ByteTrack-compatible tracker |
| `sightloom-smooth` | Detection smoothing, trajectory history, velocity / jitter |
| `sightloom-analytics` | Zone dwell, occupancy, hysteresis, anchor policy, class filter |
| `sightloom-memory` | Versioned manifest, track stream, mask store, event/subject index |

## Pipeline shape

```text
external detections  ──┐
                       ├──► Observation / Detection
optional detector  ────┘
         │
         ▼
   ByteTrack (stable TrackId)
         │
         ├──► smoothing / trajectory
         ├──► zone analytics (dwell, occupancy)
         └──► video memory (tracks, masks, events, provenance)
```

What SightLoom intentionally does **not** include: video decode/encode,
`VideoSink`, pixel annotators, GUI helpers, notebook helpers, dataset-conversion
zoos, or model-specific Python wrappers. Those belong to host products.

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

## Compatibility fixtures

[Roboflow Supervision](https://github.com/roboflow/supervision) 0.30.0 is an
MIT-licensed behavioral reference for selected overlap and filtering fixtures.
SightLoom owns its Rust API. Fixture provenance lives in
[evidence/fixture-generation.md](evidence/fixture-generation.md).

## License and affiliation

SightLoom is licensed under the [MIT License](LICENSE). It is independent of,
not endorsed by, and not an official product of Roboflow. See
[third-party notices](THIRD_PARTY_NOTICES.md) for referenced upstream projects.
