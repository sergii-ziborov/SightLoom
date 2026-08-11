<div align="center">

# SightLoom

### Portable Rust primitives for deterministic vision events

[![Status](https://img.shields.io/badge/status-Phase%200--1%20core%20alpha-2563eb)](https://github.com/sergii-ziborov/SightLoom)
[![CI](https://github.com/sergii-ziborov/SightLoom/actions/workflows/ci.yml/badge.svg)](https://github.com/sergii-ziborov/SightLoom/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/Rust-1.97%2B-000000?logo=rust)](https://www.rust-lang.org/)
[![Target](https://img.shields.io/badge/target-no__std-7c3aed)](https://docs.rust-embedded.org/book/intro/no-std.html)
[![License](https://img.shields.io/badge/license-MIT-2563eb)](LICENSE)

</div>

SightLoom provides a portable, model-agnostic core for the geometry and
state transitions around computer-vision detections. It contains no inference
runtime, camera integration, codec stack, or board-specific HAL.

## Current capabilities

The `sightloom-core` crate is a `no_std`-capable, zero-runtime-dependency
library with optional `alloc` and `std` feature profiles. Its public API
includes:

- finite points and non-inverted, half-open rectangles;
- finite-score detections and caller-owned detection batches;
- intersection area, IoU, and IoS overlap metrics;
- deterministic, allocation-free, slice-first NMS with caller-provided
  ordering and suppression scratch space;
- line segments, polygon geometry, line/polygon zone monitors, and compact
  enter, exit, and crossing events.

The core preserves input ordering for retained NMS detections and resolves
equal scores by lower original input index. It accepts only finite geometry and
scores, so its overlap and NMS behavior is defined for zero-area rectangles as
well as ordinary boxes.

## Verification

The repository validates the core on the host and checks its two embedded-safe
feature profiles. Run the complete local gate with:

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

The property suite covers bounded finite geometry, IoU symmetry and range,
positive-area self-IoU, NMS idempotence, and zero-area geometry. Deterministic
Criterion benchmarks exercise pairwise IoU and the allocation-free NMS call at
16, 64, and 256 detections:

```powershell
cargo bench -p sightloom-core --bench core
```

The RISC-V checks demonstrate generic `riscv32imac-unknown-none-elf` compile
compatibility only. They are not Raspberry Pi, ESP32-S3, or ESP32-P4 runtime
validation, and no device performance claim is made here.

## Compatibility fixtures

[Roboflow Supervision](https://github.com/roboflow/supervision) 0.30.0 is an
MIT-licensed behavioral reference for selected overlap and filtering fixtures.
SightLoom owns its Rust API and documents its deterministic behavior where it
differs from the reference. Fixture source version, dtype, tolerance, hashes,
and generation provenance are recorded in
[evidence/fixture-generation.md](evidence/fixture-generation.md).

## License and affiliation

SightLoom is licensed under the [MIT License](LICENSE). It is independent of,
not endorsed by, and not an official product of Roboflow. See
[third-party notices](THIRD_PARTY_NOTICES.md) for referenced upstream projects.
