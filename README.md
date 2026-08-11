<div align="center">

# SightLoom

### Rust-native vision pipelines for edge and embedded systems

Turn model detections into deterministic geometry, filtering, zones, and
compact events — from servers and Raspberry Pi-class computers down to
constrained microcontrollers.

[![Status](https://img.shields.io/badge/status-early%20development-f59e0b)](https://github.com/sergii-ziborov/SightLoom)
[![Rust](https://img.shields.io/badge/Rust-1.97%2B-000000?logo=rust)](https://www.rust-lang.org/)
[![Target](https://img.shields.io/badge/target-no__std-7c3aed)](https://docs.rust-embedded.org/book/intro/no-std.html)
[![License](https://img.shields.io/badge/license-MIT-2563eb)](LICENSE)

</div>

## What SightLoom is

SightLoom is a model-agnostic computer-vision application layer written in
Rust. It sits between an inference backend and an application:

```text
frames or external detections
            │
            ▼
      validated detections
            │
            ▼
 geometry ─ NMS ─ tracking ─ zones
            │
            ▼
    compact events and overlays
```

The project focuses on everything that happens *around* a vision model:
validated boxes, overlap filtering, tracking interfaces, spatial zones,
counters, compact masks, event generation, and eventually lightweight frame
annotation.

It does not train models and it is not tied to a single detector or inference
runtime.

## Why another vision library?

Many practical computer-vision pipelines are easy to prototype but difficult
to deploy on small devices. Python, NumPy, OpenCV, native codec stacks, and
framework-specific objects can turn a compact idea into a large and fragile
runtime.

SightLoom is built around a different set of constraints:

| Principle | Meaning |
|---|---|
| **Small-device first** | The behavioral core is designed for `no_std`, fixed memory, and caller-owned buffers. |
| **No hidden allocation** | Hot-path APIs receive their output and scratch storage from the caller. |
| **Deterministic** | Ordering, float boundaries, buffer exhaustion, and invalid inputs have documented behavior. |
| **Backend independent** | Model runtimes, cameras, codecs, and board HALs stay outside the core. |
| **Evidence before claims** | Host, Raspberry Pi, and ESP results are reported separately and only after real execution. |

## Target profiles

### Portable core

A heap-free Rust library for geometry, detections, IoU/IoS, deterministic NMS,
zones, and compact events. The base profile requires neither `std` nor a global
allocator.

### Pure-Rust edge

A future single-binary profile for Rust-native image handling and CPU inference
on Linux edge systems. RTen and tract are candidates, but the selection will be
made from same-model benchmarks rather than preference.

### Native edge adapters

Optional adapters for system camera and codec stacks such as Raspberry Pi
`libcamera`, FFmpeg, GStreamer, and vendor accelerators. Native dependencies
will never leak into the portable core.

### Embedded

Fixed-capacity detection processing, zones, counters, and event generation for
devices such as ESP32-S3 and ESP32-P4. On-device inference is a separate
research track; the first useful embedded profile consumes detections produced
by a sensor or external accelerator.

## Current focus

SightLoom is in early development. The first core alpha is focused on:

- finite, validated point and rectangle types;
- allocation-free IoU, IoS, and deterministic NMS;
- caller-owned detection batches;
- line and polygon zone state;
- compact enter, exit, and crossing events;
- differential behavior fixtures from Supervision 0.30.0;
- host benchmarks and explicit `NOT RUN` markers for unavailable hardware.

No crate has been published to crates.io yet, and no Raspberry Pi or ESP
runtime performance claim has been made.

## Compatibility philosophy

[Roboflow Supervision](https://github.com/roboflow/supervision) 0.30.0 is used
as an MIT-licensed behavioral reference for selected geometry and filtering
operations. SightLoom owns its Rust API and intentionally chooses deterministic
embedded-friendly behavior where the reference behavior is unstable or tied
to Python/NumPy details.

Compatibility fixtures will record their source version, dtype, tolerance, and
generation provenance.

## Development

The repository is being implemented incrementally with test-driven development.
Build and test commands will appear here with the first Rust core commit. Public
source, tests, fixtures, and evidence reports remain reviewable; local planning
documents are intentionally excluded from version control.

## License and affiliation

SightLoom is licensed under the [MIT License](LICENSE).

SightLoom is an independent project. It is not affiliated with, endorsed by,
or an official product of Roboflow. See [third-party notices](THIRD_PARTY_NOTICES.md)
for referenced upstream projects.
