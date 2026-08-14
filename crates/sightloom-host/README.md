# sightloom-host

**Step 1 host model package** for [SightLoom](https://github.com/sergii-ziborov/SightLoom).

SightLoom ranks and remembers. This crate owns the **host side** of:

```text
photo / frame → detect / embed → SightLoom IndexSession
```

## What step 1 includes

| Piece | Status |
| --- | --- |
| `HostBundleConfig` / `ModelSpec` / device preference | **yes** |
| Pure-Rust preprocess (resize, CHW normalize, crop) | **yes** |
| Local model cache registry (`FilesystemFetcher`) | **yes** (no network) |
| Reference detectors + embedders (deterministic, **no weights**) | **yes** |
| `HostPipeline` enroll / search / ingest | **yes** |
| Real ONNX Runtime | **not yet** (`onnx` feature reserved) |
| Auto model download | **not yet** |

## Quick start

```toml
sightloom-host = "0.1"
```

```rust,no_run
use sightloom_host::{FrameView, HostPipeline, PhotoView, PixelFormat};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut pipe = HostPipeline::new("demo")?;
    let rgb = vec![128_u8; 32 * 64 * 3];
    let frame = FrameView::new(32, 64, 32 * 3, PixelFormat::Rgb8, &rgb);
    let photo = PhotoView::from_frame(frame);
    let subject = pipe.enroll_photos(&[photo], false)?;
    let hits = pipe.search_photo(&photo, false, 3)?;
    assert_eq!(hits[0].subject_id, subject);
    Ok(())
}
```

```bash
cargo run -p sightloom-host --example photo_to_subject
```

## Honest boundary

Reference models **prove wiring**, not re-id accuracy. Drop real `.onnx` files
into `.sightloom-models/` (or set `ModelSpec.local_path`) when you add a runtime
backend in step 2+.

## Docs

[docs.rs/sightloom-host](https://docs.rs/sightloom-host) (after publish)
