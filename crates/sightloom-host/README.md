# sightloom-host

Host model package for [SightLoom](https://github.com/sergii-ziborov/SightLoom).

SightLoom ranks and remembers. This crate owns:

```text
photo / frame → detect / embed (reference or ONNX) → SightLoom IndexSession
```

## Features

| Feature | What |
| --- | --- |
| `std` (default) | Config, preprocess, reference models, `HostPipeline` |
| `onnx` | Real ONNX via pure-Rust **tract** (`OnnxEmbedder`, `OnnxDetector`) |

Weights stay **on the host disk**. Nothing is downloaded automatically (step 2).

## Install

```toml
[dependencies]
sightloom-host = "0.1"   # crates.io line 0.1.5+
# optional ONNX:
# sightloom-host = { version = "0.1", features = ["onnx"] }
```

## Quick start (reference models — no weights)

```bash
cargo run -p sightloom-host --example photo_to_subject
```

## Step 2: ONNX weights

1. Create cache (optional helper):

```rust
use sightloom_host::write_cache_readme;
write_cache_readme(std::path::Path::new(".sightloom-models"))?;
```

2. Place a float32 ONNX file, e.g.:

```text
.sightloom-models/person_reid.onnx
```

**Embedder contract**

- Input: NCHW RGB `f32` (see `PreprocessConfig`, default ImageNet mean/std)
- Output: embedding vector `f32` (L2-normalized by `OnnxEmbedder`)

**Detector contract**

- Input: NCHW RGB `f32`
- Output: flat `N×6` (`x1,y1,x2,y2,score,class`) or YOLO-like rows

3. Run:

```bash
cargo run -p sightloom-host --features onnx --example onnx_photo_search
```

If weights are missing the example exits with code `2` and prints setup help (CI-friendly).

```rust,ignore
use sightloom_host::{EmbeddingTask, ModelSpec, ModelTask, OnnxEmbedder, PreprocessConfig};
use std::path::Path;

let mut spec = ModelSpec::embedder("person_reid", ModelTask::PersonReId, 512);
spec.local_path = Some(Path::new(".sightloom-models/person_reid.onnx").into());
spec.preprocess = PreprocessConfig::imagenet_like(128, 256);
let embedder = OnnxEmbedder::load(spec, Path::new(".sightloom-models"), EmbeddingTask::PersonReId)?;
```

### Why tract, not Microsoft ORT?

`ort` prebuilts are missing on some targets (e.g. `x86_64-pc-windows-gnu`).  
**tract-onnx** is pure Rust, portable, and enough for step-2 host integration.  
Hosts that want CUDA ORT can still implement `PhotoEmbeddingAdapter` themselves.

## Step 3: evidence packs

```bash
cargo run -p sightloom-host --example write_evidence_pack -- ./evidence-out
```

Writes MOT smoke + MOTChallenge export, re-id ROC/EER, and redaction pixel
reports. See [evidence/README.md](../../evidence/README.md).

```rust,no_run
use sightloom::tracking::ByteTrackConfig;
use sightloom_host::{build_synthetic_evidence_pack, write_evidence_pack};

let pack = build_synthetic_evidence_pack("demo", &ByteTrackConfig::default()).unwrap();
assert!(pack.all_smoke_pass());
write_evidence_pack(&pack, "./evidence-out").unwrap();
```

## Honest boundary

| Path | Accuracy claim |
| --- | --- |
| Reference embedders | wiring only |
| Your ONNX weights + tract | depends on **your** model |
| Synthetic evidence pack | harness smoke only — not MOT17 / production re-id |

## Docs

[docs.rs/sightloom-host](https://docs.rs/sightloom-host) (after publish)
