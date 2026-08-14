# sightloom

Facade crate for [SightLoom](https://github.com/sergii-ziborov/SightLoom): multi-source tracking,
re-identification, VisionIndex packages, and host export APIs.

**SightLoom understands and remembers video memory.** Hosts own pixels and models.

```toml
sightloom = "0.1"
```

## Quick start

```rust,no_run
use sightloom::core::{Detection, FrameStamp, MediaTime, Rect, SourceId};
use sightloom::tracking::ByteTrackConfig;
use sightloom::{IndexSession, SourceLifecycle};

let mut session = IndexSession::new("demo", ByteTrackConfig::default()).unwrap();
let stamp = FrameStamp::new(
    SourceId(1),
    0,
    MediaTime::new(0, 1_000_000_000).unwrap(),
    None,
);
let det = Detection::new(
    Rect::new(10.0, 10.0, 40.0, 80.0).unwrap(),
    0.9,
    None,
    None,
)
.unwrap();
session.ingest_detections(stamp, &[det]).unwrap();
session.apply_source_lifecycle(&SourceLifecycle::Reset {
    source_id: SourceId(1),
});
```

## Workspace crates

| Crate | Role |
| --- | --- |
| `sightloom-core` | Geometry, NMS, zones, stamps |
| `sightloom-tracking` | MOT baseline + multi-source |
| `sightloom-index` | VisionIndex + packages |
| `sightloom-reid` | Gallery + ANN + multi-factor score |
| `sightloom-analysis` | Patterns + anomaly backends |
| `sightloom` | This facade (`IndexSession`) |

## Examples

```bash
cargo run -p sightloom --example host_sketch
cargo run -p sightloom --example host_model_stub
```

## Docs

[docs.rs/sightloom](https://docs.rs/sightloom) · full architecture in the
[repository README](https://github.com/sergii-ziborov/SightLoom#readme).
