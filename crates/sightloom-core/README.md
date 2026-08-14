# sightloom-core

Portable geometry, compact detections, NMS, zone monitors, stamps, and event
envelopes for [SightLoom](https://github.com/sergii-ziborov/SightLoom).

`no_std` by default; enable `alloc` / `std` as needed.

```toml
sightloom-core = "0.1"
```

## Features

| Feature | What it enables |
| --- | --- |
| (default empty) | Pure geometry / detections without heap |
| `alloc` | Owned detection batches, growable helpers |
| `std` | Host-friendly extras (`Error` impls) |

## Highlights

- Finite `Point` / `Rect` / polygon / line geometry with IoU / IoS
- Hard NMS, **soft-NMS**, **merge-NMS**
- Inference **tiling** (`generate_tiles` / `tile_to_global`) for large frames
- Line and polygon zone monitors (`Entered` / `Exited` / `Crossed`)
- `FrameStamp` / `MediaTime` multi-source time
- Portable `EventEnvelope` vocabulary

## Docs

[docs.rs/sightloom-core](https://docs.rs/sightloom-core)
