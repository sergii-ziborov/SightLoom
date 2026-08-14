# sightloom-index

`VisionIndex` document model, compact masks (Moore contour tracing),
transactional packages, Arrow-shaped track export (`SLARROW1`), subject
queries, and evidence reel builders for
[SightLoom](https://github.com/sergii-ziborov/SightLoom).

```toml
sightloom-index = "0.1"
```

## Features

| Feature | What it enables |
| --- | --- |
| `std` (default via facade) | In-memory index, queries, snapshots |
| `package` | Transactional on-disk generations (CBOR tracks) |
| `sqlite` | Optional `events.sqlite` query sidecar |
| `cbor` | CBOR track/event codecs |

## Highlights

- Dense / cropped / RLE / polygon masks + morphology + **Moore contours**
- Track sample revisions (`sample_id` / `supersedes` / `revision`)
- Subject query AST, spatial query, streaming cursor, keyword NL bridge
- Auto appearances / visits / subject profiles
- Package generations: `CURRENT` → `gen-XXXXXXXX/`
- `encode_track_arrow` / `decode_track_arrow` analytics sidecar (package default remains CBOR)

## Docs

[docs.rs/sightloom-index](https://docs.rs/sightloom-index)
