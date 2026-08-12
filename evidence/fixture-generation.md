# Geometry reference fixture provenance

SightLoom pins a small suite of geometry and NMS cases under
`fixtures/geometry-reference/` to lock numerical contracts for overlap and
filtering. No third-party source code is vendored for these fixtures.

## Numeric contract

- Absolute tolerance: `1e-6`
- Relative tolerance: `1e-6`
- Input coordinates: finite `f32` rectangles
- Metrics covered: IoU, IoS
- NMS: class-aware, descending score, lower original index as equal-score
  tie-break

## Integrity

| File | SHA-256 |
| --- | --- |
| `overlap.json` | `ece0e53d8deb2b4ab512c5fb716fff4cfdc76e4b0fd2e3736bae2b12a3df4001` |
| `nms.json` | `364f2dd942d23e630cc59061315e1fda82cae008aee6c00b5eab3135e5d1c42d` |

Hashes are also stored in `manifest.json`.

## Rust characterization

`cargo test -p sightloom-core --test compat_overlap` and
`cargo test -p sightloom-core --test compat_nms` exercise the pinned cases.
Parsers use the dev-only `blazingly-json` dependency and do not enter the
embedded runtime graph.
