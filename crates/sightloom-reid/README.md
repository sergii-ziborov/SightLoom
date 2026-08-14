# sightloom-reid

Subject gallery, multi-factor identity scoring, uncertainty intervals, and
embedding search for [SightLoom](https://github.com/sergii-ziborov/SightLoom).

```toml
sightloom-reid = "0.1"
```

## ANN (pure Rust)

| Backend | Notes |
| --- | --- |
| `BruteForceAnn` | Exact top-k |
| `LshAnn` | Random-projection LSH |
| `HnswAnn` | Graph ANN (`AnnKind::hnsw_default`) |
| `HostAnnAdapter` | Host FAISS/HNSWlib — **no FAISS link in this crate** |

## Calibration

ROC/EER from labeled pairs (`compute_roc`) → recommended accept/reject
thresholds for `ResolveConfig`.

## Topology

`CameraTopology` gates impossible cross-camera hops
(`set_edge` / `set_bidirectional`, `strict_camera_topology`).

## Docs

[docs.rs/sightloom-reid](https://docs.rs/sightloom-reid)
