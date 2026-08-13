# sightloom-reid

Subject gallery, multi-factor identity scoring, uncertainty intervals, and
embedding search for [SightLoom](https://github.com/sergii-ziborov/SightLoom).

**ANN (pure Rust):** `BruteForceAnn`, random-projection `LshAnn`, graph
`HnswAnn` (`AnnKind::hnsw_default`). External FAISS/HNSWlib via
`HostAnnAdapter` only — this crate does not link FAISS.

**Calibration:** ROC/EER from labeled pairs (`compute_roc`) → recommended
accept/reject thresholds for `ResolveConfig`.

```toml
sightloom-reid = "0.1"
```
