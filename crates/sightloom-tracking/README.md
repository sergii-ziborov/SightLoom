# sightloom-tracking

Baseline multi-object tracking (ByteTrack-style association), Kalman box
filter, multi-source tracker pool, smoothers, and baseline MOT metrics for
[SightLoom](https://github.com/sergii-ziborov/SightLoom).

Includes CLEAR-style helpers (`evaluate_baseline_mot`), deterministic synthetic
scenarios (`run_synthetic_parallel_walk`, `run_synthetic_crossing`, occlusion /
triple-lane), a multi-scenario **`run_mot_smoke_suite`** report, and
`MOTChallenge` text export for host-side TrackEval — **not** published
MOT17/MOT20 scores.

```toml
sightloom-tracking = "0.1"
```
