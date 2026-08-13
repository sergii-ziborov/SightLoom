# sightloom-tracking

Baseline multi-object tracking (ByteTrack-style association), Kalman box
filter, multi-source tracker pool, smoothers, and baseline MOT metrics for
[SightLoom](https://github.com/sergii-ziborov/SightLoom).

Includes CLEAR-style helpers (`evaluate_baseline_mot`) and deterministic
synthetic scenarios (`run_synthetic_parallel_walk`, `run_synthetic_crossing`)
for regression smoke tests — **not** published MOT17/MOT20 scores.

```toml
sightloom-tracking = "0.1"
```
