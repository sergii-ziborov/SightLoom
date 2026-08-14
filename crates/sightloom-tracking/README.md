# sightloom-tracking

Baseline multi-object tracking (ByteTrack-style association), Kalman box
filter, multi-source tracker pool, smoothers, and MOT helpers for
[SightLoom](https://github.com/sergii-ziborov/SightLoom).

```toml
sightloom-tracking = "0.1"
```

## Highlights

- High/low confidence association tracker + Kalman box filter
- **Multi-source pool**: independent motion per `SourceId`, global `TrackUid`
- `reset_source` / `remove_source` for host reconnect
- Exponential bbox smoothing + trajectory history
- CLEAR-style metrics (`evaluate_baseline_mot`)
- Synthetic scenarios + **`run_mot_smoke_suite`** report
- `MOTChallenge` text export for host-side TrackEval

**Honest boundary:** not published MOT17/MOT20 leaderboard scores.

## Docs

[docs.rs/sightloom-tracking](https://docs.rs/sightloom-tracking)
