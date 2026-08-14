# sightloom-analysis

Zone dwell/occupancy analytics, pattern miners, and pluggable anomaly backends
for [SightLoom](https://github.com/sergii-ziborov/SightLoom).

```toml
sightloom-analysis = "0.1"
```

## Anomaly backends (`AnomalyDetector`)

| Backend | Type |
| --- | --- |
| `StatisticalAnomalyDetector` | z-score, MAD, CUSUM change-points |
| `IsolationForestDetector` | pure-Rust isolation forest |
| `OcSvmDetector` | pure-Rust RBF one-class SVM (not libsvm SMO) |

Hosts may also plug graph / quantum models behind the same trait.

## Pattern miners

Time-of-day, day-of-week, visit periodicity, dwell distribution, route
sequences, co-occurrence, expected absence, group formation,
event-before-event.

## Docs

[docs.rs/sightloom-analysis](https://docs.rs/sightloom-analysis)
