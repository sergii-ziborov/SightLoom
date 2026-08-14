# SightLoom evidence packs

Evaluation **harness** artifacts for tracking, re-id, and redaction quality.

These packs **do not** claim MOT17 leaderboard scores or production re-id accuracy
by themselves. They show how hosts generate portable reports.

## Generate a synthetic pack

```bash
cargo run -p sightloom-host --example write_evidence_pack -- ./evidence-out
```

Layout:

```text
evidence-out/
  SUMMARY.md
  manifest.json
  mot/
    suite.md              # synthetic CLEAR smoke table
    parallel_walk_gt.txt  # MOTChallenge GT
    parallel_walk_hyp.txt
    crossing_gt.txt
    crossing_hyp.txt
    TRACK_EVAL.md         # how to run external TrackEval
  reid/
    roc.md                # EER + recommended thresholds
    scores.csv            # labeled cosine pairs
  redaction/
    report.md             # leakage / collateral ratios
    samples.json          # host pixel counts
```

## Real datasets (host responsibility)

| Domain | What to do |
| --- | --- |
| MOT | Run host detector+tracker on MOT17/20; export via `write_mot_challenge_sequence`; evaluate with TrackEval offline |
| Re-id | Collect genuine/impostor pairs from your gallery; `compute_roc` / `LabeledScore` |
| Redaction | After host blur, fill `RedactionPixelSample` and `evaluate_redaction_pixels` |

## Geometry fixtures

See [fixture-generation.md](./fixture-generation.md) for pinned IoU/NMS cases under
`fixtures/geometry-reference/`.
