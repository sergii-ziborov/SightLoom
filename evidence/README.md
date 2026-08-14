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
    parallel_baseline_clear.md  # in-tree CLEAR rescore of exported text
    host_track_eval.md    # optional host-imported HOTA/MOTA/IDF1
  reid/
    roc.md                # EER + recommended thresholds
    scores.csv            # labeled cosine pairs
  redaction/
    report.md             # leakage / collateral ratios
    samples.json          # host pixel counts
  anomaly/
    far.md                # FAR calibration + subject/camera scopes
```

## TrackEval bridge

```rust,ignore
use sightloom_host::{MotEvidence, build_synthetic_mot_evidence};
use sightloom::tracking::ByteTrackConfig;

let mut mot = build_synthetic_mot_evidence(&ByteTrackConfig::default())?;
// After running external TrackEval on the exported gt/hyp files:
mot.attach_track_eval_summary_text(
    r#"{"sequence":"parallel_walk","evaluator":"TrackEval","mota":0.99,"hota":0.9,"idf1":0.95}"#,
)?;
// Or re-score with in-tree CLEAR: mot.rescore_parallel_baseline(0.5)?;
```

Session export for live tracks:

```rust,ignore
let hyp = session.export_mot_challenge(Some(SourceId(1)));
// write hyp next to your GT; run TrackEval offline; attach summary above
```

## Real datasets (host responsibility)

| Domain | What to do |
| --- | --- |
| MOT | Export via `write_mot_challenge_sequence` / `export_mot_challenge`; evaluate with TrackEval offline; `attach_track_eval_summary_text` |
| Re-id | Collect genuine/impostor pairs from your gallery; `compute_roc` / `LabeledScore` |
| Redaction | After host blur, fill `RedactionPixelSample` and `evaluate_redaction_pixels` |

## Geometry fixtures

See [fixture-generation.md](./fixture-generation.md) for pinned IoU/NMS cases under
`fixtures/geometry-reference/`.
