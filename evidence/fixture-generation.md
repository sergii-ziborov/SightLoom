# Supervision 0.30.0 fixture provenance

SightLoom uses Supervision 0.30.0 as a behavioral oracle for versioned
compatibility fixtures. No Python source or upstream implementation code is
committed to this repository.

## Isolated generation environment

- Generated at: `2026-08-11T08:31:11.260742+00:00`
- Python: 3.12.13
- Supervision: 0.30.0
- NumPy: 2.5.2
- Backend: pure NumPy; OpenCV was not installed
- Input dtype: `float64`
- Oracle overlap output dtype: `float32`
- Absolute tolerance: `1e-6`
- Relative tolerance: `1e-6`

The generator ran in a unique system-temporary virtual environment outside the
repository. The reproducible command recorded by the manifest is:

```text
venv/Scripts/python.exe generate_fixtures.py --output out
```

The environment was created with an explicit Python 3.12 interpreter, followed
by `pip install supervision==0.30.0`. The installed package version was read
from `supervision.__version__` before generation.

## Oracle APIs and coverage

- `supervision.box_iou_batch` generated scalar IoU and IoS values.
- `supervision.box_non_max_suppression` generated boolean keep masks.
- Cases cover empty input, touching edges, zero-area boxes, negative
  coordinates, equal scores, class-aware filtering, and thresholds 0.0, 0.5,
  and 1.0.

Supervision 0.30.0 keeps the second of two identical equal-score boxes in this
environment. SightLoom's approved deterministic contract instead prefers the
original lower input index. Task 5 must preserve that explicit SightLoom rule
and record the oracle case as an intentional compatibility difference.

## Integrity

| File | SHA-256 |
| --- | --- |
| `overlap.json` | `a0da3061fb896f5a54183f7dd7478e40c77aa4474746b80c66e6a574cce8787b` |
| `nms.json` | `f4d234120e5589b2f250eb0e967eb35f67e764e4723a6206661043684ed48af9` |

The hashes are also stored in `manifest.json` and are verified before the
fixtures are committed. These files prove behavior only for the pinned oracle
and recorded environment; they are not claims about other Supervision versions
or physical device execution.

## Rust characterization result

`cargo test -p sightloom-core --test compat_overlap` parsed the manifest and
all six overlap cases. Existing Rust IoU/IoS behavior matched every pinned
Supervision result within the recorded absolute and relative tolerances, so no
production implementation change was required. The test uses
`blazingly-json 0.1.4` for typed JSON deserialization. Coordinate lists are
explicitly checked for exactly four values before `Rect` construction. The
parser remains a dev-only dependency and does not enter the embedded runtime
graph.
