//! Step-3 **evidence packs**: portable MOT / re-id / redaction reports.
//!
//! These are **host-side evaluation artifacts**, not crates.io leaderboard claims.
//! Synthetic defaults prove the harness; hosts replace inputs with real datasets
//! and attach external `TrackEval` numbers.

mod anomaly;
mod mot;
mod pack;
mod redaction;
mod reid;

pub use anomaly::{AnomalyEvidence, build_synthetic_anomaly_evidence};
pub use mot::{MotEvidence, build_synthetic_mot_evidence};
// TrackEval summary types re-exported via sightloom::tracking.
pub use pack::{
    EvidencePack, EvidencePackPaths, build_synthetic_evidence_pack, write_evidence_pack,
};
pub use redaction::{RedactionEvidence, build_synthetic_redaction_evidence};
pub use reid::{ReidEvidence, build_synthetic_reid_evidence};
