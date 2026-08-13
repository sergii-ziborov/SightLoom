#![cfg_attr(not(feature = "std"), no_std)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Subject references and identity resolution for `SightLoom`.
//!
//! Pipeline:
//! `Detection → TrackId → TrackFragment → embedding aggregation → SubjectId`.
//!
//! Provides:
//! - reference galleries with positive / negative samples
//! - threshold resolver with uncertain band
//! - merge / split subject clusters
//! - manual confirmation hooks
//! - audit trail for identity decisions

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
mod aggregate;
#[cfg(feature = "alloc")]
mod ann;
#[cfg(feature = "alloc")]
mod calibrate;
mod embedding;
#[cfg(feature = "alloc")]
mod gallery;
#[cfg(feature = "alloc")]
mod intervals;
#[cfg(feature = "alloc")]
mod resolver;
mod score;
#[cfg(feature = "alloc")]
mod search;
mod types;

pub use embedding::{EmbeddingError, cosine_similarity, mean_pool};
#[cfg(feature = "alloc")]
pub use score::{CameraEdge, CameraTopology};
pub use score::{IdentityScoreFactors, ScoreContext, class_compatibility, temporal_plausibility};
pub use types::{
    IdentityMatch, MatchDecision, ReferenceSample, SubjectModality, SubjectReference, TrackFragment,
};

#[cfg(feature = "alloc")]
pub use aggregate::{EmbeddingObservation, aggregate_fragment};
#[cfg(feature = "alloc")]
pub use ann::{
    AnnBackend, AnnHit, AnnIndex, AnnKind, BruteForceAnn, HnswAnn, HostAnnAdapter, LshAnn,
    search_with_host_ann,
};
#[cfg(feature = "alloc")]
pub use calibrate::{
    CalibrationReport, LabeledScore, RocPoint, compute_roc, labeled_scores_from_pairs,
    resolve_config_from_calibration,
};
#[cfg(feature = "alloc")]
pub use embedding::{EmbeddingModelId, EmbeddingStore};
#[cfg(feature = "alloc")]
pub use gallery::{IdentityAuditEvent, SubjectGallery};
#[cfg(feature = "alloc")]
pub use intervals::{
    IdentityInterval, IdentityPoint, coalesce_identity_intervals, interval_from_match,
    uncertain_only,
};
#[cfg(feature = "alloc")]
pub use resolver::{ResolveConfig, SubjectLastSeen, ThresholdResolver};
#[cfg(feature = "alloc")]
pub use search::{PhotoQuery, PhotoSearchHit, rank_subjects_by_cosine, search_gallery_by_photo};
#[cfg(feature = "alloc")]
pub use types::IdentityResolver;
