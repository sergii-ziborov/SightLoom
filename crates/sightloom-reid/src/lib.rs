#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::cast_precision_loss)]
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
mod embedding;
#[cfg(feature = "alloc")]
mod gallery;
#[cfg(feature = "alloc")]
mod resolver;
mod types;

pub use embedding::{EmbeddingError, cosine_similarity, mean_pool};
pub use types::{
    IdentityMatch, MatchDecision, ReferenceSample, SubjectModality, SubjectReference, TrackFragment,
};

#[cfg(feature = "alloc")]
pub use aggregate::{EmbeddingObservation, aggregate_fragment};
#[cfg(feature = "alloc")]
pub use embedding::EmbeddingStore;
#[cfg(feature = "alloc")]
pub use gallery::{IdentityAuditEvent, SubjectGallery};
#[cfg(feature = "alloc")]
pub use resolver::{ResolveConfig, ThresholdResolver};
#[cfg(feature = "alloc")]
pub use types::IdentityResolver;
