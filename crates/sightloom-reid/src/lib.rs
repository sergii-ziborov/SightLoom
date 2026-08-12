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

mod aggregate;
mod embedding;
mod gallery;
mod resolver;
mod types;

pub use aggregate::EmbeddingObservation;
pub use embedding::{EmbeddingError, EmbeddingStore, cosine_similarity, mean_pool};
pub use types::{
    IdentityMatch, IdentityResolver, MatchDecision, ReferenceSample, SubjectModality,
    SubjectReference, TrackFragment,
};

#[cfg(feature = "alloc")]
pub use aggregate::aggregate_fragment;
#[cfg(feature = "alloc")]
pub use gallery::{IdentityAuditEvent, SubjectGallery};
#[cfg(feature = "alloc")]
pub use resolver::{ResolveConfig, ThresholdResolver};
