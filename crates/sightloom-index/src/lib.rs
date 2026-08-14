//! Observations, compact masks, and `VisionIndex` storage for `SightLoom`.
//!
//! `SightLoom` owns the **`VisionIndex`** document. Sibling products own
//! separate documents that must not be mixed in:
//! - `CaptureProject`
//! - `SemanticEditPlan`
//! - `RenderGraph`
//! - `ExecutionPlan`

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

mod attributes;
mod entities;
mod error;
mod event_index;
#[cfg(feature = "std")]
mod evidence;
#[cfg(feature = "alloc")]
mod keypoints;
mod manifest;
mod mask;
mod mask_store;
#[cfg(feature = "std")]
mod memory_build;
#[cfg(feature = "std")]
mod nl;
mod observation;
mod oriented;
#[cfg(feature = "package")]
mod package;
mod provenance;
#[cfg(feature = "std")]
mod query;
#[cfg(feature = "std")]
mod query_ast;
#[cfg(feature = "std")]
mod ranking;
mod snapshot;
mod store;
#[cfg(feature = "std")]
mod streaming_query;
mod track_stream;
#[cfg(feature = "std")]
mod validate;
mod vision_index;

pub use attributes::ObservationAttributes;
pub use entities::{
    Appearance, CoOccurrence, RedactionIntent, RedactionInterval, Route, SourceTransition,
    SubjectProfile, Visit, ZoneStay,
};
pub use error::MemoryError;
pub use event_index::EventRecord;
#[cfg(feature = "alloc")]
pub use keypoints::{Keypoint, KeypointSet, KeypointStore};
pub use manifest::{MANIFEST_VERSION, MemoryManifest, SourceEntry};
pub use mask::{
    CroppedMask, DenseMask, MaskError, PolygonMask, RleMask, bbox_to_polygon, cropped_mask_iou,
    cropped_to_polygon_approx, dense_mask_difference, dense_mask_iou, dense_mask_union,
    dense_to_bbox, dense_to_contour, dense_to_rle, dilate, erode, feather, fill_holes,
    mask_nms_by_iou, polygon_to_dense, rle_to_dense,
};
#[cfg(feature = "alloc")]
pub use mask::dense_to_contours;
pub use observation::{Observation, effective_observations, observation_idempotency_seen};
pub use oriented::{OrientedDetection, OrientedRect, oriented_aabb_iou, oriented_nms_aabb};
pub use provenance::{ModelProvenance, SourceHash};
pub use track_stream::TrackSample;
pub use vision_index::{VISION_INDEX_VERSION, VisionIndexHeader, source_entry};

// Re-export analysis entity types commonly stored in the index.
pub use sightloom_analysis::{AnomalyEvent, AnomalyReason, PatternKind, PatternRecord, Severity};

#[cfg(feature = "std")]
pub use event_index::EventIndex;
#[cfg(feature = "std")]
pub use mask_store::MaskStore;
#[cfg(feature = "std")]
pub use snapshot::{
    AnomalyEventDto, AppearanceDto, CoOccurrenceDto, EventEnvelopeDto, EvidenceReelDto,
    MediaTimeDto, ObservationDto, PatternRecordDto, RedactionIntervalDto, ReelSegmentDto, RouteDto,
    SourceTransitionDto, SubjectProfileDto, TrackSampleDto, VisionIndexSnapshot, VisitDto,
    ZoneStayDto,
};
#[cfg(feature = "std")]
pub use store::VideoMemory;
#[cfg(feature = "std")]
pub use track_stream::TrackStream;
#[cfg(feature = "std")]
pub use vision_index::VisionIndex;

#[cfg(feature = "std")]
pub use evidence::{EvidenceReel, EvidenceReelBuilder, ReelId, ReelSegment, build_subject_reel};
#[cfg(feature = "std")]
pub use memory_build::{
    MemoryBuildConfig, RedactionSpec, append_redaction_intervals, build_appearances,
    build_redaction_from_appearances, build_redaction_from_specs, build_subject_profiles,
    build_visits, rebuild_memory_entities, rebuild_subject_profiles, set_redaction_intervals,
};
#[cfg(feature = "std")]
pub use nl::{NlParseError, NlParseResult, parse_nl_query};
#[cfg(all(feature = "package", feature = "sqlite"))]
pub use package::sqlite_query;
#[cfg(feature = "package")]
pub use package::{CURRENT_FILE, GALLERY_FILE, MANIFEST_FILE, VisionIndexPackage};
#[cfg(feature = "std")]
pub use query::{
    Page, QueryOrder, SpatialHit, SpatialQuery, SubjectHit, SubjectQuery, ThenSeenIn,
    execute_spatial_query, execute_subject_query,
};
#[cfg(feature = "std")]
pub use query_ast::{QueryNode, SubjectPredicate, execute_query_ast};
#[cfg(feature = "std")]
pub use ranking::{SubjectRank, most_frequent_subject, rank_subjects_by_frequency};
#[cfg(feature = "std")]
pub use streaming_query::{QueryCursor, StreamingSubjectQuery};
#[cfg(feature = "std")]
pub use validate::{ValidationIssue, ValidationReport, ValidationSeverity};
