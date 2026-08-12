//! `VisionIndex` — `SightLoom`-owned queryable video understanding document.
//!
//! This document is **not** a `CaptureProject`, `SemanticEditPlan`,
//! `RenderGraph`, or `ExecutionPlan`. Those belong to sibling products and must
//! not be mixed into this schema.

use crate::{
    AnomalyEvent, Appearance, CoOccurrence, EventIndex, MaskStore, MemoryError, MemoryManifest,
    ModelProvenance, PatternRecord, Route, SourceEntry, SourceHash, SourceTransition,
    SubjectProfile, TrackSample, TrackStream, Visit, ZoneStay,
};
use sightloom_core::EventEnvelope;

/// Schema version written by this crate for `VisionIndex` documents.
pub const VISION_INDEX_VERSION: u32 = 1;

/// Header describing a `VisionIndex` package on disk or in memory.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct VisionIndexHeader {
    /// Schema version.
    pub version: u32,
    /// Human-readable index name.
    pub name: String,
    /// Media sources referenced by stamps and samples.
    pub sources: Vec<SourceEntry>,
    /// Relative path to the track sample stream.
    pub track_stream_path: String,
    /// Relative path to the compact mask store.
    pub mask_store_path: String,
    /// Relative path to the event envelope index.
    pub event_index_path: String,
    /// Relative path to appearance/visit/identity tables.
    pub entity_store_path: String,
    /// Optional model/threshold provenance.
    pub provenance: Option<ModelProvenance>,
}

impl VisionIndexHeader {
    /// Creates a v1 header with default relative paths.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            version: VISION_INDEX_VERSION,
            name: name.into(),
            sources: Vec::new(),
            track_stream_path: "tracks.cbor".into(),
            mask_store_path: "masks.bin".into(),
            event_index_path: "events.idx".into(),
            entity_store_path: "entities.json".into(),
            provenance: None,
        }
    }

    /// Validates version and required path fields.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Invalid`] when the header is unusable.
    pub fn validate(&self) -> Result<(), MemoryError> {
        if self.version == 0 || self.version > VISION_INDEX_VERSION {
            return Err(MemoryError::Invalid);
        }
        if self.name.is_empty()
            || self.track_stream_path.is_empty()
            || self.mask_store_path.is_empty()
            || self.event_index_path.is_empty()
            || self.entity_store_path.is_empty()
        {
            return Err(MemoryError::Invalid);
        }
        Ok(())
    }
}

#[cfg(feature = "std")]
impl VisionIndexHeader {
    /// Serializes the header to pretty JSON.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Serde`] on failure.
    pub fn to_json(&self) -> Result<String, MemoryError> {
        serde_json::to_string_pretty(self).map_err(|error| MemoryError::Serde(error.to_string()))
    }

    /// Parses and validates a header from JSON.
    ///
    /// # Errors
    ///
    /// Returns serde or validation errors.
    pub fn from_json(text: &str) -> Result<Self, MemoryError> {
        let header: Self =
            serde_json::from_str(text).map_err(|error| MemoryError::Serde(error.to_string()))?;
        header.validate()?;
        Ok(header)
    }
}

/// In-memory `VisionIndex` document.
///
/// Holds the queryable understanding surface for one analysis package:
/// detections/tracks/masks, identities, appearances, visits, events, patterns,
/// anomalies, and evidence handles.
#[cfg(feature = "std")]
#[derive(Clone, Debug)]
pub struct VisionIndex {
    /// Document header / sidecar layout.
    pub header: VisionIndexHeader,
    /// Track samples.
    pub tracks: TrackStream,
    /// Compact masks.
    pub masks: MaskStore,
    /// Event envelopes (also mirrored into a lightweight kind index).
    pub events: Vec<EventEnvelope>,
    /// Legacy kind/subject index helpers.
    pub event_index: EventIndex,
    /// Appearances.
    pub appearances: Vec<Appearance>,
    /// Visits.
    pub visits: Vec<Visit>,
    /// Routes.
    pub routes: Vec<Route>,
    /// Zone stays.
    pub zone_stays: Vec<ZoneStay>,
    /// Co-occurrences.
    pub co_occurrences: Vec<CoOccurrence>,
    /// Cross-source transitions.
    pub source_transitions: Vec<SourceTransition>,
    /// Subject profiles.
    pub subjects: Vec<SubjectProfile>,
    /// Patterns.
    pub patterns: Vec<PatternRecord>,
    /// Anomalies.
    pub anomalies: Vec<AnomalyEvent>,
}

#[cfg(feature = "std")]
impl VisionIndex {
    /// Creates an empty named `VisionIndex`.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            header: VisionIndexHeader::new(name),
            tracks: TrackStream::new(),
            masks: MaskStore::new(),
            events: Vec::new(),
            event_index: EventIndex::new(),
            appearances: Vec::new(),
            visits: Vec::new(),
            routes: Vec::new(),
            zone_stays: Vec::new(),
            co_occurrences: Vec::new(),
            source_transitions: Vec::new(),
            subjects: Vec::new(),
            patterns: Vec::new(),
            anomalies: Vec::new(),
        }
    }

    /// Registers a media source on the header.
    pub fn add_source(&mut self, entry: SourceEntry) {
        self.header.sources.push(entry);
    }

    /// Attaches model provenance.
    pub fn set_provenance(&mut self, provenance: ModelProvenance) {
        self.header.provenance = Some(provenance);
    }

    /// Appends a track sample.
    pub fn push_track(&mut self, sample: TrackSample) {
        self.tracks.push(sample);
    }

    /// Appends an event envelope.
    pub fn push_event(&mut self, envelope: EventEnvelope) {
        let kind = event_kind_name(envelope.kind);
        self.event_index.insert(
            kind,
            envelope.track_id,
            envelope.subject_id,
            envelope.zone_id,
            envelope.stamp.pts.as_nanos(),
            None,
        );
        self.events.push(envelope);
    }

    /// Validates the document header.
    ///
    /// # Errors
    ///
    /// Propagates header validation errors.
    pub fn validate(&self) -> Result<(), MemoryError> {
        self.header.validate()
    }

    /// Builds a legacy [`MemoryManifest`] view of the header for older callers.
    #[must_use]
    pub fn to_memory_manifest(&self) -> MemoryManifest {
        MemoryManifest {
            version: self.header.version,
            name: self.header.name.clone(),
            sources: self.header.sources.clone(),
            track_stream_path: self.header.track_stream_path.clone(),
            mask_store_path: self.header.mask_store_path.clone(),
            event_index_path: self.header.event_index_path.clone(),
            provenance: self.header.provenance.clone(),
        }
    }
}

/// Helper to attach an optional content hash to a source entry at construction.
#[must_use]
pub fn source_entry(
    source_id: u32,
    uri: impl Into<String>,
    hash: Option<SourceHash>,
) -> SourceEntry {
    SourceEntry {
        source_id,
        uri: uri.into(),
        hash,
    }
}

fn event_kind_name(kind: sightloom_core::EventKind) -> &'static str {
    use sightloom_core::EventKind::{Anomaly, Custom, Dwell, Identity, Occupancy, Pattern, Zone};
    match kind {
        Zone => "zone",
        Dwell => "dwell",
        Occupancy => "occupancy",
        Identity => "identity",
        Pattern => "pattern",
        Anomaly => "anomaly",
        Custom => "custom",
    }
}
