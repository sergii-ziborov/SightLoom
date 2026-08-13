//! Typed identifiers shared across `SightLoom` crates.

/// A model-specific class identifier.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ClassId(pub u16);

/// A track identifier local to a single media source / tracker instance.
///
/// Local ids are **not** globally unique across cameras. Use [`TrackKey`] or
/// [`TrackUid`] when identity must span sources.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct TrackId(pub u32);

/// Local track id alias for multi-source APIs.
pub type LocalTrackId = TrackId;

/// An application-specific zone identifier.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ZoneId(pub u16);

/// A media source (camera, file, stream) identifier.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct SourceId(pub u32);

/// Composite key uniquely identifying a local track within one source.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct TrackKey {
    /// Media source that owns the local track id space.
    pub source_id: SourceId,
    /// Tracker-local id inside that source.
    pub local_track_id: LocalTrackId,
}

impl TrackKey {
    /// Creates a composite track key.
    #[must_use]
    pub const fn new(source_id: SourceId, local_track_id: LocalTrackId) -> Self {
        Self {
            source_id,
            local_track_id,
        }
    }
}

/// Globally unique track identifier across all sources in a session.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct TrackUid(pub u64);

/// A unique observation identifier within a processing context.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ObservationId(pub u64);

/// A stable subject / identity identifier linked across tracks.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct SubjectId(pub u64);

/// An opaque handle to stored evidence (frame crop, reel, sidecar blob).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct EvidenceRef(pub u64);

/// An opaque handle to a compact mask stored out-of-line.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct MaskRef(pub u64);

/// An opaque handle to an embedding vector stored out-of-line.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct EmbeddingRef(pub u64);

/// An opaque handle to a keypoint set stored out-of-line.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct KeypointSetRef(pub u64);

/// A unique event identifier inside a `VisionIndex` document.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct EventId(pub u64);

/// A unique appearance identifier (one continuous sighting of a subject).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct AppearanceId(pub u64);

/// A unique visit identifier (subject presence in a scene or zone window).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct VisitId(pub u64);

/// A unique pattern identifier produced by analysis.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct PatternId(pub u64);

/// A unique anomaly identifier produced by analysis.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct AnomalyId(pub u64);
