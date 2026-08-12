//! Typed identifiers shared across `SightLoom` crates.

/// A model-specific class identifier.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ClassId(pub u16);

/// An externally assigned or tracker-generated track identifier.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct TrackId(pub u32);

/// An application-specific zone identifier.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ZoneId(pub u16);

/// A media source (camera, file, stream) identifier.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct SourceId(pub u32);

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
