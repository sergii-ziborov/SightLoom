//! Compact observation attributes without heap strings in the base profile.

/// Small fixed set of numeric/boolean attributes attached to an observation.
///
/// Host layers may map these slots to named vocabularies. The core stays free
/// of string allocations.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ObservationAttributes {
    /// Application-defined attribute flags (bitfield).
    pub flags: u32,
    /// Optional numeric slots (for example speed, dwell proxy, quality).
    pub values: [f32; 4],
}

impl ObservationAttributes {
    /// Empty attributes.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            flags: 0,
            values: [0.0; 4],
        }
    }
}
