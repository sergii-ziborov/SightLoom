//! Model and threshold provenance for reproducible memory.

/// Provenance for detections that produced a memory record.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct ModelProvenance {
    /// Free-form model name or URI.
    pub model_name: String,
    /// Model version or digest.
    pub model_version: String,
    /// Detector confidence threshold used at capture time.
    pub confidence_threshold: f32,
    /// NMS / match threshold used at capture time, if any.
    pub match_threshold: Option<f32>,
}

impl ModelProvenance {
    /// Creates provenance when thresholds are finite.
    #[must_use]
    pub fn new(
        model_name: impl Into<String>,
        model_version: impl Into<String>,
        confidence_threshold: f32,
        match_threshold: Option<f32>,
    ) -> Self {
        Self {
            model_name: model_name.into(),
            model_version: model_version.into(),
            confidence_threshold,
            match_threshold,
        }
    }
}

/// Content hash of a media source for integrity and multi-camera joins.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct SourceHash {
    /// Algorithm name (`sha256`, `blake3`, ...).
    pub algorithm: String,
    /// Lower-hex digest.
    pub digest_hex: String,
}
