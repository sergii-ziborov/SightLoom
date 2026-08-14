//! Device preference for host inference (CPU / GPU / auto).

use serde::{Deserialize, Serialize};

/// Where the host prefers to run a model.
///
/// Step 1 only records intent. Real ONNX/CUDA selection lands behind `onnx`
/// (or a host-private runtime crate) in a later step.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DevicePreference {
    /// Let the runtime choose.
    #[default]
    Auto,
    /// Force CPU.
    Cpu,
    /// Prefer discrete / integrated GPU when available.
    Gpu,
    /// Explicit device index (runtime-defined).
    DeviceIndex(u32),
}

impl DevicePreference {
    /// Human-readable label for logs / metrics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
            Self::DeviceIndex(_) => "device_index",
        }
    }
}
