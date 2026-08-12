#![cfg_attr(not(feature = "std"), no_std)]
//! Rich observations for `SightLoom` video understanding.
//!
//! [`Observation`] sits above compact [`sightloom_core::Detection`] and carries
//! identity, evidence, and optional out-of-line mask/embedding handles.

#[cfg(feature = "alloc")]
extern crate alloc;

mod attributes;
mod observation;
mod oriented;

pub use attributes::ObservationAttributes;
pub use observation::Observation;
pub use oriented::OrientedRect;
