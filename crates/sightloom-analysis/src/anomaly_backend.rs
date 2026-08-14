//! Pluggable anomaly detector trait (statistical / classical / quantum hosts).
//!
//! `SightLoom` ships statistical, Isolation Forest, and RBF One-Class SVM
//! baselines. Hosts plug graph models or optional quantum backends behind this
//! trait.

use crate::anomaly::AnomalyEvent;
use crate::input::AnalysisSeries;

/// Backend-neutral anomaly detector.
///
/// Quantum ML adapters should implement **this** trait, not live inside
/// sibling products (render / intelligence hosts).
pub trait AnomalyDetector {
    /// Detector-specific error.
    type Error: core::fmt::Debug;

    /// Fits / refreshes internal baseline from history (optional no-op).
    ///
    /// # Errors
    ///
    /// Backend-defined fit failures.
    fn fit(&mut self, history: &AnalysisSeries) -> Result<(), Self::Error>;

    /// Scores live series and returns backend-neutral events.
    ///
    /// # Errors
    ///
    /// Backend-defined detection failures.
    fn detect(
        &mut self,
        live: &AnalysisSeries,
        next_id: &mut u64,
    ) -> Result<Vec<AnomalyEvent>, Self::Error>;
}

/// Statistical z-score backend implementing [`AnomalyDetector`].
#[derive(Clone, Debug, Default)]
pub struct StatisticalAnomalyDetector {
    /// Config.
    pub config: crate::stat_anomaly::StatAnomalyConfig,
    /// Frozen baseline after fit.
    pub baseline: Option<crate::stat_anomaly::BaselineStats>,
}

impl StatisticalAnomalyDetector {
    /// Creates with config.
    #[must_use]
    pub const fn new(config: crate::stat_anomaly::StatAnomalyConfig) -> Self {
        Self {
            config,
            baseline: None,
        }
    }
}

impl AnomalyDetector for StatisticalAnomalyDetector {
    type Error = &'static str;

    fn fit(&mut self, history: &AnalysisSeries) -> Result<(), Self::Error> {
        self.baseline = Some(crate::stat_anomaly::build_baseline(history, self.config));
        Ok(())
    }

    fn detect(
        &mut self,
        live: &AnalysisSeries,
        next_id: &mut u64,
    ) -> Result<Vec<AnomalyEvent>, Self::Error> {
        let baseline = self
            .baseline
            .clone()
            .unwrap_or_else(|| crate::stat_anomaly::build_baseline(live, self.config));
        Ok(crate::stat_anomaly::detect_statistical(
            live,
            &baseline,
            self.config,
            next_id,
        ))
    }
}
