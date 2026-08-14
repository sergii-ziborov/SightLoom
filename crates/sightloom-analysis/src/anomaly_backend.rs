//! Pluggable anomaly detector trait (statistical / classical / quantum hosts).
//!
//! `SightLoom` ships statistical, Isolation Forest, RBF One-Class SVM, and
//! graph/relational baselines. Hosts plug heavier graph models or optional
//! quantum backends behind this trait.

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
    /// Frozen global baseline after fit.
    pub baseline: Option<crate::stat_anomaly::BaselineStats>,
    /// Optional subject/source scoped baselines (preferred when set).
    pub scoped: Option<crate::scoped_baseline::ScopedBaselineStore>,
    /// When true, [`Self::detect`] uses scoped baselines.
    pub use_scoped: bool,
}

impl StatisticalAnomalyDetector {
    /// Creates with config.
    #[must_use]
    pub const fn new(config: crate::stat_anomaly::StatAnomalyConfig) -> Self {
        Self {
            config,
            baseline: None,
            scoped: None,
            use_scoped: false,
        }
    }

    /// Enables subject/camera scoped baselines on next fit/detect.
    pub fn enable_scoped(&mut self, enabled: bool) {
        self.use_scoped = enabled;
    }

    /// Applies a FAR calibration report to the z-threshold.
    pub fn apply_far_report(&mut self, report: &crate::far_calibrate::FarCalibrationReport) {
        self.config = crate::far_calibrate::apply_far_to_stat_config(self.config, report);
    }
}

impl AnomalyDetector for StatisticalAnomalyDetector {
    type Error = &'static str;

    fn fit(&mut self, history: &AnalysisSeries) -> Result<(), Self::Error> {
        self.baseline = Some(crate::stat_anomaly::build_baseline(history, self.config));
        if self.use_scoped {
            self.scoped = Some(crate::scoped_baseline::ScopedBaselineStore::from_series(
                history,
                self.config,
            ));
        }
        Ok(())
    }

    fn detect(
        &mut self,
        live: &AnalysisSeries,
        next_id: &mut u64,
    ) -> Result<Vec<AnomalyEvent>, Self::Error> {
        if self.use_scoped {
            let store = self.scoped.clone().unwrap_or_else(|| {
                crate::scoped_baseline::ScopedBaselineStore::from_series(live, self.config)
            });
            return Ok(crate::scoped_baseline::detect_statistical_scoped(
                live,
                &store,
                self.config,
                next_id,
            ));
        }
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
