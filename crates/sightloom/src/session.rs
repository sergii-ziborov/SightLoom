//! In-memory session that builds a `VisionIndex` from detections, zones, and re-id.
//!
//! Multi-source safety: each [`SourceId`] has an independent tracker. Identity maps
//! are keyed by [`TrackKey`] `(source_id, local_track_id)`. Globally unique
//! [`TrackUid`] values are assigned by [`MultiSourceTracker`].

#![allow(
    clippy::similar_names,
    clippy::struct_field_names,
    clippy::wrong_self_convention
)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use sightloom_analysis::{AnalyticsEvent, ZoneAnalytics, analytics_to_envelope};
use sightloom_core::{
    Detection, EmbeddingRef, EventId, FrameStamp, MediaTime, Point, Rect, SourceId, SubjectId,
    TrackId, TrackKey, TrackUid,
};
use sightloom_index::{
    SourceEntry, TrackSample, VisionIndex, VisionIndexPackage, VisionIndexSnapshot,
};
use sightloom_reid::{
    EmbeddingError, EmbeddingObservation, EmbeddingStore, IdentityAuditEvent, IdentityMatch,
    MatchDecision, ReferenceSample, ResolveConfig, SubjectGallery, SubjectModality,
    SubjectReference, TrackFragment, aggregate_fragment,
};
use sightloom_tracking::{
    ByteTrackConfig, MultiSourceCheckpoint, MultiSourceTracker, SourceTrackerCheckpoint, Track,
    TrackError, TrackState, TrackedDetection, TrackerSnapshot, UidMapEntry,
};

use crate::ingest::{
    IngestDecision, IngestMetrics, IngestPolicy, SourceLifecycle, SourceWatermark, evaluate_stamp,
};

/// Errors raised while materializing a `VisionIndex` session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SessionError {
    /// Tracker configuration or runtime failure.
    Track(TrackError),
    /// Zone analytics capacity or config failure.
    Analytics,
    /// Identity / embedding failure.
    Identity(EmbeddingError),
    /// Snapshot serialization failure message.
    Serialize(String),
    /// Frame rejected as late under ingest policy.
    LateFrame,
    /// Frame rejected as out-of-order under ingest policy.
    OutOfOrderFrame,
    /// Frame dropped by ingest policy.
    DroppedFrame,
}

impl From<TrackError> for SessionError {
    fn from(value: TrackError) -> Self {
        Self::Track(value)
    }
}

impl From<EmbeddingError> for SessionError {
    fn from(value: EmbeddingError) -> Self {
        Self::Identity(value)
    }
}

/// Automatic video-memory rebuild schedule during ingest.
///
/// When `every_n_frames > 0`, each accepted frame increments a counter; at the
/// threshold the session rebuilds appearances/visits (and optionally subject
/// profiles). Default is off so hosts opt in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryAutoRebuild {
    /// Rebuild after this many accepted frames (`0` = disabled).
    pub every_n_frames: u64,
    /// Also rebuild `SubjectProfile` rows when auto-rebuilding.
    pub rebuild_profiles: bool,
}

impl Default for MemoryAutoRebuild {
    fn default() -> Self {
        Self {
            every_n_frames: 0,
            rebuild_profiles: true,
        }
    }
}

/// Host-facing session: multi-source tracker + identity gallery + `VisionIndex`.
pub struct IndexSession {
    tracker: MultiSourceTracker,
    index: VisionIndex,
    gallery: SubjectGallery,
    next_event_id: u64,
    /// Stable subject assignment per composite track key.
    track_subjects: HashMap<(u32, u32), SubjectId>,
    /// Pending embedding observations keyed by composite track key.
    pending_embeddings: HashMap<(u32, u32), Vec<EmbeddingObservation>>,
    /// When true, accepted matches auto-write `subject_id` onto tracks.
    auto_assign_subjects: bool,
    /// Default modality for fragment resolution.
    default_modality: SubjectModality,
    /// Optional model identity recorded into checkpoints.
    embedding_model_id: Option<String>,
    /// Ingest policy for late / OOO frames.
    ingest_policy: IngestPolicy,
    /// Per-source watermarks.
    watermarks: HashMap<u32, SourceWatermark>,
    /// Ingest metrics counters.
    metrics: IngestMetrics,
    /// Next pattern id for miners.
    next_pattern_id: u64,
    /// Next anomaly id for detectors.
    next_anomaly_id: u64,
    /// Next appearance id for memory materialization.
    next_appearance_id: u64,
    /// Next visit id for memory materialization.
    next_visit_id: u64,
    /// Next redaction provenance interval id.
    next_redaction_id: u64,
    /// Statistical anomaly config.
    anomaly_config: sightloom_analysis::StatAnomalyConfig,
    /// Optional frozen baseline for anomaly detection (history).
    anomaly_baseline: Option<sightloom_analysis::BaselineStats>,
    /// Config for auto appearances / visits from tracks.
    memory_build: sightloom_index::MemoryBuildConfig,
    /// Auto-rebuild schedule (opt-in).
    memory_auto: MemoryAutoRebuild,
    /// Accepted frames since last auto memory rebuild.
    frames_since_memory_rebuild: u64,
    /// Counts from the last auto rebuild: `(appearances, visits, subjects)`.
    last_auto_memory_rebuild: Option<(usize, usize, usize)>,
    /// Latest embedding handle per track key for unlabeled track search.
    track_embeddings: HashMap<(u32, u32), EmbeddingRef>,
}

impl IndexSession {
    /// Creates a session with a validated tracker config and empty index/gallery.
    ///
    /// # Errors
    ///
    /// Returns tracker configuration errors.
    pub fn new(
        name: impl Into<String>,
        track_config: ByteTrackConfig,
    ) -> Result<Self, SessionError> {
        Ok(Self {
            tracker: MultiSourceTracker::new(track_config)?,
            index: VisionIndex::new(name),
            gallery: SubjectGallery::new(),
            next_event_id: 1,
            track_subjects: HashMap::new(),
            pending_embeddings: HashMap::new(),
            auto_assign_subjects: true,
            default_modality: SubjectModality::PersonAppearance,
            embedding_model_id: None,
            ingest_policy: IngestPolicy::default(),
            watermarks: HashMap::new(),
            metrics: IngestMetrics::default(),
            next_pattern_id: 1,
            next_anomaly_id: 1,
            next_appearance_id: 1,
            next_visit_id: 1,
            next_redaction_id: 1,
            anomaly_config: sightloom_analysis::StatAnomalyConfig::default(),
            anomaly_baseline: None,
            memory_build: sightloom_index::MemoryBuildConfig::default(),
            memory_auto: MemoryAutoRebuild::default(),
            frames_since_memory_rebuild: 0,
            last_auto_memory_rebuild: None,
            track_embeddings: HashMap::new(),
        })
    }

    /// Overrides ingest policy (late / OOO / queue hints).
    pub fn set_ingest_policy(&mut self, policy: IngestPolicy) {
        self.ingest_policy = policy;
    }

    /// Configures appearance/visit materialization gaps.
    pub fn set_memory_build_config(&mut self, config: sightloom_index::MemoryBuildConfig) {
        self.memory_build = config;
    }

    /// Enables or configures automatic memory rebuild during ingest.
    ///
    /// Pass `every_n_frames: 0` to disable. When enabled, every accepted frame
    /// counts toward the threshold (multi-source frames share one counter).
    pub fn set_memory_auto_rebuild(&mut self, config: MemoryAutoRebuild) {
        self.memory_auto = config;
        if config.every_n_frames == 0 {
            self.frames_since_memory_rebuild = 0;
        }
    }

    /// Current auto-rebuild schedule.
    #[must_use]
    pub const fn memory_auto_rebuild(&self) -> MemoryAutoRebuild {
        self.memory_auto
    }

    /// Accepted frames counted since the last auto memory rebuild.
    #[must_use]
    pub const fn frames_since_memory_rebuild(&self) -> u64 {
        self.frames_since_memory_rebuild
    }

    /// Counts from the last auto rebuild, if any: `(appearances, visits, subjects)`.
    #[must_use]
    pub const fn last_auto_memory_rebuild(&self) -> Option<(usize, usize, usize)> {
        self.last_auto_memory_rebuild
    }

    /// Runs memory rebuild if the auto schedule threshold is met.
    ///
    /// Returns the rebuild counts when a rebuild ran.
    pub fn maybe_auto_rebuild_memory(&mut self) -> Option<(usize, usize, usize)> {
        let n = self.memory_auto.every_n_frames;
        if n == 0 || self.frames_since_memory_rebuild < n {
            return None;
        }
        let counts = if self.memory_auto.rebuild_profiles {
            self.rebuild_memory_from_tracks()
        } else {
            let (a, v) = self.rebuild_appearances_and_visits();
            (a, v, self.index.subjects.len())
        };
        self.frames_since_memory_rebuild = 0;
        self.last_auto_memory_rebuild = Some(counts);
        Some(counts)
    }

    fn note_accepted_frame_for_memory(&mut self) {
        if self.memory_auto.every_n_frames == 0 {
            return;
        }
        self.frames_since_memory_rebuild = self.frames_since_memory_rebuild.saturating_add(1);
        let _ = self.maybe_auto_rebuild_memory();
    }

    /// Rebuilds `appearances` and `visits` from effective track samples.
    ///
    /// Idempotent: replaces existing appearance/visit tables. Returns
    /// `(appearance_count, visit_count)`.
    pub fn rebuild_appearances_and_visits(&mut self) -> (usize, usize) {
        // Keep ids monotonic across rebuilds.
        let max_app = self
            .index
            .appearances
            .iter()
            .map(|a| a.appearance_id.0)
            .max()
            .unwrap_or(0);
        let max_vis = self
            .index
            .visits
            .iter()
            .map(|v| v.visit_id.0)
            .max()
            .unwrap_or(0);
        self.next_appearance_id = self
            .next_appearance_id
            .max(max_app.saturating_add(1))
            .max(1);
        self.next_visit_id = self.next_visit_id.max(max_vis.saturating_add(1)).max(1);
        sightloom_index::rebuild_memory_entities(
            &mut self.index,
            self.memory_build,
            &mut self.next_appearance_id,
            &mut self.next_visit_id,
        )
    }

    /// Rebuilds `index.subjects` from appearances (or track samples).
    ///
    /// Preserves host-supplied labels and embeddings. When a gallery subject
    /// has a reference embedding and the profile has none, copies that handle.
    /// Returns the number of subject profiles.
    pub fn rebuild_subject_profiles(&mut self) -> usize {
        let n = sightloom_index::rebuild_subject_profiles(&mut self.index);
        self.enrich_subject_embeddings_from_gallery();
        n
    }

    /// Rebuilds appearances, visits, then subject profiles in one call.
    ///
    /// Returns `(appearances, visits, subjects)`.
    pub fn rebuild_memory_from_tracks(&mut self) -> (usize, usize, usize) {
        let (a, v) = self.rebuild_appearances_and_visits();
        let s = self.rebuild_subject_profiles();
        (a, v, s)
    }

    fn enrich_subject_embeddings_from_gallery(&mut self) {
        for profile in &mut self.index.subjects {
            if profile.embedding.is_some() {
                continue;
            }
            let Some(subj) = self
                .gallery
                .subjects()
                .iter()
                .find(|s| s.subject_id == profile.subject_id)
            else {
                continue;
            };
            if let Some(sample) = subj.samples.iter().find(|s| s.embedding.is_some()) {
                profile.embedding = sample.embedding;
            }
        }
    }

    /// Sets a host display label on an existing (or empty) subject profile.
    pub fn set_subject_label(&mut self, subject_id: SubjectId, label: impl Into<String>) {
        let label = label.into();
        if let Some(profile) = self
            .index
            .subjects
            .iter_mut()
            .find(|p| p.subject_id == subject_id)
        {
            profile.label = Some(label);
            return;
        }
        self.index.subjects.push(sightloom_index::SubjectProfile {
            subject_id,
            label: Some(label),
            appearance_count: 0,
            source_count: 0,
            total_duration_ns: 0,
            first_seen: None,
            last_seen: None,
            embedding: None,
        });
    }

    fn bump_redaction_id(&mut self) {
        let max_id = self
            .index
            .redaction_intervals
            .iter()
            .map(|r| r.interval_id.0)
            .max()
            .unwrap_or(0);
        self.next_redaction_id = self.next_redaction_id.max(max_id.saturating_add(1)).max(1);
    }

    /// Plans blur-subject redaction intervals from appearances of `subject_id`.
    ///
    /// Replaces any existing redaction table. Returns the number of intervals.
    pub fn plan_redaction_subject(&mut self, subject_id: SubjectId, tag: u32) -> usize {
        self.bump_redaction_id();
        if self.index.appearances.is_empty() {
            let _ = self.rebuild_appearances_and_visits();
        }
        let rows = sightloom_index::build_redaction_from_appearances(
            &self.index.appearances,
            Some(subject_id),
            None,
            sightloom_index::RedactionIntent::BlurSubject,
            tag,
            &mut self.next_redaction_id,
        );
        let n = rows.len();
        sightloom_index::set_redaction_intervals(&mut self.index, rows);
        n
    }

    /// Plans blur-everyone-except `keep_subject` from appearances of others.
    ///
    /// Replaces any existing redaction table. Returns the number of intervals.
    pub fn plan_redaction_blur_others(&mut self, keep_subject: SubjectId, tag: u32) -> usize {
        self.bump_redaction_id();
        if self.index.appearances.is_empty() {
            let _ = self.rebuild_appearances_and_visits();
        }
        let rows = sightloom_index::build_redaction_from_appearances(
            &self.index.appearances,
            None,
            Some(keep_subject),
            sightloom_index::RedactionIntent::BlurOthers,
            tag,
            &mut self.next_redaction_id,
        );
        let n = rows.len();
        sightloom_index::set_redaction_intervals(&mut self.index, rows);
        n
    }

    /// Plans uncertain-hold provenance from re-id uncertain intervals.
    ///
    /// Replaces any existing redaction table. Returns the number of intervals.
    pub fn plan_redaction_uncertain(&mut self, tag: u32) -> usize {
        self.bump_redaction_id();
        let specs: Vec<sightloom_index::RedactionSpec> = self
            .uncertain_intervals()
            .into_iter()
            .map(|i| sightloom_index::RedactionSpec {
                subject_id: i.subject_id,
                source_id: i.source_id,
                track_id: Some(i.track_id),
                start: i.start,
                end: i.end,
                intent: sightloom_index::RedactionIntent::UncertainHold,
                evidence: None,
                mask_ref: 0,
                peak_confidence: i.peak_score.unwrap_or(0.0),
                tag,
            })
            .collect();
        let rows = sightloom_index::build_redaction_from_specs(&specs, &mut self.next_redaction_id);
        let n = rows.len();
        sightloom_index::set_redaction_intervals(&mut self.index, rows);
        n
    }

    /// Clears the redaction provenance table.
    pub fn clear_redaction_intervals(&mut self) {
        self.index.redaction_intervals.clear();
    }

    /// JSON export of redaction provenance intervals (demo step 10 / Intelligence handoff).
    ///
    /// # Errors
    ///
    /// Returns serialization failures.
    pub fn export_redaction_intervals_json(&self) -> Result<String, SessionError> {
        let rows: Vec<crate::analysis_bridge::RedactionIntervalExportDto> = self
            .index
            .redaction_intervals
            .iter()
            .map(|r| crate::analysis_bridge::RedactionIntervalExportDto {
                interval_id: r.interval_id.0,
                subject_id: r.subject_id.map(|id| id.0),
                source_id: r.source_id.0,
                track_id: r.track_id.map(|id| id.0),
                start_ticks: r.start.ticks(),
                start_timescale: r.start.timescale(),
                end_ticks: r.end.ticks(),
                end_timescale: r.end.timescale(),
                intent: r.intent.as_str().into(),
                evidence: r.evidence.map(|e| e.0),
                mask_ref: r.mask_ref,
                peak_confidence: r.peak_confidence,
                appearance_id: r.appearance_id.map(|id| id.0),
                tag: r.tag,
            })
            .collect();
        serde_json::to_string_pretty(&rows)
            .map_err(|error| SessionError::Serialize(error.to_string()))
    }

    /// Returns ingest metrics snapshot.
    #[must_use]
    pub const fn ingest_metrics(&self) -> IngestMetrics {
        self.metrics
    }

    /// Applies a source lifecycle event (reset / reconnect).
    pub fn apply_source_lifecycle(&mut self, event: &SourceLifecycle) {
        match *event {
            SourceLifecycle::Added { source_id } | SourceLifecycle::Reconnected { source_id } => {
                self.watermarks
                    .entry(source_id.0)
                    .or_insert_with(|| SourceWatermark::new(source_id));
            }
            SourceLifecycle::Removed {
                source_id,
                reset_tracker,
            } => {
                if reset_tracker {
                    self.watermarks
                        .insert(source_id.0, SourceWatermark::new(source_id));
                    self.metrics.source_resets = self.metrics.source_resets.saturating_add(1);
                }
            }
            SourceLifecycle::Reset { source_id } => {
                self.watermarks
                    .insert(source_id.0, SourceWatermark::new(source_id));
                self.metrics.source_resets = self.metrics.source_resets.saturating_add(1);
            }
        }
    }

    /// Uncertain identity intervals from the gallery audit trail.
    #[must_use]
    pub fn uncertain_intervals(&self) -> Vec<sightloom_reid::IdentityInterval> {
        self.gallery.uncertain_intervals()
    }

    /// Seeds a subject onto a host-provided box (click demo): ingest one
    /// detection, assign a new or existing subject to the resulting track key.
    ///
    /// # Errors
    ///
    /// Propagates tracker / ingest errors.
    pub fn seed_subject_from_box(
        &mut self,
        stamp: FrameStamp,
        bbox: Rect,
        score: f32,
        class_id: Option<sightloom_core::ClassId>,
        subject_id: Option<SubjectId>,
    ) -> Result<(TrackedDetection, SubjectId), SessionError> {
        let detection = Detection::new(bbox, score, class_id, None)
            .map_err(|_| SessionError::Track(TrackError::NonFinite))?;
        let tracked = self.ingest_detections(stamp, &[detection])?;
        let item = tracked
            .into_iter()
            .next()
            .ok_or(SessionError::Track(TrackError::NonFinite))?;
        let subject = subject_id.unwrap_or_else(|| self.register_subject(self.default_modality));
        self.assign_subject(item.track_key, subject);
        Ok((item, subject))
    }

    /// Demo helper: seed click box and return a compact [`SeedResult`].
    ///
    /// # Errors
    ///
    /// Propagates tracker / ingest errors.
    pub fn seed_click(
        &mut self,
        stamp: FrameStamp,
        bbox: Rect,
        score: f32,
        subject_id: Option<SubjectId>,
    ) -> Result<crate::analysis_bridge::SeedResult, SessionError> {
        let (item, subject) = self.seed_subject_from_box(
            stamp,
            bbox,
            score,
            Some(sightloom_core::ClassId(0)),
            subject_id,
        )?;
        Ok(crate::analysis_bridge::SeedResult {
            source_id: item.track_key.source_id,
            track_id: item.track_key.local_track_id,
            track_uid: item.track_uid,
            subject_id: subject,
        })
    }

    /// Manually assigns a subject to a track key (host-accepted special box / tid).
    pub fn assign_subject(&mut self, key: TrackKey, subject_id: SubjectId) {
        self.track_subjects
            .insert((key.source_id.0, key.local_track_id.0), subject_id);
        self.patch_latest_track_subject(key, subject_id);
    }

    /// Accepts a host-known local track id + optional subject (special frame/tid).
    ///
    /// Does not re-run the tracker; only updates identity maps and latest sample.
    pub fn accept_host_track(
        &mut self,
        source_id: SourceId,
        local_track_id: TrackId,
        subject_id: Option<SubjectId>,
    ) -> TrackKey {
        let key = TrackKey::new(source_id, local_track_id);
        if let Some(subject_id) = subject_id {
            self.assign_subject(key, subject_id);
        }
        key
    }

    /// Exports effective track/mask spans for host `MaskTimeline` construction.
    #[must_use]
    pub fn export_track_spans(&self) -> Vec<TrackSpanExport> {
        let mut spans = Vec::new();
        for sample in self.index.tracks.effective_samples() {
            spans.push(TrackSpanExport {
                sample_id: sample.sample_id,
                source_id: sample.source_id,
                frame_index: sample.frame_index,
                pts: sample.pts,
                track_key: sample.track_key(),
                track_uid: sample.track_uid,
                subject_id: sample.subject_id,
                left: sample.left,
                top: sample.top,
                right: sample.right,
                bottom: sample.bottom,
                confidence: sample.confidence,
                mask_ref: sample.mask_ref,
                revision: sample.revision,
            });
        }
        spans
    }

    /// JSON export of effective track spans (host-friendly, no pixels).
    ///
    /// # Errors
    ///
    /// Returns serialization failures.
    pub fn export_track_spans_json(&self) -> Result<String, SessionError> {
        let rows: Vec<crate::analysis_bridge::DemoSpanDto> = self
            .export_track_spans()
            .into_iter()
            .map(|s| crate::analysis_bridge::DemoSpanDto {
                sample_id: s.sample_id,
                source_id: s.source_id.0,
                frame_index: s.frame_index,
                pts_ticks: s.pts.ticks(),
                pts_timescale: s.pts.timescale(),
                track_id: s.track_key.local_track_id.0,
                track_uid: s.track_uid.map(|u| u.0),
                subject_id: s.subject_id.map(|id| id.0),
                left: s.left,
                top: s.top,
                right: s.right,
                bottom: s.bottom,
                confidence: s.confidence,
                mask_ref: s.mask_ref,
                revision: s.revision,
            })
            .collect();
        serde_json::to_string_pretty(&rows)
            .map_err(|error| SessionError::Serialize(error.to_string()))
    }

    /// JSON export of uncertain identity intervals for demo step 5.
    ///
    /// # Errors
    ///
    /// Returns serialization failures.
    pub fn export_uncertain_intervals_json(&self) -> Result<String, SessionError> {
        let rows: Vec<crate::analysis_bridge::UncertainIntervalDto> = self
            .uncertain_intervals()
            .into_iter()
            .map(|i| crate::analysis_bridge::UncertainIntervalDto {
                source_id: i.source_id.0,
                track_id: i.track_id.0,
                subject_id: i.subject_id.map(|id| id.0),
                start_ticks: i.start.ticks(),
                start_timescale: i.start.timescale(),
                end_ticks: i.end.ticks(),
                end_timescale: i.end.timescale(),
                peak_score: i.peak_score,
            })
            .collect();
        serde_json::to_string_pretty(&rows)
            .map_err(|error| SessionError::Serialize(error.to_string()))
    }

    /// Sets statistical anomaly detection thresholds.
    pub fn set_anomaly_config(&mut self, config: sightloom_analysis::StatAnomalyConfig) {
        self.anomaly_config = config;
    }

    /// Freezes the current index as the anomaly baseline (history window).
    pub fn freeze_anomaly_baseline(&mut self) {
        self.anomaly_baseline = Some(crate::analysis_bridge::baseline_from_index(
            &self.index,
            self.anomaly_config,
        ));
    }

    /// Mines patterns from the live index and appends them to `index.patterns`.
    ///
    /// Returns the number of new patterns.
    pub fn mine_and_store_patterns(&mut self) -> usize {
        let mined = crate::analysis_bridge::mine_patterns_from_index(
            &self.index,
            &mut self.next_pattern_id,
        );
        let n = mined.len();
        self.index.patterns.extend(mined);
        n
    }

    /// Runs statistical anomaly detection and appends to `index.anomalies`.
    ///
    /// Uses a frozen baseline when present; otherwise builds baseline from the
    /// same live series (useful only with enough history).
    ///
    /// Returns the number of new anomalies.
    pub fn detect_and_store_anomalies(&mut self) -> usize {
        let baseline = self.anomaly_baseline.clone().unwrap_or_else(|| {
            crate::analysis_bridge::baseline_from_index(&self.index, self.anomaly_config)
        });
        let found = crate::analysis_bridge::detect_anomalies_from_index(
            &self.index,
            &baseline,
            self.anomaly_config,
            &mut self.next_anomaly_id,
        );
        let n = found.len();
        self.index.anomalies.extend(found);
        n
    }

    /// Builds an analysis series view of the current index (read-only helper).
    #[must_use]
    pub fn analysis_series(&self) -> sightloom_analysis::AnalysisSeries {
        crate::analysis_bridge::analysis_series_from_index(&self.index)
    }

    /// Runs a subject query against the live index.
    #[must_use]
    pub fn query_subjects(
        &self,
        query: &sightloom_index::SubjectQuery,
    ) -> Vec<sightloom_index::SubjectHit> {
        sightloom_index::execute_subject_query(&self.index, query)
    }

    /// Builds a coalesced evidence reel for a subject (handles only, no pixels).
    #[must_use]
    pub fn build_subject_reel(
        &self,
        subject_id: SubjectId,
        max_gap_ns: i64,
    ) -> sightloom_index::EvidenceReel {
        sightloom_index::build_subject_reel(&self.index, subject_id, max_gap_ns)
    }

    /// Builds one sample-per-segment reel for a subject.
    #[must_use]
    pub fn build_subject_reel_samples(
        &self,
        subject_id: SubjectId,
    ) -> sightloom_index::EvidenceReel {
        sightloom_index::EvidenceReelBuilder::new().from_subject_samples(&self.index, subject_id, 0)
    }

    /// Builds a coalesced reel and **stores** it on the index (package-persisted).
    pub fn store_subject_reel(
        &mut self,
        subject_id: SubjectId,
        max_gap_ns: i64,
        tag: u32,
    ) -> sightloom_index::EvidenceReel {
        let mut reel = self.build_subject_reel(subject_id, max_gap_ns);
        // Allocate a stable reel id relative to existing stored reels.
        let next = self
            .index
            .evidence_reels
            .iter()
            .map(|r| r.reel_id.0)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
            .max(1);
        reel.reel_id = sightloom_index::ReelId(next);
        reel.tag = tag;
        self.index.evidence_reels.push(reel.clone());
        reel
    }

    /// Stored evidence reels (package / snapshot).
    #[must_use]
    pub fn evidence_reels(&self) -> &[sightloom_index::EvidenceReel] {
        &self.index.evidence_reels
    }

    /// Clears stored evidence reels.
    pub fn clear_evidence_reels(&mut self) {
        self.index.evidence_reels.clear();
    }

    /// Appends a revision that supersedes the latest effective sample for `key`.
    ///
    /// Returns the new sample id when a prior sample existed.
    pub fn revise_latest_track_sample(
        &mut self,
        key: TrackKey,
        bbox: Rect,
        confidence: f32,
        mask_ref: Option<u64>,
    ) -> Option<u64> {
        let prior = self
            .index
            .tracks
            .effective_samples()
            .into_iter()
            .rev()
            .find(|s| s.track_key() == key)?;
        let subject_id = self
            .track_subjects
            .get(&(key.source_id.0, key.local_track_id.0))
            .copied()
            .or(prior.subject_id);
        let mut sample = prior;
        sample.left = bbox.left();
        sample.top = bbox.top();
        sample.right = bbox.right();
        sample.bottom = bbox.bottom();
        sample.confidence = confidence;
        sample.subject_id = subject_id;
        if let Some(mask) = mask_ref {
            sample.mask_ref = mask;
        }
        self.index.tracks.push_revision(sample, prior.sample_id);
        self.index.tracks.samples().last().map(|s| s.sample_id)
    }

    /// Ranks subjects by track-sample frequency (most frequent first).
    #[must_use]
    pub fn rank_subjects(&self) -> Vec<sightloom_index::SubjectRank> {
        sightloom_index::rank_subjects_by_frequency(&self.index)
    }

    /// Most frequent subject + coalesced evidence reel (handles only).
    #[must_use]
    pub fn most_frequent_subject_reel(
        &self,
        max_gap_ns: i64,
    ) -> Option<(sightloom_index::SubjectRank, sightloom_index::EvidenceReel)> {
        let rank = sightloom_index::most_frequent_subject(&self.index)?;
        let reel = self.build_subject_reel(rank.subject_id, max_gap_ns);
        Some((rank, reel))
    }

    /// Registers a subject and attaches one or more reference photo embeddings.
    ///
    /// Hosts compute embeddings externally (detector / face model). `SightLoom`
    /// only stores vectors and ranks them.
    ///
    /// # Errors
    ///
    /// Propagates embedding validation errors.
    pub fn enroll_subject_photos(
        &mut self,
        modality: SubjectModality,
        photos: &[Vec<f32>],
    ) -> Result<SubjectId, SessionError> {
        let subject = self.register_subject(modality);
        for photo in photos {
            self.gallery
                .add_reference_photo(subject, photo.clone(), Some(1.0), None, None)?;
        }
        Ok(subject)
    }

    /// Adds a reference photo embedding to an existing subject.
    ///
    /// # Errors
    ///
    /// Propagates gallery errors.
    pub fn add_subject_photo(
        &mut self,
        subject_id: SubjectId,
        vector: impl Into<Vec<f32>>,
    ) -> Result<EmbeddingRef, SessionError> {
        Ok(self
            .gallery
            .add_reference_photo(subject_id, vector, Some(1.0), None, None)?)
    }

    /// Searches enrolled subjects by a query photo embedding (top-k).
    ///
    /// # Errors
    ///
    /// Propagates embedding / resolve errors.
    pub fn search_by_photo(
        &mut self,
        vector: impl Into<Vec<f32>>,
        modality: SubjectModality,
        top_k: usize,
    ) -> Result<Vec<sightloom_reid::PhotoSearchHit>, SessionError> {
        let handle = self.gallery.embeddings.insert(vector)?;
        let query = sightloom_reid::PhotoQuery {
            embedding: handle,
            quality: 1.0,
            modality,
            class_id: None,
            source_id: SourceId(0),
            at: MediaTime::default(),
        };
        Ok(self.gallery.search_by_photo(&query, top_k)?)
    }

    /// Search by photo and attach an evidence reel for each Accept hit.
    ///
    /// # Errors
    ///
    /// Propagates search errors.
    pub fn search_photo_with_reels(
        &mut self,
        vector: impl Into<Vec<f32>>,
        modality: SubjectModality,
        top_k: usize,
        max_gap_ns: i64,
    ) -> Result<Vec<PhotoSearchResult>, SessionError> {
        let hits = self.search_by_photo(vector, modality, top_k)?;
        let mut out = Vec::with_capacity(hits.len());
        for hit in hits {
            let reel = if hit.decision == MatchDecision::Accept
                || hit.decision == MatchDecision::Uncertain
            {
                Some(self.build_subject_reel(hit.subject_id, max_gap_ns))
            } else {
                None
            };
            out.push(PhotoSearchResult { hit, reel });
        }
        Ok(out)
    }

    /// Registers a media source on the index header.
    pub fn add_source(&mut self, entry: SourceEntry) {
        self.index.add_source(entry);
    }

    /// Enables or disables automatic subject assignment on Accept matches.
    pub fn set_auto_assign_subjects(&mut self, enabled: bool) {
        self.auto_assign_subjects = enabled;
    }

    /// Sets the default re-id modality used when callers omit one.
    pub fn set_default_modality(&mut self, modality: SubjectModality) {
        self.default_modality = modality;
    }

    /// Records an embedding model identity/version for session checkpoints.
    pub fn set_embedding_model_id(&mut self, model_id: impl Into<String>) {
        self.embedding_model_id = Some(model_id.into());
    }

    /// Sets identity resolve thresholds.
    ///
    /// # Errors
    ///
    /// Propagates gallery config validation errors.
    pub fn set_resolve_config(&mut self, config: ResolveConfig) -> Result<(), SessionError> {
        self.gallery.set_resolve_config(config)?;
        Ok(())
    }

    /// Returns a shared reference to the live index.
    #[must_use]
    pub fn index(&self) -> &VisionIndex {
        &self.index
    }

    /// Mutable access for advanced entity writers (appearances, subjects, …).
    pub fn index_mut(&mut self) -> &mut VisionIndex {
        &mut self.index
    }

    /// Returns the identity gallery.
    #[must_use]
    pub fn gallery(&self) -> &SubjectGallery {
        &self.gallery
    }

    /// Mutable access to the identity gallery.
    pub fn gallery_mut(&mut self) -> &mut SubjectGallery {
        &mut self.gallery
    }

    /// Multi-source tracker pool.
    #[must_use]
    pub fn tracker(&self) -> &MultiSourceTracker {
        &self.tracker
    }

    /// Registers a new subject identity in the gallery.
    pub fn register_subject(&mut self, modality: SubjectModality) -> SubjectId {
        self.gallery.register_subject(modality)
    }

    /// Adds a positive/negative/unlabeled reference sample to a subject.
    ///
    /// # Errors
    ///
    /// Propagates gallery errors when the subject is unknown.
    pub fn add_subject_reference(
        &mut self,
        subject_id: SubjectId,
        sample: ReferenceSample,
    ) -> Result<(), SessionError> {
        self.gallery.add_reference(subject_id, sample)?;
        Ok(())
    }

    /// Inserts an embedding vector and records it for a composite track key.
    ///
    /// # Errors
    ///
    /// Propagates embedding store validation errors.
    pub fn note_track_embedding(
        &mut self,
        key: TrackKey,
        vector: impl Into<Vec<f32>>,
        at: MediaTime,
    ) -> Result<EmbeddingRef, SessionError> {
        let handle = self.gallery.embeddings.insert(vector)?;
        let map_key = (key.source_id.0, key.local_track_id.0);
        self.pending_embeddings
            .entry(map_key)
            .or_default()
            .push(EmbeddingObservation {
                embedding: handle,
                at,
            });
        // Latest embedding is searchable even before identity resolve.
        self.track_embeddings.insert(map_key, handle);
        Ok(handle)
    }

    /// Convenience: note embedding for `(source_id, track_id)`.
    ///
    /// # Errors
    ///
    /// Propagates embedding store validation errors.
    pub fn note_track_embedding_for(
        &mut self,
        source_id: SourceId,
        track_id: TrackId,
        vector: impl Into<Vec<f32>>,
        at: MediaTime,
    ) -> Result<EmbeddingRef, SessionError> {
        self.note_track_embedding(TrackKey::new(source_id, track_id), vector, at)
    }

    /// Returns the subject currently assigned to a composite track key, if any.
    #[must_use]
    pub fn subject_for_track_key(&self, key: TrackKey) -> Option<SubjectId> {
        self.track_subjects
            .get(&(key.source_id.0, key.local_track_id.0))
            .copied()
    }

    /// Returns the subject for `(source_id, local track_id)`.
    #[must_use]
    pub fn subject_for_track(&self, source_id: SourceId, track_id: TrackId) -> Option<SubjectId> {
        self.subject_for_track_key(TrackKey::new(source_id, track_id))
    }

    /// Looks up the global [`TrackUid`] for a composite key.
    #[must_use]
    pub fn track_uid(&self, key: TrackKey) -> Option<TrackUid> {
        self.tracker.uid_of(key)
    }

    /// Ingests one frame of detections for the stamp's source only.
    ///
    /// Motion state is isolated per [`SourceId`]. Local track ids may collide
    /// across sources; [`TrackUid`] values and sample `track_uid` fields do not.
    ///
    /// # Errors
    ///
    /// Propagates tracker errors.
    pub fn ingest_detections(
        &mut self,
        stamp: FrameStamp,
        detections: &[Detection],
    ) -> Result<Vec<TrackedDetection>, SessionError> {
        {
            let watermark = self
                .watermarks
                .entry(stamp.source_id.0)
                .or_insert_with(|| SourceWatermark::new(stamp.source_id));
            let decision = evaluate_stamp(&self.ingest_policy, watermark, stamp);
            self.metrics.record(decision);
            match decision {
                IngestDecision::Accept => {}
                IngestDecision::Drop => return Err(SessionError::DroppedFrame),
                IngestDecision::RejectLate => return Err(SessionError::LateFrame),
                IngestDecision::RejectOutOfOrder => return Err(SessionError::OutOfOrderFrame),
            }
        }

        let tracked = self.tracker.update(stamp.source_id, detections)?;
        for item in &tracked {
            let bbox = item.detection.bbox();
            let subject_id = self
                .track_subjects
                .get(&(item.track_key.source_id.0, item.track_key.local_track_id.0))
                .copied();
            self.index.push_track(TrackSample {
                sample_id: 0,
                supersedes: None,
                revision: 0,
                idempotency_key: 0,
                source_id: stamp.source_id,
                frame_index: stamp.frame_index,
                pts: stamp.pts,
                track_id: item.track_key.local_track_id,
                track_uid: Some(item.track_uid),
                subject_id,
                class_id: item.detection.class_id(),
                left: bbox.left(),
                top: bbox.top(),
                right: bbox.right(),
                bottom: bbox.bottom(),
                confidence: item.detection.score(),
                mask_ref: 0,
            });
        }
        self.watermarks
            .entry(stamp.source_id.0)
            .or_insert_with(|| SourceWatermark::new(stamp.source_id))
            .advance(stamp);
        self.note_accepted_frame_for_memory();
        Ok(tracked)
    }

    /// Ingests multiple frames in order (strict: first error aborts the batch).
    ///
    /// # Errors
    ///
    /// Propagates the first ingest/tracker error.
    pub fn ingest_detection_batch(
        &mut self,
        frames: &[(FrameStamp, Vec<Detection>)],
    ) -> Result<Vec<Vec<TrackedDetection>>, SessionError> {
        let mut out = Vec::with_capacity(frames.len());
        for (stamp, detections) in frames {
            out.push(self.ingest_detections(*stamp, detections)?);
        }
        Ok(out)
    }

    /// Soft multi-frame ingest: late / OOO / drop policy rejections are skipped
    /// (already counted in metrics); hard tracker errors still abort.
    ///
    /// Returns one entry per **accepted** frame only.
    ///
    /// # Errors
    ///
    /// Returns tracker failures (non-policy).
    pub fn ingest_detection_batch_soft(
        &mut self,
        frames: &[(FrameStamp, Vec<Detection>)],
    ) -> Result<Vec<Vec<TrackedDetection>>, SessionError> {
        let mut out = Vec::new();
        for (stamp, detections) in frames {
            match self.ingest_detections(*stamp, detections) {
                Ok(tracked) => out.push(tracked),
                Err(
                    SessionError::LateFrame
                    | SessionError::OutOfOrderFrame
                    | SessionError::DroppedFrame,
                ) => {}
                Err(other) => return Err(other),
            }
        }
        Ok(out)
    }

    /// Pops and ingests up to `max_frames` from a host [`crate::ingest::FrameQueue`].
    ///
    /// Soft on policy rejects; updates `metrics.queue_hwm` from the queue HWM.
    ///
    /// # Errors
    ///
    /// Returns hard tracker errors.
    pub fn drain_frame_queue(
        &mut self,
        queue: &mut crate::ingest::FrameQueue,
        max_frames: Option<usize>,
    ) -> Result<Vec<Vec<TrackedDetection>>, SessionError> {
        let hwm = u64::try_from(queue.high_water_mark()).unwrap_or(u64::MAX);
        self.metrics.queue_hwm = self.metrics.queue_hwm.max(hwm);
        let frames = queue.drain(max_frames);
        let mut out = Vec::with_capacity(frames.len());
        for item in frames {
            match self.ingest_detections(item.stamp, &item.detections) {
                Ok(tracked) => out.push(tracked),
                Err(
                    SessionError::LateFrame
                    | SessionError::OutOfOrderFrame
                    | SessionError::DroppedFrame,
                ) => {}
                Err(other) => return Err(other),
            }
        }
        Ok(out)
    }

    /// Aggregates pending embeddings for a track key, resolves identity, and
    /// patches the latest track sample when a subject is assigned.
    ///
    /// # Errors
    ///
    /// Returns identity errors when no pending embeddings exist or pooling fails.
    pub fn resolve_track_identity(
        &mut self,
        key: TrackKey,
        modality: Option<SubjectModality>,
        at: MediaTime,
    ) -> Result<(TrackFragment, Vec<IdentityMatch>), SessionError> {
        let map_key = (key.source_id.0, key.local_track_id.0);
        let observations = self
            .pending_embeddings
            .remove(&map_key)
            .ok_or(EmbeddingError::InvalidVector)?;
        if observations.is_empty() {
            return Err(EmbeddingError::InvalidVector.into());
        }
        let modality = modality.unwrap_or(self.default_modality);
        let known_subject = self.track_subjects.get(&map_key).copied();
        let fragment = aggregate_fragment(
            &mut self.gallery.embeddings,
            key.local_track_id,
            key.source_id,
            modality,
            &observations,
            known_subject,
        )?;
        if let Some(emb) = fragment.embedding {
            self.track_embeddings.insert(map_key, emb);
        }
        let (fragment, matches) =
            self.gallery
                .resolve_and_audit(fragment, self.auto_assign_subjects, at);
        if let Some(subject_id) = fragment.subject_id {
            self.track_subjects.insert(map_key, subject_id);
            self.patch_latest_track_subject(key, subject_id);
        }
        Ok((fragment, matches))
    }

    /// Searches track embeddings (not only enrolled subjects) by cosine similarity.
    ///
    /// Hosts call [`Self::note_track_embedding`] while ingesting video so tracks
    /// become searchable before / without gallery enrollment.
    ///
    /// # Errors
    ///
    /// Propagates embedding store errors.
    pub fn search_tracks_by_embedding(
        &mut self,
        vector: impl Into<Vec<f32>>,
        top_k: usize,
    ) -> Result<Vec<TrackEmbeddingHit>, SessionError> {
        let query = self.gallery.embeddings.insert(vector)?;
        let q = self.gallery.embeddings.get(query)?;
        let mut hits = Vec::new();
        for (&(source, local), &handle) in &self.track_embeddings {
            let Ok(vector) = self.gallery.embeddings.get(handle) else {
                continue;
            };
            let Some(score) = sightloom_reid::cosine_similarity(q, vector) else {
                continue;
            };
            let key = TrackKey::new(SourceId(source), TrackId(local));
            hits.push(TrackEmbeddingHit {
                track_key: key,
                track_uid: self.tracker.uid_of(key),
                subject_id: self.track_subjects.get(&(source, local)).copied(),
                embedding: handle,
                score,
            });
        }
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| a.track_key.source_id.0.cmp(&b.track_key.source_id.0))
                .then_with(|| {
                    a.track_key
                        .local_track_id
                        .0
                        .cmp(&b.track_key.local_track_id.0)
                })
        });
        if top_k > 0 && hits.len() > top_k {
            hits.truncate(top_k);
        }
        Ok(hits)
    }

    /// Resolves identity for every track that has pending embeddings.
    ///
    /// Each pending key carries its own `source_id` — frames from different
    /// cameras are never mixed into one resolve context.
    ///
    /// Returns the number of tracks resolved.
    ///
    /// # Errors
    ///
    /// Propagates the first identity resolution error.
    pub fn resolve_pending_identities(
        &mut self,
        at: MediaTime,
        modality: Option<SubjectModality>,
    ) -> Result<usize, SessionError> {
        let keys: Vec<(u32, u32)> = self.pending_embeddings.keys().copied().collect();
        let mut resolved = 0_usize;
        for (source, local) in keys {
            let key = TrackKey::new(SourceId(source), TrackId(local));
            self.resolve_track_identity(key, modality, at)?;
            resolved = resolved.saturating_add(1);
        }
        Ok(resolved)
    }

    /// Applies a manual identity confirmation and patches the track map/sample.
    ///
    /// # Errors
    ///
    /// Propagates gallery audit lookup errors.
    pub fn confirm_identity(
        &mut self,
        audit_id: u64,
        confirm: bool,
        subject_id: Option<SubjectId>,
        key: TrackKey,
    ) -> Result<(), SessionError> {
        self.gallery.confirm_manual(audit_id, confirm, subject_id)?;
        let map_key = (key.source_id.0, key.local_track_id.0);
        if confirm {
            if let Some(subject_id) = subject_id {
                self.track_subjects.insert(map_key, subject_id);
                self.patch_latest_track_subject(key, subject_id);
            }
        } else {
            self.track_subjects.remove(&map_key);
        }
        Ok(())
    }

    /// Feeds tracked boxes into a zone monitor and records envelopes.
    ///
    /// Envelopes inherit any subject already assigned to the track key.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::Analytics`] when the zone monitor cannot accept
    /// an update (capacity / output).
    pub fn ingest_zone_updates<const N: usize>(
        &mut self,
        stamp: FrameStamp,
        zone: &mut ZoneAnalytics<'_, N>,
        tracked: &[TrackedDetection],
    ) -> Result<usize, SessionError> {
        let mut total = 0_usize;
        let mut output = [AnalyticsEvent::OccupancyChanged {
            zone_id: sightloom_core::ZoneId(0),
            occupancy: 0,
        }; 8];

        for item in tracked {
            let track_id = item.track_key.local_track_id;
            let subject_id = self
                .track_subjects
                .get(&(item.track_key.source_id.0, track_id.0))
                .copied();
            let count = zone
                .update(
                    track_id,
                    item.detection.bbox(),
                    item.detection.class_id(),
                    None,
                    stamp.frame_index,
                    stamp.pts,
                    &mut output,
                )
                .map_err(|_| SessionError::Analytics)?;
            for event in output.iter().take(count).copied() {
                let event_id = EventId(self.next_event_id);
                self.next_event_id = self.next_event_id.saturating_add(1);
                let envelope = analytics_to_envelope(event_id, stamp, event, subject_id);
                self.index.push_event(envelope);
                total = total.saturating_add(1);
            }
        }
        Ok(total)
    }

    /// Stores a compact mask blob and returns its handle value for track rows.
    pub fn store_mask_bytes(&mut self, bytes: impl Into<Vec<u8>>) -> u64 {
        self.index.masks.insert(bytes).0
    }

    /// Attaches a mask handle by appending a correction sample for a track key.
    pub fn attach_mask_to_latest_track(&mut self, key: TrackKey, mask_ref: u64) -> bool {
        let samples = self.index.tracks.samples();
        let Some(sample) = samples
            .iter()
            .rev()
            .find(|sample| sample.track_key() == key)
            .copied()
        else {
            return false;
        };
        let mut updated = sample;
        updated.mask_ref = mask_ref;
        if let Some(subject_id) = self.subject_for_track_key(key) {
            updated.subject_id = Some(subject_id);
        }
        if sample.sample_id != 0 {
            self.index.tracks.push_revision(updated, sample.sample_id);
        } else {
            self.index.push_track(updated);
        }
        true
    }

    /// Materializes a JSON `VisionIndex` snapshot.
    ///
    /// # Errors
    ///
    /// Returns serialization failures.
    pub fn materialize_json(&self) -> Result<String, SessionError> {
        self.index
            .validate_fast()
            .map_err(|error| SessionError::Serialize(format!("invalid index: {error:?}")))?;
        VisionIndexSnapshot::from_index(&self.index)
            .to_json()
            .map_err(|error| SessionError::Serialize(format!("{error:?}")))
    }

    /// Saves the live `VisionIndex` as an on-disk package directory.
    ///
    /// Also writes `gallery.json` (identity gallery + track embedding index)
    /// into the active generation so package load can restore re-id state
    /// without a full session checkpoint.
    ///
    /// # Errors
    ///
    /// Returns package I/O or serialization failures.
    pub fn save_package(&self, dir: impl AsRef<Path>) -> Result<(), SessionError> {
        let dir = dir.as_ref();
        VisionIndexPackage::save(&self.index, dir)
            .map_err(|error| SessionError::Serialize(format!("{error:?}")))?;
        self.write_gallery_sidecar(dir)?;
        Ok(())
    }

    fn write_gallery_sidecar(&self, package_dir: &Path) -> Result<(), SessionError> {
        let payload_dir = VisionIndexPackage::active_payload_dir(package_dir);
        let dto = PackageGalleryDto {
            schema_version: 1,
            gallery: gallery_to_dto(&self.gallery),
            track_embeddings: self
                .track_embeddings
                .iter()
                .map(|(&(source, local), emb)| TrackEmbeddingEntry {
                    source_id: source,
                    local_track_id: local,
                    embedding: emb.0,
                })
                .collect(),
            track_subjects: self
                .track_subjects
                .iter()
                .map(|(&(source, local), subject)| TrackSubjectEntry {
                    source_id: source,
                    local_track_id: local,
                    subject_id: subject.0,
                })
                .collect(),
        };
        let text = serde_json::to_string_pretty(&dto)
            .map_err(|error| SessionError::Serialize(error.to_string()))?;
        std::fs::write(payload_dir.join(sightloom_index::GALLERY_FILE), text)
            .map_err(|error| SessionError::Serialize(error.to_string()))?;
        Ok(())
    }

    fn try_load_gallery_sidecar(&mut self, package_dir: &Path) -> Result<(), SessionError> {
        let path =
            VisionIndexPackage::active_payload_dir(package_dir).join(sightloom_index::GALLERY_FILE);
        if !path.exists() {
            return Ok(());
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|error| SessionError::Serialize(error.to_string()))?;
        let dto: PackageGalleryDto = serde_json::from_str(&text)
            .map_err(|error| SessionError::Serialize(error.to_string()))?;
        restore_gallery(&mut self.gallery, &dto.gallery)?;
        self.track_embeddings.clear();
        for entry in dto.track_embeddings {
            self.track_embeddings.insert(
                (entry.source_id, entry.local_track_id),
                EmbeddingRef(entry.embedding),
            );
        }
        for entry in dto.track_subjects {
            self.track_subjects.insert(
                (entry.source_id, entry.local_track_id),
                SubjectId(entry.subject_id),
            );
        }
        Ok(())
    }

    /// Saves index package **and** full live session runtime checkpoint.
    ///
    /// After a process restart, [`Self::load_checkpoint`] continues ingest with
    /// the same counters, tracker/Kalman state, gallery embeddings, and pending
    /// identity work.
    ///
    /// # Errors
    ///
    /// Returns package or checkpoint serialization failures.
    pub fn save_checkpoint(&self, dir: impl AsRef<Path>) -> Result<(), SessionError> {
        let dir = dir.as_ref();
        self.save_package(dir)?;
        let dto = self.checkpoint_dto();
        let text = serde_json::to_string_pretty(&dto)
            .map_err(|error| SessionError::Serialize(error.to_string()))?;
        // Write into the active generation when present, else package root.
        let target = checkpoint_write_path(dir);
        std::fs::write(&target, text)
            .map_err(|error| SessionError::Serialize(error.to_string()))?;
        Ok(())
    }

    /// Loads a package directory into a new session (fresh tracker/gallery).
    ///
    /// Prefer [`Self::load_checkpoint`] when continuing live ingest.
    ///
    /// # Errors
    ///
    /// Returns package load or tracker config failures.
    pub fn load_package(
        dir: impl AsRef<Path>,
        track_config: ByteTrackConfig,
    ) -> Result<Self, SessionError> {
        let dir = dir.as_ref();
        let index = VisionIndexPackage::load(dir)
            .map_err(|error| SessionError::Serialize(format!("{error:?}")))?;
        let mut session = Self::new(index.header.name.clone(), track_config)?;
        session.index = index;
        // Prefer gallery sidecar when present (subjects + embeddings + track index).
        session.try_load_gallery_sidecar(dir)?;
        // Fill any missing track→subject from samples (latest wins).
        for sample in session.index.tracks.samples() {
            if let Some(subject_id) = sample.subject_id {
                session
                    .track_subjects
                    .entry((sample.source_id.0, sample.track_id.0))
                    .or_insert(subject_id);
            }
        }
        // Advance event id counter past loaded events.
        session.next_event_id = session
            .index
            .events
            .iter()
            .map(|e| e.event_id.0)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
            .max(1);
        Ok(session)
    }

    /// Loads index package + full session runtime checkpoint.
    ///
    /// # Errors
    ///
    /// Returns I/O, deserialization, or restore failures.
    pub fn load_checkpoint(dir: impl AsRef<Path>) -> Result<Self, SessionError> {
        let dir = dir.as_ref();
        let checkpoint_path = find_checkpoint_path(dir)
            .ok_or_else(|| SessionError::Serialize("session_checkpoint.json not found".into()))?;
        let text = std::fs::read_to_string(&checkpoint_path)
            .map_err(|error| SessionError::Serialize(error.to_string()))?;
        let dto: SessionCheckpointDto = serde_json::from_str(&text)
            .map_err(|error| SessionError::Serialize(error.to_string()))?;
        let track_config = ByteTrackConfig {
            track_high_thresh: dto.track_config.track_high_thresh,
            track_activation_thresh: dto.track_config.track_activation_thresh,
            track_low_thresh: dto.track_config.track_low_thresh,
            match_thresh: dto.track_config.match_thresh,
            max_time_lost: dto.track_config.max_time_lost,
            class_aware: dto.track_config.class_aware,
        };
        let index = VisionIndexPackage::load(dir)
            .map_err(|error| SessionError::Serialize(format!("{error:?}")))?;
        let tracker = MultiSourceTracker::restore(track_config, dto.tracker.to_runtime())
            .map_err(SessionError::from)?;
        let mut gallery = SubjectGallery::new();
        restore_gallery(&mut gallery, &dto.gallery)?;
        let mut track_subjects = HashMap::new();
        for entry in dto.track_subjects {
            track_subjects.insert(
                (entry.source_id, entry.local_track_id),
                SubjectId(entry.subject_id),
            );
        }
        let mut pending_embeddings = HashMap::new();
        for entry in dto.pending_embeddings {
            let obs = entry
                .observations
                .into_iter()
                .map(|o| {
                    let at = MediaTime::new(o.at_ticks, o.at_timescale)
                        .unwrap_or_else(|_| MediaTime::default());
                    EmbeddingObservation {
                        embedding: EmbeddingRef(o.embedding),
                        at,
                    }
                })
                .collect();
            pending_embeddings.insert((entry.source_id, entry.local_track_id), obs);
        }
        let mut track_embeddings = HashMap::new();
        for entry in dto.track_embeddings {
            track_embeddings.insert(
                (entry.source_id, entry.local_track_id),
                EmbeddingRef(entry.embedding),
            );
        }
        let next_redaction_id = index
            .redaction_intervals
            .iter()
            .map(|r| r.interval_id.0)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
            .max(1);
        Ok(Self {
            tracker,
            index,
            gallery,
            next_event_id: dto.next_event_id.max(1),
            track_subjects,
            pending_embeddings,
            auto_assign_subjects: dto.auto_assign_subjects,
            default_modality: modality_from_str(&dto.default_modality),
            embedding_model_id: dto.embedding_model_id,
            ingest_policy: IngestPolicy::default(),
            watermarks: HashMap::new(),
            metrics: IngestMetrics::default(),
            next_pattern_id: 1,
            next_anomaly_id: 1,
            next_appearance_id: 1,
            next_visit_id: 1,
            next_redaction_id,
            anomaly_config: sightloom_analysis::StatAnomalyConfig::default(),
            anomaly_baseline: None,
            memory_build: sightloom_index::MemoryBuildConfig::default(),
            memory_auto: MemoryAutoRebuild::default(),
            frames_since_memory_rebuild: 0,
            last_auto_memory_rebuild: None,
            track_embeddings,
        })
    }

    /// Helper for tests: bottom-center anchor point of a box.
    #[must_use]
    pub fn bottom_center(bbox: Rect) -> Point {
        Point::new(bbox.left() * 0.5 + bbox.right() * 0.5, bbox.bottom())
            .unwrap_or_else(|_| bbox.center())
    }

    fn patch_latest_track_subject(&mut self, key: TrackKey, subject_id: SubjectId) {
        let samples = self.index.tracks.samples();
        let Some(sample) = samples
            .iter()
            .rev()
            .find(|sample| sample.track_key() == key)
            .copied()
        else {
            return;
        };
        let mut updated = sample;
        updated.subject_id = Some(subject_id);
        // Revision semantics: supersede prior row so effective view is unique.
        if sample.sample_id != 0 {
            self.index.tracks.push_revision(updated, sample.sample_id);
        } else {
            self.index.push_track(updated);
        }
    }

    fn checkpoint_dto(&self) -> SessionCheckpointDto {
        let mut track_subjects = Vec::new();
        for ((source, local), subject) in &self.track_subjects {
            track_subjects.push(TrackSubjectEntry {
                source_id: *source,
                local_track_id: *local,
                subject_id: subject.0,
            });
        }
        track_subjects.sort_by_key(|e| (e.source_id, e.local_track_id));

        let mut pending_embeddings = Vec::new();
        for ((source, local), obs) in &self.pending_embeddings {
            pending_embeddings.push(PendingEmbeddingEntry {
                source_id: *source,
                local_track_id: *local,
                observations: obs
                    .iter()
                    .map(|o| PendingObservationDto {
                        embedding: o.embedding.0,
                        at_ticks: o.at.ticks(),
                        at_timescale: o.at.timescale(),
                    })
                    .collect(),
            });
        }
        pending_embeddings.sort_by_key(|e| (e.source_id, e.local_track_id));

        let mut track_embeddings = Vec::new();
        for (&(source, local), emb) in &self.track_embeddings {
            track_embeddings.push(TrackEmbeddingEntry {
                source_id: source,
                local_track_id: local,
                embedding: emb.0,
            });
        }
        track_embeddings.sort_by_key(|e| (e.source_id, e.local_track_id));

        let cfg = self.tracker.config();
        SessionCheckpointDto {
            schema_version: SESSION_CHECKPOINT_VERSION,
            next_event_id: self.next_event_id,
            auto_assign_subjects: self.auto_assign_subjects,
            default_modality: modality_to_str(self.default_modality).into(),
            embedding_model_id: self.embedding_model_id.clone(),
            track_config: TrackConfigDto {
                track_high_thresh: cfg.track_high_thresh,
                track_activation_thresh: cfg.track_activation_thresh,
                track_low_thresh: cfg.track_low_thresh,
                match_thresh: cfg.match_thresh,
                max_time_lost: cfg.max_time_lost,
                class_aware: cfg.class_aware,
            },
            tracker: MultiSourceCheckpointDto::from_runtime(self.tracker.checkpoint()),
            track_subjects,
            pending_embeddings,
            gallery: gallery_to_dto(&self.gallery),
            track_embeddings,
        }
    }
}

/// Photo search hit plus optional evidence reel for host UI.
#[derive(Clone, Debug, PartialEq)]
pub struct PhotoSearchResult {
    /// Ranked gallery match.
    pub hit: sightloom_reid::PhotoSearchHit,
    /// Coalesced reel when Accept or Uncertain.
    pub reel: Option<sightloom_index::EvidenceReel>,
}

/// Cosine hit against a track embedding (may be unlabeled).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackEmbeddingHit {
    /// Track key.
    pub track_key: TrackKey,
    /// Global uid when known.
    pub track_uid: Option<TrackUid>,
    /// Subject if already assigned.
    pub subject_id: Option<SubjectId>,
    /// Embedding handle.
    pub embedding: EmbeddingRef,
    /// Cosine similarity.
    pub score: f32,
}

/// Host-facing export of one effective track/mask sample (no pixels).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackSpanExport {
    /// Sample id.
    pub sample_id: u64,
    /// Source.
    pub source_id: SourceId,
    /// Frame index.
    pub frame_index: u64,
    /// Presentation time.
    pub pts: MediaTime,
    /// Composite track key.
    pub track_key: TrackKey,
    /// Global track uid.
    pub track_uid: Option<TrackUid>,
    /// Subject when known.
    pub subject_id: Option<SubjectId>,
    /// Box edges.
    pub left: f32,
    /// Top.
    pub top: f32,
    /// Right.
    pub right: f32,
    /// Bottom.
    pub bottom: f32,
    /// Confidence.
    pub confidence: f32,
    /// Mask handle (`0` = none).
    pub mask_ref: u64,
    /// Revision number.
    pub revision: u32,
}

const SESSION_CHECKPOINT_VERSION: u32 = 1;
const CHECKPOINT_FILE: &str = "session_checkpoint.json";

fn checkpoint_write_path(package_dir: &Path) -> PathBuf {
    if let Some(generation) = VisionIndexPackage::current_generation(package_dir) {
        package_dir.join(generation).join(CHECKPOINT_FILE)
    } else {
        package_dir.join(CHECKPOINT_FILE)
    }
}

fn find_checkpoint_path(package_dir: &Path) -> Option<PathBuf> {
    if let Some(generation) = VisionIndexPackage::current_generation(package_dir) {
        let path = package_dir.join(generation).join(CHECKPOINT_FILE);
        if path.exists() {
            return Some(path);
        }
    }
    let root = package_dir.join(CHECKPOINT_FILE);
    if root.exists() { Some(root) } else { None }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct SessionCheckpointDto {
    schema_version: u32,
    next_event_id: u64,
    auto_assign_subjects: bool,
    default_modality: String,
    embedding_model_id: Option<String>,
    track_config: TrackConfigDto,
    tracker: MultiSourceCheckpointDto,
    track_subjects: Vec<TrackSubjectEntry>,
    pending_embeddings: Vec<PendingEmbeddingEntry>,
    gallery: GalleryCheckpointDto,
    #[serde(default)]
    track_embeddings: Vec<TrackEmbeddingEntry>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct TrackConfigDto {
    track_high_thresh: f32,
    track_activation_thresh: f32,
    track_low_thresh: f32,
    match_thresh: f32,
    max_time_lost: u32,
    class_aware: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct TrackSubjectEntry {
    source_id: u32,
    local_track_id: u32,
    subject_id: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct TrackEmbeddingEntry {
    source_id: u32,
    local_track_id: u32,
    embedding: u64,
}

/// Sidecar written as `gallery.json` inside a package generation.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct PackageGalleryDto {
    schema_version: u32,
    gallery: GalleryCheckpointDto,
    track_embeddings: Vec<TrackEmbeddingEntry>,
    track_subjects: Vec<TrackSubjectEntry>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct PendingEmbeddingEntry {
    source_id: u32,
    local_track_id: u32,
    observations: Vec<PendingObservationDto>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct PendingObservationDto {
    embedding: u64,
    at_ticks: i64,
    at_timescale: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct MultiSourceCheckpointDto {
    next_uid: u64,
    sources: Vec<SourceTrackerDto>,
    uids: Vec<UidMapEntryDto>,
}

impl MultiSourceCheckpointDto {
    fn from_runtime(cp: MultiSourceCheckpoint) -> Self {
        Self {
            next_uid: cp.next_uid,
            sources: cp
                .sources
                .into_iter()
                .map(|s| SourceTrackerDto {
                    source_id: s.source_id,
                    tracker: TrackerSnapshotDto::from_runtime(s.tracker),
                })
                .collect(),
            uids: cp
                .uids
                .into_iter()
                .map(|u| UidMapEntryDto {
                    source_id: u.source_id,
                    local_track_id: u.local_track_id,
                    track_uid: u.track_uid,
                })
                .collect(),
        }
    }

    fn to_runtime(self) -> MultiSourceCheckpoint {
        MultiSourceCheckpoint {
            next_uid: self.next_uid,
            sources: self
                .sources
                .into_iter()
                .map(|s| SourceTrackerCheckpoint {
                    source_id: s.source_id,
                    tracker: s.tracker.to_runtime(),
                })
                .collect(),
            uids: self
                .uids
                .into_iter()
                .map(|u| UidMapEntry {
                    source_id: u.source_id,
                    local_track_id: u.local_track_id,
                    track_uid: u.track_uid,
                })
                .collect(),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct SourceTrackerDto {
    source_id: u32,
    tracker: TrackerSnapshotDto,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct UidMapEntryDto {
    source_id: u32,
    local_track_id: u32,
    track_uid: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct TrackerSnapshotDto {
    frame_id: u64,
    next_id: u32,
    tracks: Vec<TrackDto>,
}

impl TrackerSnapshotDto {
    fn from_runtime(s: TrackerSnapshot) -> Self {
        Self {
            frame_id: s.frame_id,
            next_id: s.next_id,
            tracks: s.tracks.into_iter().map(TrackDto::from_runtime).collect(),
        }
    }

    fn to_runtime(self) -> TrackerSnapshot {
        TrackerSnapshot {
            frame_id: self.frame_id,
            next_id: self.next_id,
            tracks: self.tracks.into_iter().map(TrackDto::to_runtime).collect(),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct TrackDto {
    id: u32,
    state: u8,
    class_id: Option<u16>,
    mean: [f32; 8],
    variance: [f32; 8],
    score: f32,
    time_since_update: u32,
    hits: u32,
    start_frame: u64,
    frame_id: u64,
}

impl TrackDto {
    fn from_runtime(t: Track) -> Self {
        Self {
            id: t.id.0,
            state: match t.state {
                TrackState::New => 0,
                TrackState::Tracked => 1,
                TrackState::Lost => 2,
                TrackState::Removed => 3,
            },
            class_id: t.class_id.map(|c| c.0),
            mean: t.kalman.mean,
            variance: t.kalman.variance,
            score: t.score,
            time_since_update: t.time_since_update,
            hits: t.hits,
            start_frame: t.start_frame,
            frame_id: t.frame_id,
        }
    }

    fn to_runtime(self) -> Track {
        Track {
            id: TrackId(self.id),
            state: match self.state {
                1 => TrackState::Tracked,
                2 => TrackState::Lost,
                3 => TrackState::Removed,
                _ => TrackState::New,
            },
            class_id: self.class_id.map(sightloom_core::ClassId),
            kalman: sightloom_tracking::KalmanState {
                mean: self.mean,
                variance: self.variance,
            },
            score: self.score,
            time_since_update: self.time_since_update,
            hits: self.hits,
            start_frame: self.start_frame,
            frame_id: self.frame_id,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct GalleryCheckpointDto {
    next_subject_id: u64,
    next_audit_id: u64,
    resolve_config: ResolveConfigDto,
    embeddings_next_id: u64,
    embeddings: Vec<EmbeddingEntryDto>,
    subjects: Vec<SubjectRefDto>,
    audit: Vec<AuditEventDto>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ResolveConfigDto {
    accept_threshold: f32,
    reject_threshold: f32,
    require_same_modality: bool,
    negative_reject_threshold: f32,
    strict_camera_topology: bool,
    max_identity_gap_ns: Option<i64>,
    default_source_accept: Option<f32>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct EmbeddingEntryDto {
    handle: u64,
    vector: Vec<f32>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct SubjectRefDto {
    subject_id: u64,
    modality: String,
    samples: Vec<ReferenceSampleDto>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ReferenceSampleDto {
    source_id: Option<u32>,
    track_id: Option<u32>,
    at_ticks: Option<i64>,
    at_timescale: Option<u32>,
    embedding: Option<u64>,
    evidence: Option<u64>,
    is_positive: Option<bool>,
    quality: Option<f32>,
    class_id: Option<u16>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct AuditEventDto {
    audit_id: u64,
    track_id: u32,
    source_id: u32,
    assigned_subject: Option<u64>,
    manual_confirmation: Option<bool>,
    at_ticks: i64,
    at_timescale: u32,
    modality: String,
    best_subject: Option<u64>,
    best_score: Option<f32>,
    best_decision: Option<String>,
}

fn gallery_to_dto(gallery: &SubjectGallery) -> GalleryCheckpointDto {
    let cfg = gallery.resolve_config();
    GalleryCheckpointDto {
        next_subject_id: gallery.next_subject_id(),
        next_audit_id: gallery.next_audit_id(),
        resolve_config: ResolveConfigDto {
            accept_threshold: cfg.accept_threshold,
            reject_threshold: cfg.reject_threshold,
            require_same_modality: cfg.require_same_modality,
            negative_reject_threshold: cfg.negative_reject_threshold,
            strict_camera_topology: cfg.strict_camera_topology,
            max_identity_gap_ns: cfg.max_identity_gap_ns,
            default_source_accept: cfg.default_source_accept,
        },
        embeddings_next_id: gallery.embeddings.next_id(),
        embeddings: gallery
            .embeddings
            .entries()
            .iter()
            .map(|(h, v)| EmbeddingEntryDto {
                handle: h.0,
                vector: v.clone(),
            })
            .collect(),
        subjects: gallery
            .subjects()
            .iter()
            .map(|s| SubjectRefDto {
                subject_id: s.subject_id.0,
                modality: modality_to_str(s.modality).into(),
                samples: s
                    .samples
                    .iter()
                    .map(|sample| ReferenceSampleDto {
                        source_id: sample.source_id.map(|id| id.0),
                        track_id: sample.track_id.map(|id| id.0),
                        at_ticks: sample.at.map(MediaTime::ticks),
                        at_timescale: sample.at.map(MediaTime::timescale),
                        embedding: sample.embedding.map(|e| e.0),
                        evidence: sample.evidence.map(|e| e.0),
                        is_positive: sample.is_positive,
                        quality: sample.quality,
                        class_id: sample.class_id.map(|c| c.0),
                    })
                    .collect(),
            })
            .collect(),
        audit: gallery
            .audit()
            .iter()
            .map(|a| AuditEventDto {
                audit_id: a.audit_id,
                track_id: a.fragment.track_id.0,
                source_id: a.fragment.source_id.0,
                assigned_subject: a.assigned_subject.map(|s| s.0),
                manual_confirmation: a.manual_confirmation,
                at_ticks: a.at.ticks(),
                at_timescale: a.at.timescale(),
                modality: modality_to_str(a.fragment.modality).into(),
                best_subject: a.best_match.map(|m| m.subject_id.0),
                best_score: a.best_match.map(|m| m.score),
                best_decision: a.best_match.map(|m| decision_to_str(m.decision).into()),
            })
            .collect(),
    }
}

fn restore_gallery(
    gallery: &mut SubjectGallery,
    dto: &GalleryCheckpointDto,
) -> Result<(), SessionError> {
    let mut store = EmbeddingStore::new();
    store.restore_from(
        dto.embeddings_next_id,
        dto.embeddings
            .iter()
            .map(|e| (EmbeddingRef(e.handle), e.vector.clone()))
            .collect(),
    );
    let subjects: Vec<SubjectReference> = dto
        .subjects
        .iter()
        .map(|s| {
            let mut subject =
                SubjectReference::new(SubjectId(s.subject_id), modality_from_str(&s.modality));
            for sample in &s.samples {
                let at = match (sample.at_ticks, sample.at_timescale) {
                    (Some(ticks), Some(timescale)) => MediaTime::new(ticks, timescale).ok(),
                    _ => None,
                };
                subject.push_sample(ReferenceSample {
                    source_id: sample.source_id.map(SourceId),
                    track_id: sample.track_id.map(TrackId),
                    at,
                    embedding: sample.embedding.map(EmbeddingRef),
                    evidence: sample.evidence.map(sightloom_core::EvidenceRef),
                    is_positive: sample.is_positive,
                    quality: sample.quality,
                    class_id: sample.class_id.map(sightloom_core::ClassId),
                });
            }
            subject
        })
        .collect();
    let audit: Vec<IdentityAuditEvent> = dto
        .audit
        .iter()
        .map(|a| {
            let at = MediaTime::new(a.at_ticks, a.at_timescale).unwrap_or_default();
            let best_match = match (a.best_subject, a.best_score, a.best_decision.as_deref()) {
                (Some(sid), Some(score), Some(dec)) => Some(IdentityMatch {
                    subject_id: SubjectId(sid),
                    score,
                    decision: decision_from_str(dec),
                    factors: sightloom_reid::IdentityScoreFactors::default(),
                }),
                _ => None,
            };
            IdentityAuditEvent {
                audit_id: a.audit_id,
                fragment: TrackFragment {
                    track_id: TrackId(a.track_id),
                    source_id: SourceId(a.source_id),
                    start: at,
                    end: at,
                    embedding: None,
                    subject_id: a.assigned_subject.map(SubjectId),
                    modality: modality_from_str(&a.modality),
                    embedding_quality: 1.0,
                    class_id: None,
                },
                best_match,
                hypotheses: best_match.into_iter().collect(),
                assigned_subject: a.assigned_subject.map(SubjectId),
                manual_confirmation: a.manual_confirmation,
                at,
            }
        })
        .collect();
    let resolve = ResolveConfig {
        accept_threshold: dto.resolve_config.accept_threshold,
        reject_threshold: dto.resolve_config.reject_threshold,
        require_same_modality: dto.resolve_config.require_same_modality,
        negative_reject_threshold: dto.resolve_config.negative_reject_threshold,
        strict_camera_topology: dto.resolve_config.strict_camera_topology,
        max_identity_gap_ns: dto.resolve_config.max_identity_gap_ns,
        default_source_accept: dto.resolve_config.default_source_accept,
    };
    gallery
        .restore(
            dto.next_subject_id,
            dto.next_audit_id,
            subjects,
            audit,
            store,
            resolve,
        )
        .map_err(SessionError::from)
}

fn modality_to_str(modality: SubjectModality) -> &'static str {
    match modality {
        SubjectModality::Face => "face",
        SubjectModality::PersonAppearance => "person_appearance",
        SubjectModality::VehicleAppearance => "vehicle_appearance",
        SubjectModality::LicensePlate => "license_plate",
        SubjectModality::GenericObject => "generic_object",
    }
}

fn modality_from_str(value: &str) -> SubjectModality {
    match value {
        "face" => SubjectModality::Face,
        "vehicle_appearance" => SubjectModality::VehicleAppearance,
        "license_plate" => SubjectModality::LicensePlate,
        "generic_object" => SubjectModality::GenericObject,
        _ => SubjectModality::PersonAppearance,
    }
}

fn decision_to_str(decision: MatchDecision) -> &'static str {
    match decision {
        MatchDecision::Accept => "accept",
        MatchDecision::Reject => "reject",
        MatchDecision::Uncertain => "uncertain",
    }
}

fn decision_from_str(value: &str) -> MatchDecision {
    match value {
        "accept" => MatchDecision::Accept,
        "reject" => MatchDecision::Reject,
        _ => MatchDecision::Uncertain,
    }
}
