//! Subject gallery, merge/split helpers, and audit trail.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::collections::BTreeMap;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::{
    CameraTopology, EmbeddingError, EmbeddingStore, IdentityInterval, IdentityMatch,
    IdentityResolver, MatchDecision, ReferenceSample, ResolveConfig, ScoreContext, SubjectLastSeen,
    SubjectModality, SubjectReference, ThresholdResolver, TrackFragment, uncertain_only,
};
use sightloom_core::{MediaTime, SourceId, SubjectId};

/// One audited identity decision.
#[derive(Clone, Debug, PartialEq)]
pub struct IdentityAuditEvent {
    /// Monotonic audit id.
    pub audit_id: u64,
    /// Fragment that was resolved.
    pub fragment: TrackFragment,
    /// Best match considered (if any).
    pub best_match: Option<IdentityMatch>,
    /// All ranked hypotheses (multiple identities).
    pub hypotheses: Vec<IdentityMatch>,
    /// Assigned subject after automatic and optional manual decision.
    pub assigned_subject: Option<SubjectId>,
    /// Manual confirmation when present (`true` confirm, `false` reject).
    pub manual_confirmation: Option<bool>,
    /// Event time.
    pub at: MediaTime,
}

/// In-memory gallery of subjects plus embeddings and audit log.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug, Default)]
pub struct SubjectGallery {
    next_subject_id: u64,
    next_audit_id: u64,
    subjects: Vec<SubjectReference>,
    /// Out-of-line embedding vectors.
    pub embeddings: EmbeddingStore,
    audit: Vec<IdentityAuditEvent>,
    resolve_config: ResolveConfig,
    /// Camera travel constraints for cross-source gating.
    topology: CameraTopology,
    /// Last accepted sighting per subject: `(source, time)`.
    last_seen: SubjectLastSeen,
    /// Per-source accept thresholds (fused or cosine, depending on config path).
    source_accept: BTreeMap<u32, f32>,
    /// Max reference samples retained per subject (`0` = unlimited).
    max_references_per_subject: usize,
}

#[cfg(feature = "alloc")]
impl SubjectGallery {
    /// Creates an empty gallery with default resolve thresholds.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_subject_id: 1,
            next_audit_id: 1,
            subjects: Vec::new(),
            embeddings: EmbeddingStore::new(),
            audit: Vec::new(),
            resolve_config: ResolveConfig::default(),
            topology: CameraTopology::new(),
            last_seen: BTreeMap::new(),
            source_accept: BTreeMap::new(),
            max_references_per_subject: 0,
        }
    }

    /// Sets camera topology used during multi-factor resolve.
    pub fn set_topology(&mut self, topology: CameraTopology) {
        self.topology = topology;
    }

    /// Returns the active camera topology.
    #[must_use]
    pub fn topology(&self) -> &CameraTopology {
        &self.topology
    }

    /// Caps reference samples per subject (FIFO eviction of oldest).
    pub fn set_max_references_per_subject(&mut self, max: usize) {
        self.max_references_per_subject = max;
    }

    /// Sets a per-camera accept threshold override.
    pub fn set_source_accept_threshold(&mut self, source_id: SourceId, threshold: f32) {
        self.source_accept.insert(source_id.0, threshold);
    }

    /// Overrides resolve thresholds.
    ///
    /// # Errors
    ///
    /// Returns validation errors from [`ResolveConfig::validate`].
    pub fn set_resolve_config(&mut self, config: ResolveConfig) -> Result<(), EmbeddingError> {
        self.resolve_config = config.validate()?;
        Ok(())
    }

    /// Returns all subjects.
    #[must_use]
    pub fn subjects(&self) -> &[SubjectReference] {
        &self.subjects
    }

    /// Returns the audit trail.
    #[must_use]
    pub fn audit(&self) -> &[IdentityAuditEvent] {
        &self.audit
    }

    /// Registers a new subject with the given modality.
    pub fn register_subject(&mut self, modality: SubjectModality) -> SubjectId {
        let id = SubjectId(self.next_subject_id);
        self.next_subject_id = self.next_subject_id.saturating_add(1);
        self.subjects.push(SubjectReference::new(id, modality));
        id
    }

    /// Adds a reference sample to an existing subject.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError::NotFound`] when the subject is unknown.
    pub fn add_reference(
        &mut self,
        subject_id: SubjectId,
        sample: ReferenceSample,
    ) -> Result<(), EmbeddingError> {
        let subject = self
            .subjects
            .iter_mut()
            .find(|subject| subject.subject_id == subject_id)
            .ok_or(EmbeddingError::NotFound)?;
        subject.push_sample(sample);
        if self.max_references_per_subject > 0 {
            while subject.samples.len() > self.max_references_per_subject {
                subject.samples.remove(0);
            }
        }
        Ok(())
    }

    /// Resolves a fragment, records audit, and optionally assigns on Accept.
    ///
    /// When `auto_assign` is true and the best match is `Accept`, the fragment
    /// is returned with `subject_id` filled. Multi-factor score uses topology,
    /// temporal last-seen, embedding quality, and class compatibility.
    pub fn resolve_and_audit(
        &mut self,
        mut fragment: TrackFragment,
        auto_assign: bool,
        at: MediaTime,
    ) -> (TrackFragment, Vec<IdentityMatch>) {
        let ctx = ScoreContext {
            query_source: fragment.source_id,
            query_at: at,
            embedding_quality: fragment.embedding_quality,
            prior_identity_confidence: if fragment.subject_id.is_some() {
                0.9
            } else {
                1.0
            },
            class_id: fragment.class_id,
        };
        let Ok(resolver) = ThresholdResolver::with_context(
            &self.embeddings,
            self.resolve_config,
            ctx,
            Some(&self.topology),
            Some(&self.last_seen),
            Some(&self.source_accept),
        ) else {
            return (fragment, Vec::new());
        };
        let matches = resolver.resolve_fragment(&fragment, &self.subjects);
        let best = matches.first().copied();
        let mut assigned = fragment.subject_id;
        if auto_assign
            && let Some(best_match) = best
            && best_match.decision == MatchDecision::Accept
        {
            assigned = Some(best_match.subject_id);
            fragment.subject_id = assigned;
            self.last_seen
                .insert(best_match.subject_id.0, (fragment.source_id, at));
        }
        let audit_id = self.next_audit_id;
        self.next_audit_id = self.next_audit_id.saturating_add(1);
        self.audit.push(IdentityAuditEvent {
            audit_id,
            fragment,
            best_match: best,
            hypotheses: matches.clone(),
            assigned_subject: assigned,
            manual_confirmation: None,
            at,
        });
        (fragment, matches)
    }

    /// Adds a reference photo embedding (positive sample) to a subject.
    ///
    /// # Errors
    ///
    /// Propagates embedding validation or unknown-subject errors.
    pub fn add_reference_photo(
        &mut self,
        subject_id: SubjectId,
        vector: impl Into<alloc::vec::Vec<f32>>,
        quality: Option<f32>,
        source_id: Option<SourceId>,
        at: Option<MediaTime>,
    ) -> Result<sightloom_core::EmbeddingRef, EmbeddingError> {
        let handle = self.embeddings.insert(vector)?;
        self.add_reference(
            subject_id,
            ReferenceSample {
                source_id,
                track_id: None,
                at,
                embedding: Some(handle),
                evidence: None,
                is_positive: Some(true),
                quality,
                class_id: None,
            },
        )?;
        Ok(handle)
    }

    /// Searches the gallery with a photo embedding (multi-factor rank).
    ///
    /// # Errors
    ///
    /// Propagates resolver / store errors.
    pub fn search_by_photo(
        &self,
        query: &crate::PhotoQuery,
        top_k: usize,
    ) -> Result<Vec<crate::PhotoSearchHit>, EmbeddingError> {
        crate::search_gallery_by_photo(
            &self.embeddings,
            &self.subjects,
            self.resolve_config,
            query,
            Some(&self.topology),
            Some(&self.last_seen),
            Some(&self.source_accept),
            top_k,
        )
    }

    /// Calibrates thresholds from labeled embedding pairs `(a, b, genuine)`.
    ///
    /// # Errors
    ///
    /// Propagates store lookup and calibration errors.
    pub fn calibrate_thresholds(
        &self,
        pairs: &[(
            sightloom_core::EmbeddingRef,
            sightloom_core::EmbeddingRef,
            bool,
        )],
        n_thresholds: usize,
    ) -> Result<crate::CalibrationReport, EmbeddingError> {
        let scores = crate::labeled_scores_from_pairs(&self.embeddings, pairs)?;
        crate::compute_roc(&scores, n_thresholds)
    }

    /// Applies a calibration report to the gallery resolve config.
    ///
    /// # Errors
    ///
    /// Returns validation errors from the updated config.
    pub fn apply_calibration(
        &mut self,
        report: &crate::CalibrationReport,
    ) -> Result<(), EmbeddingError> {
        let next = crate::resolve_config_from_calibration(self.resolve_config, report);
        self.set_resolve_config(next)
    }

    /// Keeps only the newest `max` audit events. Returns number dropped.
    ///
    /// `max == 0` leaves the audit trail unchanged.
    pub fn trim_audit(&mut self, max: usize) -> usize {
        if max == 0 || self.audit.len() <= max {
            return 0;
        }
        let drop_n = self.audit.len() - max;
        self.audit.drain(0..drop_n);
        drop_n
    }

    /// Coalesced uncertain identity intervals from the audit trail.
    #[must_use]
    pub fn uncertain_intervals(&self) -> Vec<IdentityInterval> {
        self.uncertain_intervals_gapped(None)
    }

    /// Uncertain intervals with optional max gap for coalescing.
    #[must_use]
    pub fn uncertain_intervals_gapped(&self, max_gap_ns: Option<i64>) -> Vec<IdentityInterval> {
        let points: Vec<_> = self
            .audit
            .iter()
            .filter(|event| event.manual_confirmation != Some(true))
            .filter_map(|event| {
                let m = event.best_match?;
                if m.decision != MatchDecision::Uncertain {
                    return None;
                }
                Some((
                    event.fragment.source_id,
                    event.fragment.track_id,
                    Some(m.subject_id),
                    m.decision,
                    event.at,
                    Some(m.score),
                ))
            })
            .collect();
        uncertain_only(&crate::coalesce_identity_intervals_gapped(
            &points, max_gap_ns,
        ))
    }

    /// Applies manual confirmation to the latest audit entry for a fragment track.
    ///
    /// When `confirm` is true, assigns `subject_id`; when false, clears assignment
    /// for that audit row.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError::NotFound`] when no matching audit row exists.
    pub fn confirm_manual(
        &mut self,
        audit_id: u64,
        confirm: bool,
        subject_id: Option<SubjectId>,
    ) -> Result<(), EmbeddingError> {
        let event = self
            .audit
            .iter_mut()
            .find(|event| event.audit_id == audit_id)
            .ok_or(EmbeddingError::NotFound)?;
        event.manual_confirmation = Some(confirm);
        event.assigned_subject = if confirm { subject_id } else { None };
        if confirm && let Some(sid) = subject_id {
            self.last_seen
                .insert(sid.0, (event.fragment.source_id, event.at));
        }
        Ok(())
    }

    /// Open identity cases: uncertain (or multi-hypothesis) audits without manual resolution.
    #[must_use]
    pub fn open_identity_cases(&self) -> Vec<&IdentityAuditEvent> {
        self.audit
            .iter()
            .filter(|e| e.manual_confirmation.is_none())
            .filter(|e| {
                e.best_match
                    .is_some_and(|m| m.decision == MatchDecision::Uncertain)
                    || e.hypotheses.len() > 1
            })
            .collect()
    }

    /// Accepts one ranked hypothesis subject for an audit row (manual confirm).
    ///
    /// # Errors
    ///
    /// Not found / subject not in that row's hypotheses.
    pub fn accept_hypothesis(
        &mut self,
        audit_id: u64,
        subject_id: SubjectId,
    ) -> Result<(), EmbeddingError> {
        let event = self
            .audit
            .iter()
            .find(|e| e.audit_id == audit_id)
            .ok_or(EmbeddingError::NotFound)?;
        if !event.hypotheses.iter().any(|h| h.subject_id == subject_id) {
            return Err(EmbeddingError::NotFound);
        }
        self.confirm_manual(audit_id, true, Some(subject_id))
    }

    /// Rejects / dismisses an open identity case (no assignment).
    ///
    /// # Errors
    ///
    /// Not found.
    pub fn dismiss_identity_case(&mut self, audit_id: u64) -> Result<(), EmbeddingError> {
        self.confirm_manual(audit_id, false, None)
    }

    /// Per-source accept thresholds map (read-only).
    #[must_use]
    pub fn source_accept_thresholds(&self) -> &BTreeMap<u32, f32> {
        &self.source_accept
    }

    /// Clears a per-source accept override.
    pub fn clear_source_accept_threshold(&mut self, source_id: SourceId) -> bool {
        self.source_accept.remove(&source_id.0).is_some()
    }

    /// Immutable audit trail (full history, including superseded decisions).
    #[must_use]
    pub fn audit_view(&self) -> &[IdentityAuditEvent] {
        &self.audit
    }

    /// Current confirmed/assigned view: latest audit per track key with an assignment.
    #[must_use]
    pub fn assigned_identity_view(&self) -> Vec<(SourceId, sightloom_core::TrackId, SubjectId)> {
        use alloc::collections::BTreeMap as Map;
        let mut latest: Map<(u32, u32), (SourceId, sightloom_core::TrackId, SubjectId, u64)> =
            Map::new();
        for e in &self.audit {
            let key = (e.fragment.source_id.0, e.fragment.track_id.0);
            if let Some(sid) = e.assigned_subject {
                latest.insert(
                    key,
                    (e.fragment.source_id, e.fragment.track_id, sid, e.audit_id),
                );
            }
        }
        latest
            .into_values()
            .map(|(s, t, sid, _)| (s, t, sid))
            .collect()
    }

    /// Removes a subject if present (does not free embedding store slots).
    pub fn remove_subject_if_present(&mut self, subject_id: SubjectId) -> bool {
        if let Some(i) = self
            .subjects
            .iter()
            .position(|s| s.subject_id == subject_id)
        {
            self.subjects.remove(i);
            true
        } else {
            false
        }
    }

    /// Merges `absorb` into `keep`, moving all reference samples.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError::NotFound`] when either subject is missing.
    pub fn merge_subjects(
        &mut self,
        keep: SubjectId,
        absorb: SubjectId,
    ) -> Result<(), EmbeddingError> {
        if keep == absorb {
            return Ok(());
        }
        let absorb_index = self
            .subjects
            .iter()
            .position(|subject| subject.subject_id == absorb)
            .ok_or(EmbeddingError::NotFound)?;
        let absorb_subject = self.subjects.remove(absorb_index);
        let keep_subject = self
            .subjects
            .iter_mut()
            .find(|subject| subject.subject_id == keep)
            .ok_or(EmbeddingError::NotFound)?;
        for sample in absorb_subject.samples {
            keep_subject.push_sample(sample);
        }
        Ok(())
    }

    /// Resolve config currently in use.
    #[must_use]
    pub const fn resolve_config(&self) -> ResolveConfig {
        self.resolve_config
    }

    /// Next subject id counter.
    #[must_use]
    pub const fn next_subject_id(&self) -> u64 {
        self.next_subject_id
    }

    /// Next audit id counter.
    #[must_use]
    pub const fn next_audit_id(&self) -> u64 {
        self.next_audit_id
    }

    /// Restores gallery counters, subjects, audit, embeddings, and resolve config.
    ///
    /// # Errors
    ///
    /// Returns resolve-config validation errors.
    pub fn restore(
        &mut self,
        next_subject_id: u64,
        next_audit_id: u64,
        subjects: Vec<SubjectReference>,
        audit: Vec<IdentityAuditEvent>,
        embeddings: EmbeddingStore,
        resolve_config: ResolveConfig,
    ) -> Result<(), EmbeddingError> {
        self.next_subject_id = next_subject_id.max(1);
        self.next_audit_id = next_audit_id.max(1);
        self.subjects = subjects;
        self.audit = audit;
        self.embeddings = embeddings;
        self.resolve_config = resolve_config.validate()?;
        Ok(())
    }

    /// Splits samples from `source` into a new subject (by sample indices).
    ///
    /// # Errors
    ///
    /// Returns not-found or invalid index errors.
    pub fn split_subject(
        &mut self,
        source: SubjectId,
        sample_indices: &[usize],
        modality: SubjectModality,
    ) -> Result<SubjectId, EmbeddingError> {
        let source_subject = self
            .subjects
            .iter()
            .find(|subject| subject.subject_id == source)
            .ok_or(EmbeddingError::NotFound)?;
        let mut moved = Vec::new();
        for &index in sample_indices {
            let sample = source_subject
                .samples
                .get(index)
                .copied()
                .ok_or(EmbeddingError::InvalidVector)?;
            moved.push(sample);
        }
        // Remove from highest index first.
        let source_subject = self
            .subjects
            .iter_mut()
            .find(|subject| subject.subject_id == source)
            .ok_or(EmbeddingError::NotFound)?;
        let mut ordered = sample_indices.to_vec();
        ordered.sort_unstable();
        ordered.dedup();
        for &index in ordered.iter().rev() {
            if index < source_subject.samples.len() {
                source_subject.samples.remove(index);
            }
        }
        let new_id = self.register_subject(modality);
        let new_subject = self
            .subjects
            .iter_mut()
            .find(|subject| subject.subject_id == new_id)
            .ok_or(EmbeddingError::NotFound)?;
        for sample in moved {
            new_subject.push_sample(sample);
        }
        Ok(new_id)
    }
}
