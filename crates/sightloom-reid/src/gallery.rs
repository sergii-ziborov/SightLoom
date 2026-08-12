//! Subject gallery, merge/split helpers, and audit trail.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::{
    EmbeddingError, EmbeddingStore, IdentityMatch, IdentityResolver, MatchDecision,
    ReferenceSample, ResolveConfig, SubjectModality, SubjectReference, ThresholdResolver,
    TrackFragment,
};
use sightloom_core::{MediaTime, SubjectId};

/// One audited identity decision.
#[derive(Clone, Debug, PartialEq)]
pub struct IdentityAuditEvent {
    /// Monotonic audit id.
    pub audit_id: u64,
    /// Fragment that was resolved.
    pub fragment: TrackFragment,
    /// Best match considered (if any).
    pub best_match: Option<IdentityMatch>,
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
        }
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
        Ok(())
    }

    /// Resolves a fragment, records audit, and optionally assigns on Accept.
    ///
    /// When `auto_assign` is true and the best match is `Accept`, the fragment
    /// is returned with `subject_id` filled.
    pub fn resolve_and_audit(
        &mut self,
        mut fragment: TrackFragment,
        auto_assign: bool,
        at: MediaTime,
    ) -> (TrackFragment, Vec<IdentityMatch>) {
        let Ok(resolver) = ThresholdResolver::new(&self.embeddings, self.resolve_config) else {
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
        }
        let audit_id = self.next_audit_id;
        self.next_audit_id = self.next_audit_id.saturating_add(1);
        self.audit.push(IdentityAuditEvent {
            audit_id,
            fragment,
            best_match: best,
            assigned_subject: assigned,
            manual_confirmation: None,
            at,
        });
        (fragment, matches)
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
        Ok(())
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
