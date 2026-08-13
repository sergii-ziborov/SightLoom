//! Identity gallery and threshold resolver tests.

use sightloom_core::{MediaTime, SourceId, SubjectId, TrackId};
use sightloom_reid::{
    CameraTopology, EmbeddingObservation, MatchDecision, ReferenceSample, ResolveConfig,
    SubjectGallery, SubjectModality, aggregate_fragment,
};

fn pos_sample(embedding: sightloom_core::EmbeddingRef) -> ReferenceSample {
    ReferenceSample {
        source_id: Some(SourceId(1)),
        track_id: None,
        at: None,
        embedding: Some(embedding),
        evidence: None,
        is_positive: Some(true),
        quality: None,
        class_id: None,
    }
}

fn cfg(accept: f32, reject: f32, neg: f32) -> ResolveConfig {
    ResolveConfig {
        accept_threshold: accept,
        reject_threshold: reject,
        require_same_modality: true,
        negative_reject_threshold: neg,
        ..ResolveConfig::default()
    }
}

#[test]
fn accepts_close_positive_and_rejects_negative() {
    let mut gallery = SubjectGallery::new();
    gallery.set_resolve_config(cfg(0.80, 0.30, 0.75)).unwrap();

    let person = gallery.register_subject(SubjectModality::PersonAppearance);
    let pos = gallery.embeddings.insert([1.0_f32, 0.0, 0.0]).unwrap();
    gallery.add_reference(person, pos_sample(pos)).unwrap();

    let neg = gallery.embeddings.insert([0.0_f32, 1.0, 0.0]).unwrap();
    gallery
        .add_reference(
            person,
            ReferenceSample {
                source_id: Some(SourceId(1)),
                track_id: None,
                at: None,
                embedding: Some(neg),
                evidence: None,
                is_positive: Some(false),
                quality: None,
                class_id: None,
            },
        )
        .unwrap();

    let q1 = gallery.embeddings.insert([0.99_f32, 0.01, 0.0]).unwrap();
    let fragment = sightloom_reid::TrackFragment {
        track_id: TrackId(3),
        source_id: SourceId(1),
        start: MediaTime::new(0, 30).unwrap(),
        end: MediaTime::new(5, 30).unwrap(),
        embedding: Some(q1),
        subject_id: None,
        modality: SubjectModality::PersonAppearance,
        embedding_quality: 1.0,
        class_id: None,
    };
    let (assigned, matches) =
        gallery.resolve_and_audit(fragment, true, MediaTime::new(5, 30).unwrap());
    assert_eq!(assigned.subject_id, Some(person));
    assert_eq!(matches[0].decision, MatchDecision::Accept);
    assert_eq!(gallery.audit().len(), 1);
    assert!(!gallery.audit()[0].hypotheses.is_empty());

    let q2 = gallery.embeddings.insert([0.01_f32, 0.99, 0.0]).unwrap();
    let fragment2 = sightloom_reid::TrackFragment {
        track_id: TrackId(4),
        source_id: SourceId(2),
        start: MediaTime::new(10, 30).unwrap(),
        end: MediaTime::new(12, 30).unwrap(),
        embedding: Some(q2),
        subject_id: None,
        modality: SubjectModality::PersonAppearance,
        embedding_quality: 1.0,
        class_id: None,
    };
    let (assigned2, matches2) =
        gallery.resolve_and_audit(fragment2, true, MediaTime::new(12, 30).unwrap());
    assert!(assigned2.subject_id.is_none());
    assert_eq!(matches2[0].decision, MatchDecision::Reject);
}

#[test]
fn topology_blocks_impossible_cross_camera_accept() {
    let mut gallery = SubjectGallery::new();
    gallery
        .set_resolve_config(ResolveConfig {
            accept_threshold: 0.50,
            reject_threshold: 0.10,
            require_same_modality: true,
            negative_reject_threshold: 0.99,
            strict_camera_topology: true,
            max_identity_gap_ns: None,
            default_source_accept: None,
        })
        .unwrap();
    let mut topo = CameraTopology::new();
    // Minimum 60s travel A -> B
    topo.set_edge(SourceId(1), SourceId(2), 60_000_000_000);
    gallery.set_topology(topo);

    let person = gallery.register_subject(SubjectModality::PersonAppearance);
    let pos = gallery.embeddings.insert([1.0_f32, 0.0]).unwrap();
    gallery
        .add_reference(
            person,
            ReferenceSample {
                source_id: Some(SourceId(1)),
                track_id: None,
                at: Some(MediaTime::new(0, 1).unwrap()),
                embedding: Some(pos),
                evidence: None,
                is_positive: Some(true),
                quality: Some(1.0),
                class_id: None,
            },
        )
        .unwrap();

    // First accept on camera A
    let q1 = gallery.embeddings.insert([1.0_f32, 0.0]).unwrap();
    let f1 = sightloom_reid::TrackFragment {
        track_id: TrackId(1),
        source_id: SourceId(1),
        start: MediaTime::new(0, 1).unwrap(),
        end: MediaTime::new(1, 1).unwrap(),
        embedding: Some(q1),
        subject_id: None,
        modality: SubjectModality::PersonAppearance,
        embedding_quality: 1.0,
        class_id: None,
    };
    let (a1, m1) = gallery.resolve_and_audit(f1, true, MediaTime::new(1, 1).unwrap());
    assert_eq!(a1.subject_id, Some(person));
    assert_eq!(m1[0].decision, MatchDecision::Accept);

    // Same face on camera B only 2s later — physically impossible
    let q2 = gallery.embeddings.insert([1.0_f32, 0.0]).unwrap();
    let f2 = sightloom_reid::TrackFragment {
        track_id: TrackId(2),
        source_id: SourceId(2),
        start: MediaTime::new(3, 1).unwrap(),
        end: MediaTime::new(3, 1).unwrap(),
        embedding: Some(q2),
        subject_id: None,
        modality: SubjectModality::PersonAppearance,
        embedding_quality: 1.0,
        class_id: None,
    };
    let (a2, m2) = gallery.resolve_and_audit(f2, true, MediaTime::new(3, 1).unwrap());
    assert!(a2.subject_id.is_none());
    assert_eq!(m2[0].decision, MatchDecision::Reject);
    assert!((m2[0].factors.camera_topology - 0.0).abs() < 1e-6);
}

#[test]
fn aggregate_fragment_mean_pools_and_merge_split_work() {
    let mut gallery = SubjectGallery::new();
    let e1 = gallery.embeddings.insert([1.0_f32, 0.0]).unwrap();
    let e2 = gallery.embeddings.insert([0.0_f32, 1.0]).unwrap();
    let fragment = aggregate_fragment(
        &mut gallery.embeddings,
        TrackId(9),
        SourceId(1),
        SubjectModality::Face,
        &[
            EmbeddingObservation {
                embedding: e1,
                at: MediaTime::new(0, 1).unwrap(),
            },
            EmbeddingObservation {
                embedding: e2,
                at: MediaTime::new(1, 1).unwrap(),
            },
        ],
        None,
    )
    .unwrap();
    let mean = gallery.embeddings.get(fragment.embedding.unwrap()).unwrap();
    assert!((mean[0] - 0.5).abs() < 1e-5);
    assert!((mean[1] - 0.5).abs() < 1e-5);

    let a = gallery.register_subject(SubjectModality::Face);
    let b = gallery.register_subject(SubjectModality::Face);
    gallery
        .add_reference(
            a,
            ReferenceSample {
                source_id: None,
                track_id: None,
                at: None,
                embedding: Some(e1),
                evidence: None,
                is_positive: Some(true),
                quality: None,
                class_id: None,
            },
        )
        .unwrap();
    gallery
        .add_reference(
            b,
            ReferenceSample {
                source_id: None,
                track_id: None,
                at: None,
                embedding: Some(e2),
                evidence: None,
                is_positive: Some(true),
                quality: None,
                class_id: None,
            },
        )
        .unwrap();
    gallery.merge_subjects(a, b).unwrap();
    assert_eq!(gallery.subjects().len(), 1);
    assert_eq!(gallery.subjects()[0].samples.len(), 2);

    let new_id = gallery
        .split_subject(a, &[1], SubjectModality::Face)
        .unwrap();
    assert_ne!(new_id, a);
    assert_eq!(gallery.subjects().len(), 2);
}

#[test]
fn uncertain_band_and_manual_confirmation() {
    let mut gallery = SubjectGallery::new();
    gallery.set_resolve_config(cfg(0.90, 0.20, 0.95)).unwrap();
    let subject = gallery.register_subject(SubjectModality::VehicleAppearance);
    let pos = gallery.embeddings.insert([1.0_f32, 0.0]).unwrap();
    gallery
        .add_reference(
            subject,
            ReferenceSample {
                source_id: None,
                track_id: None,
                at: None,
                embedding: Some(pos),
                evidence: None,
                is_positive: Some(true),
                quality: None,
                class_id: None,
            },
        )
        .unwrap();
    let query = gallery.embeddings.insert([1.0_f32, 1.0]).unwrap();
    let fragment = sightloom_reid::TrackFragment {
        track_id: TrackId(1),
        source_id: SourceId(1),
        start: MediaTime::new(0, 1).unwrap(),
        end: MediaTime::new(1, 1).unwrap(),
        embedding: Some(query),
        subject_id: None,
        modality: SubjectModality::VehicleAppearance,
        embedding_quality: 1.0,
        class_id: None,
    };
    let (assigned, matches) =
        gallery.resolve_and_audit(fragment, true, MediaTime::new(1, 1).unwrap());
    assert!(assigned.subject_id.is_none());
    assert_eq!(matches[0].decision, MatchDecision::Uncertain);
    assert!(!gallery.uncertain_intervals().is_empty());
    let audit_id = gallery.audit()[0].audit_id;
    gallery
        .confirm_manual(audit_id, true, Some(SubjectId(subject.0)))
        .unwrap();
    assert_eq!(
        gallery.audit()[0].assigned_subject,
        Some(SubjectId(subject.0))
    );
    assert_eq!(gallery.audit()[0].manual_confirmation, Some(true));
}
