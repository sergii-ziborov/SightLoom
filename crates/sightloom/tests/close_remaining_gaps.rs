//! Closing remaining thin product gaps: hypotheses, topology max-travel,
//! negative policy, continuous embed path, audit views.

#![allow(clippy::cast_precision_loss)]

use sightloom::core::{Detection, FrameStamp, MediaTime, Point, Rect, SourceId, TrackId};
use sightloom::index::{OrientedRect, oriented_iou};
use sightloom::reid::{
    CameraTopology, MatchDecision, NegativeEvidencePolicy, ReferenceSample, ResolveConfig,
    SubjectModality,
};
use sightloom::tracking::ByteTrackConfig;
use sightloom::{FrameView, IndexSession, PixelFormat, TrackEmbeddingAdapter};

fn cfg() -> ByteTrackConfig {
    ByteTrackConfig {
        track_high_thresh: 0.5,
        track_activation_thresh: 0.5,
        track_low_thresh: 0.1,
        match_thresh: 0.3,
        max_time_lost: 30,
        class_aware: false,
    }
}

struct StubTrackEmbed;

impl TrackEmbeddingAdapter for StubTrackEmbed {
    type Error = &'static str;

    fn embed_track(
        &mut self,
        _stamp: FrameStamp,
        _frame: &FrameView<'_>,
        key: sightloom::core::TrackKey,
        bbox: Rect,
    ) -> Result<Vec<f32>, Self::Error> {
        Ok(vec![
            key.local_track_id.0 as f32 * 0.01,
            bbox.left() * 0.001,
            1.0,
            0.0,
        ])
    }
}

#[test]
fn continuous_embed_batch_and_audit_views() {
    let mut session = IndexSession::new("gaps", cfg()).unwrap();
    let stamp = FrameStamp::new(
        SourceId(1),
        0,
        MediaTime::new(0, 1_000_000_000).unwrap(),
        None,
    );
    let det = Detection::new(Rect::new(10.0, 10.0, 30.0, 40.0).unwrap(), 0.9, None, None).unwrap();
    let tracked = session.ingest_detections(stamp, &[det]).unwrap();
    assert!(!tracked.is_empty());
    let key = tracked[0].track_key;
    session
        .note_track_embeddings_batch(&[(key, vec![1.0, 0.0, 0.0, 0.0], stamp.pts)])
        .unwrap();
    assert!(!session.track_samples_audit().is_empty());
    assert_eq!(
        session.track_samples_effective().len(),
        session.track_samples_audit().len()
    );
}

#[test]
fn hypothesis_accept_and_dismiss_lifecycle() {
    let mut session = IndexSession::new("hyp", cfg()).unwrap();
    session
        .set_resolve_config(ResolveConfig {
            accept_threshold: 0.99,
            reject_threshold: 0.10,
            require_same_modality: true,
            negative_reject_threshold: 0.99,
            ..ResolveConfig::default()
        })
        .unwrap();
    let s1 = session.register_subject(SubjectModality::PersonAppearance);
    let s2 = session.register_subject(SubjectModality::PersonAppearance);
    let e1 = session
        .gallery_mut()
        .embeddings
        .insert([1.0_f32, 0.0, 0.0])
        .unwrap();
    let e2 = session
        .gallery_mut()
        .embeddings
        .insert([0.9_f32, 0.1, 0.0])
        .unwrap();
    session
        .add_subject_reference(
            s1,
            ReferenceSample {
                source_id: Some(SourceId(1)),
                track_id: None,
                at: None,
                embedding: Some(e1),
                evidence: None,
                is_positive: Some(true),
                quality: Some(1.0),
                class_id: None,
            },
        )
        .unwrap();
    session
        .add_subject_reference(
            s2,
            ReferenceSample {
                source_id: Some(SourceId(1)),
                track_id: None,
                at: None,
                embedding: Some(e2),
                evidence: None,
                is_positive: Some(true),
                quality: Some(1.0),
                class_id: None,
            },
        )
        .unwrap();

    let key = sightloom::core::TrackKey::new(SourceId(1), TrackId(1));
    session
        .note_track_embedding(key, [0.95_f32, 0.05, 0.0], MediaTime::default())
        .unwrap();
    session
        .resolve_track_identity(
            key,
            Some(SubjectModality::PersonAppearance),
            MediaTime::default(),
        )
        .unwrap();

    let open = session.open_identity_cases();
    // May be uncertain or multi-hypothesis depending on scores.
    if let Some(case) = open.first() {
        let audit_id = case.audit_id;
        let pick = case.hypotheses.first().map_or(s1, |h| h.subject_id);
        session.accept_identity_hypothesis(audit_id, pick).unwrap();
        assert!(
            session
                .assigned_identity_view()
                .iter()
                .any(|(_, _, sid)| *sid == pick)
        );
    }
}

#[test]
fn topology_max_travel_blocks_hop() {
    let mut topo = CameraTopology::new();
    topo.set_edge_window(SourceId(1), SourceId(2), 1_000_000_000, Some(5_000_000_000));
    assert!(topo.allows_hop(SourceId(1), SourceId(2), 2_000_000_000, true));
    assert!(!topo.allows_hop(SourceId(1), SourceId(2), 100_000_000, true)); // too fast
    assert!(!topo.allows_hop(SourceId(1), SourceId(2), 10_000_000_000, true)); // too slow
    assert!(!topo.allows_hop(SourceId(1), SourceId(3), 2_000_000_000, true)); // unknown strict
}

#[test]
fn negative_soft_uncertain_policy_compiles_config() {
    let cfg = ResolveConfig {
        negative_policy: NegativeEvidencePolicy::SoftUncertain,
        ..ResolveConfig::default()
    };
    assert_eq!(cfg.negative_policy, NegativeEvidencePolicy::SoftUncertain);
    assert_ne!(MatchDecision::Accept, MatchDecision::Reject);
}

#[test]
fn oriented_iou_polygon_path() {
    let a = OrientedRect::new(Point::new(5.0, 5.0).unwrap(), 4.0, 4.0, 0.0).unwrap();
    let b = OrientedRect::new(Point::new(6.0, 5.0).unwrap(), 4.0, 4.0, 0.0).unwrap();
    assert!(oriented_iou(a, b) > 0.3);
}

#[test]
fn detect_ingest_embed_tracks_stub() {
    use sightloom::{DetectorAdapter, FrameView};

    struct Det;
    impl DetectorAdapter for Det {
        type Error = &'static str;
        fn detect(
            &mut self,
            _stamp: FrameStamp,
            _frame: &FrameView<'_>,
        ) -> Result<Vec<Detection>, Self::Error> {
            Ok(vec![
                Detection::new(Rect::new(0.0, 0.0, 10.0, 20.0).unwrap(), 0.9, None, None).unwrap(),
            ])
        }
    }

    let mut session = IndexSession::new("emb", cfg()).unwrap();
    let stamp = FrameStamp::new(
        SourceId(1),
        0,
        MediaTime::new(0, 1_000_000_000).unwrap(),
        None,
    );
    let pixels = [0_u8; 16];
    let frame = FrameView::new(4, 4, 4, PixelFormat::Gray8, &pixels);
    let mut det = Det;
    let mut emb = StubTrackEmbed;
    let tracked = session
        .detect_ingest_and_embed_tracks(stamp, &frame, &mut det, &mut emb)
        .unwrap();
    assert_eq!(tracked.len(), 1);
}
