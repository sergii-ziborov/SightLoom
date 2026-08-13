//! Thin host sketch: fake detector → seed click → track → export (no render, no pixels).
//!
//! Run: `cargo run -p sightloom --example host_sketch`
//!
//! This is **not** a render or intelligence product. It only shows how a host feeds
//! `SightLoom` and consumes spans / reels / uncertain intervals as data.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::doc_markdown
)]

use sightloom::IndexSession;
use sightloom_core::{FrameStamp, MediaTime, Rect, SourceId, ZoneId};
use sightloom_index::{SourceEntry, SubjectQuery};
use sightloom_tracking::ByteTrackConfig;

fn main() {
    let config = ByteTrackConfig {
        track_high_thresh: 0.5,
        track_activation_thresh: 0.5,
        track_low_thresh: 0.1,
        match_thresh: 0.3,
        max_time_lost: 30,
        class_aware: false,
    };
    let mut session = IndexSession::new("host-sketch", config).expect("session");
    session.add_source(SourceEntry {
        source_id: 1,
        uri: "file://demo.mp4".into(),
        hash: None,
    });

    // Host UI: user clicked a person at frame 0.
    let stamp0 = FrameStamp::new(SourceId(1), 0, MediaTime::new(0, 30).unwrap(), None);
    let click = Rect::new(40.0, 40.0, 80.0, 120.0).unwrap();
    let seed = session.seed_click(stamp0, click, 0.93, None).expect("seed");
    println!(
        "seeded subject={} track={} uid={}",
        seed.subject_id.0, seed.track_id.0, seed.track_uid.0
    );

    // Host detector: subsequent frames as Detection list (fake).
    for frame in 1..=5 {
        let stamp = FrameStamp::new(
            SourceId(1),
            frame,
            MediaTime::new(i64::try_from(frame).unwrap(), 30).unwrap(),
            None,
        );
        let dx = frame as f32 * 2.0;
        let det = sightloom_core::Detection::new(
            Rect::new(40.0 + dx, 40.0, 80.0 + dx, 120.0).unwrap(),
            0.9,
            Some(sightloom_core::ClassId(0)),
            None,
        )
        .unwrap();
        let tracked = session.ingest_detections(stamp, &[det]).expect("track");
        for item in &tracked {
            // Keep subject map when tracker reuses the local id.
            if item.track_key.local_track_id == seed.track_id {
                session.assign_subject(item.track_key, seed.subject_id);
            }
        }
    }

    let spans_json = session.export_track_spans_json().expect("spans");
    println!("track spans (for host MaskTimeline):\n{spans_json}");

    let uncertain = session
        .export_uncertain_intervals_json()
        .expect("uncertain");
    println!("uncertain intervals:\n{uncertain}");

    let reel = session.build_subject_reel(seed.subject_id, 1_000_000_000);
    println!(
        "evidence reel id={} segments={} span_ns={:?}",
        reel.reel_id.0,
        reel.len(),
        reel.span_ns()
    );

    let hits = session.query_subjects(&SubjectQuery::new().seen_on(SourceId(1)));
    println!("subjects on source 1: {}", hits.len());

    // Zone predicates need zone_stays filled by host analytics; empty is fine here.
    let _ = ZoneId(1);
    let _ = session.mine_and_store_patterns();

    let package = std::env::temp_dir().join("sightloom-host-sketch-package");
    let _ = std::fs::remove_dir_all(&package);
    session.save_package(&package).expect("package");
    println!("wrote VisionIndex package to {}", package.display());
}
