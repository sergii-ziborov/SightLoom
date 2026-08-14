//! Step-1 killer path demo: enroll two people, search by photo.
//!
//! ```bash
//! cargo run -p sightloom-host --example photo_to_subject
//! ```
//!
//! Uses **reference** (fake) embedders — real ONNX lands in a later step.

use sightloom_host::{
    FrameView, HostBundleConfig, HostPipeline, PhotoView, PixelFormat, write_cache_readme,
};

fn solid(width: u32, height: u32, red: u8, green: u8, blue: u8) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for _ in 0..(width * height) {
        pixels.extend_from_slice(&[red, green, blue]);
    }
    pixels
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = HostBundleConfig::default();
    let _ = write_cache_readme(&cfg.cache_dir);

    println!("sightloom-host step 1 — photo → embed → rank");
    println!("cache dir: {}", cfg.cache_dir.display());
    println!("models (reference, no weights):");
    for spec in cfg.all_specs() {
        println!("  - {} [{}]", spec.id, spec.task.as_str());
    }

    let mut pipe = HostPipeline::new("photo-demo")?;

    let alice = solid(48, 96, 220, 60, 60);
    let bob = solid(48, 96, 60, 60, 220);
    let fa = FrameView::new(48, 96, 48 * 3, PixelFormat::Rgb8, &alice);
    let fb = FrameView::new(48, 96, 48 * 3, PixelFormat::Rgb8, &bob);

    let sid_a = pipe.enroll_photos(&[PhotoView::from_frame(fa)], false)?;
    let sid_b = pipe.enroll_photos(&[PhotoView::from_frame(fb)], false)?;
    println!("enrolled Alice={sid_a:?} Bob={sid_b:?}");

    let hits = pipe.search_photo(&PhotoView::from_frame(fa), false, 2)?;
    println!("query Alice-like top hits:");
    for h in &hits {
        println!(
            "  subject={:?} score={:.3} decision={:?}",
            h.subject_id, h.score, h.decision
        );
    }
    assert_eq!(hits[0].subject_id, sid_a, "Alice should rank first");

    println!("ok — end-to-end photo path wired (reference embedders only)");
    Ok(())
}
