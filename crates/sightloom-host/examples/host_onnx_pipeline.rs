//! Host-facing ONNX pipeline: detect + enroll/search JPEG.
//!
//! ```bash
//! cargo run -p sightloom-host --features full --example host_onnx_pipeline
//! cargo run -p sightloom-host --features full --example host_onnx_pipeline -- \
//!     --scene .sightloom-models/bus.jpg \
//!     --enroll .sightloom-models/person_a.jpg \
//!     --other .sightloom-models/person_b.jpg
//! ```
//!
//! Missing weights → exit 2 (CI-friendly, same as `onnx_photo_search`).

use sightloom::core::{FrameStamp, MediaTime, SourceId};
use sightloom_host::{
    FrameView, HostPipeline, PixelFormat, decode_encoded_rgb, write_cache_readme,
};
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("host_onnx_pipeline: {e}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<ExitCode, Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_help();
        return Ok(ExitCode::SUCCESS);
    }

    let cache = PathBuf::from(".sightloom-models");
    let _ = write_cache_readme(&cache);

    println!("sightloom-host — HostPipeline ONNX");
    let mut pipe = match HostPipeline::from_onnx_cache("host-onnx", &cache) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ONNX weights not ready: {e}");
            eprintln!(
                "Place float32 models at:\n  {}\\person_detect.onnx   (YOLO NCHW RGB → boxes)\n  {}\\person_reid.onnx     (NCHW RGB → embedding, L2)\nOptional: face_embed.onnx",
                cache.display(),
                cache.display()
            );
            return Ok(ExitCode::from(2));
        }
    };
    println!("onnx loaded (detect + re-id)");

    if let Some(scene) = arg_value(&args, "--scene") {
        let bytes = std::fs::read(scene)?;
        let decoded = decode_encoded_rgb(&bytes)?;
        let frame = FrameView::new(
            decoded.width,
            decoded.height,
            (decoded.width as usize) * 3,
            PixelFormat::Rgb8,
            &decoded.rgb,
        );
        let stamp = FrameStamp::new(SourceId(1), 0, MediaTime::new(0, 1_000_000_000)?, None);
        let dets = pipe.detect_frame(stamp, &frame)?;
        println!("detect {}: {} box(es)", scene, dets.len());
        for (i, d) in dets.iter().take(8).enumerate() {
            let b = d.bbox();
            println!(
                "  [{i}] class={:?} score={:.3} ({:.0},{:.0})-({:.0},{:.0})",
                d.class_id(),
                d.score(),
                b.left(),
                b.top(),
                b.right(),
                b.bottom()
            );
        }
        match pipe.ingest_frame(stamp, &frame) {
            Ok(tracked) => println!("ingest_frame tracks={}", tracked.len()),
            Err(e) => eprintln!("ingest_frame skipped: {e}"),
        }
    }

    if let Some(enroll) = arg_value(&args, "--enroll") {
        let jpeg = std::fs::read(enroll)?;
        let sid = pipe.enroll_photo(&jpeg)?;
        println!("enroll_photo {enroll} → {sid:?}");
        let hits = pipe.search_photo_jpeg(&jpeg, 3)?;
        println!("search same photo:");
        for h in &hits {
            println!(
                "  subject={:?} score={:.3} decision={:?}",
                h.subject_id, h.score, h.decision
            );
        }
        if let Some(other) = arg_value(&args, "--other") {
            let b = std::fs::read(other)?;
            let other_hits = pipe.search_photo_jpeg(&b, 3)?;
            println!("search other {other}:");
            for h in &other_hits {
                println!(
                    "  subject={:?} score={:.3} decision={:?}",
                    h.subject_id, h.score, h.decision
                );
            }
        }
    } else {
        println!("pass --scene photo.jpg and/or --enroll a.jpg [--other b.jpg] to run");
    }

    Ok(ExitCode::SUCCESS)
}

fn arg_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].as_str())
}

fn print_help() {
    println!(
        "usage: host_onnx_pipeline [--scene photo.jpg] [--enroll a.jpg] [--other b.jpg]\n\
         looks for .sightloom-models/person_detect.onnx + person_reid.onnx"
    );
}
