//! Step-2 ONNX photo search (requires weights on disk).
//!
//! ```bash
//! # Place a re-id ONNX under .sightloom-models/person_reid.onnx (NCHW RGB in → embedding out)
//! cargo run -p sightloom-host --features onnx --example onnx_photo_search -- path/to/a.rgb path/to/b.rgb
//! ```
//!
//! Without args, prints setup instructions and exits 0 if the model path resolves,
//! or 2 if weights are missing (so CI without models is not a hard fail).

use sightloom::reid::SubjectModality;
use sightloom_host::{
    DevicePreference, EmbeddingTask, FrameView, HostPipeline, ModelSpec, ModelTask, OnnxEmbedder,
    PhotoEmbeddingAdapter, PhotoView, PixelFormat, PreprocessConfig, write_cache_readme,
};
use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache = PathBuf::from(".sightloom-models");
    let _ = write_cache_readme(&cache);

    let mut spec = ModelSpec::embedder("person_reid", ModelTask::PersonReId, 512);
    spec.preprocess = PreprocessConfig::imagenet_like(128, 256);
    spec.device = DevicePreference::Cpu;
    // Prefer explicit file if present.
    let candidate = cache.join("person_reid.onnx");
    if candidate.is_file() {
        spec.local_path = Some(candidate.clone());
    }

    println!("sightloom-host step 2 — ONNX embed path");
    println!("looking for weights: {}", candidate.display());

    match OnnxEmbedder::load(spec.clone(), &cache, EmbeddingTask::PersonReId) {
        Ok(embedder) => {
            println!("loaded ONNX: {}", embedder.path().display());
            let args: Vec<String> = env::args().skip(1).collect();
            if args.len() < 2 {
                println!(
                    "pass two raw RGB files (WxH known) to run enroll/search, e.g.:\n  \
                     --features onnx --example onnx_photo_search -- alice.rgb bob.rgb\n\
                     (this example expects 128x256 RGB8 blobs for simplicity)"
                );
                return Ok(());
            }
            // Demo path using HostPipeline + reference detector still, but real embedder.
            let mut pipe = HostPipeline::new("onnx-demo")?;
            // Swap person re-id model: enroll/search use session APIs with OnnxEmbedder.
            let w = 128_u32;
            let h = 256_u32;
            let a = std::fs::read(&args[0])?;
            let b = std::fs::read(&args[1])?;
            let fa = FrameView::new(w, h, (w * 3) as usize, PixelFormat::Rgb8, &a);
            let fb = FrameView::new(w, h, (w * 3) as usize, PixelFormat::Rgb8, &b);

            let mut emb = embedder;
            let va = emb.embed_photo(&PhotoView::from_frame(fa))?;
            let vb = emb.embed_photo(&PhotoView::from_frame(fb))?;
            let sid = pipe
                .session_mut()
                .enroll_subject_photos(SubjectModality::PersonAppearance, &[va])?;
            let _ = pipe
                .session_mut()
                .enroll_subject_photos(SubjectModality::PersonAppearance, &[vb])?;
            // Search with a fresh embed of A.
            let mut emb2 = OnnxEmbedder::load(spec, &cache, EmbeddingTask::PersonReId)?;
            let hits = pipe.session_mut().search_photo_with_adapter(
                &PhotoView::from_frame(fa),
                &mut emb2,
                3,
            )?;
            println!("enrolled query subject={sid:?}");
            for h in &hits {
                println!(
                    "  hit subject={:?} score={:.3} decision={:?}",
                    h.subject_id, h.score, h.decision
                );
            }
            Ok(())
        }
        Err(e) => {
            eprintln!("ONNX weights not ready: {e}");
            eprintln!(
                "Place a float32 re-id model at:\n  {}\n\
                 Input: NCHW RGB f32 (see PreprocessConfig)\n\
                 Output: embedding vector f32",
                candidate.display()
            );
            std::process::exit(2);
        }
    }
}
