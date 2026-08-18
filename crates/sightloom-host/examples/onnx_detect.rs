//! ONNX object detection (`YOLOv5` / `YOLOv8` / `YOLOv11` / flat `N×6`).
//!
//! ```bash
//! # Place a detector ONNX under .sightloom-models/person_detect.onnx
//! cargo run -p sightloom-host --features onnx --example onnx_detect
//!
//! # Optional: path to weights and/or a JPEG/PNG (needs image-decode)
//! cargo run -p sightloom-host --features full --example onnx_detect -- \
//!     --model .sightloom-models/yolov8n.onnx --image photo.jpg
//! ```
//!
//! Without weights the example prints setup help and exits 2 (CI-friendly).
//! Without an image it builds a synthetic RGB frame so the load path is
//! still exercised.

use sightloom::core::{FrameStamp, MediaTime, SourceId};
use sightloom::{DetectorAdapter, FrameView, PixelFormat};
use sightloom_host::{
    ModelSpec, ModelTask, OnnxDetector, PreprocessConfig, decode_encoded_rgb, write_cache_readme,
};
use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const COCO80: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("onnx_detect: {e}");
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

    let model_path = arg_value(&args, "--model").map(PathBuf::from).or_else(|| {
        first_existing(&[
            cache.join("person_detect.onnx"),
            cache.join("yolov8n.onnx"),
            cache.join("yolov5n.onnx"),
        ])
    });
    let image_path = arg_value(&args, "--image").map(PathBuf::from);

    println!("sightloom-host — ONNX detect");
    let Some(model_path) = model_path else {
        println!("weights: (missing)");
        print_help();
        eprintln!(
            "Place a float32 YOLO ONNX at:\n  {}\n\
             Input: NCHW RGB f32 640×640 (PreprocessConfig::yolo_detect)\n\
             Output: [1, 4+C, N] (YOLOv8) / [1, N, 5+C] (YOLOv5) / N×6 xyxy",
            cache.join("person_detect.onnx").display()
        );
        return Ok(ExitCode::from(2));
    };
    println!("weights: {}", model_path.display());

    let mut spec = ModelSpec::detector("person_detect", ModelTask::PersonDetect);
    spec.preprocess = PreprocessConfig::yolo_detect(640, 640);
    spec.local_path = Some(model_path);

    let mut detector = match OnnxDetector::load(spec, &cache) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("failed to load ONNX: {e}");
            return Ok(ExitCode::from(2));
        }
    };
    println!("loaded: {}", detector.path().display());

    let (rgb, width, height, label) = load_or_synth(image_path.as_deref())?;
    println!("frame: {label} {width}x{height}");

    let frame = FrameView::new(width, height, (width as usize) * 3, PixelFormat::Rgb8, &rgb);
    let stamp = FrameStamp::new(
        SourceId(1),
        0,
        MediaTime::new(0, 1_000_000_000).unwrap(),
        None,
    );
    let dets = detector.detect(stamp, &frame)?;
    println!("detections: {}", dets.len());
    for (i, d) in dets.iter().take(20).enumerate() {
        let b = d.bbox();
        let class = d.class_id().map_or(0, |c| c.0);
        let name = COCO80.get(class as usize).copied().unwrap_or("class");
        println!(
            "  [{i}] {name}#{class} score={:.3} box=({:.1},{:.1})-({:.1},{:.1})",
            d.score(),
            b.left(),
            b.top(),
            b.right(),
            b.bottom()
        );
    }
    if dets.len() > 20 {
        println!("  … {} more", dets.len() - 20);
    }
    Ok(ExitCode::SUCCESS)
}

type LoadedFrame = (Vec<u8>, u32, u32, String);

fn load_or_synth(image: Option<&Path>) -> Result<LoadedFrame, Box<dyn std::error::Error>> {
    if let Some(path) = image {
        let bytes = std::fs::read(path)?;
        match decode_encoded_rgb(&bytes) {
            Ok(decoded) => Ok((
                decoded.rgb,
                decoded.width,
                decoded.height,
                path.display().to_string(),
            )),
            Err(e) => Err(format!(
                "could not decode {}: {e} (enable --features full / image-decode for JPEG/PNG)",
                path.display()
            )
            .into()),
        }
    } else {
        Ok((synth_scene(640, 480), 640, 480, "synthetic".into()))
    }
}

fn synth_scene(width: u32, height: u32) -> Vec<u8> {
    let mut rgb = vec![30_u8; (width * height * 3) as usize];
    // Person-like vertical blob + a "car" horizontal blob so a real YOLO has
    // something to look at; without weights this is only a load-path smoke.
    fill_rect(&mut rgb, width, height, 220, 80, 360, 400, [40, 80, 200]);
    fill_rect(&mut rgb, width, height, 40, 300, 200, 420, [200, 40, 40]);
    rgb
}

#[allow(clippy::too_many_arguments)]
fn fill_rect(
    rgb: &mut [u8],
    width: u32,
    height: u32,
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
    color: [u8; 3],
) {
    let right = right.min(width);
    let bottom = bottom.min(height);
    for y in top..bottom {
        for x in left..right {
            let i = ((y * width + x) * 3) as usize;
            rgb[i] = color[0];
            rgb[i + 1] = color[1];
            rgb[i + 2] = color[2];
        }
    }
}

fn arg_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].as_str())
}

fn first_existing(paths: &[PathBuf]) -> Option<PathBuf> {
    paths.iter().find(|p| p.is_file()).cloned()
}

fn print_help() {
    println!(
        "usage: onnx_detect [--model PATH.onnx] [--image photo.jpg]\n\
         looks in .sightloom-models/{{person_detect,yolov8n,yolov5n}}.onnx when --model is omitted"
    );
}
