//! Oriented rectangles and approximate oriented box overlap / NMS.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use sightloom_core::{CoreError, Point, Rect};

/// A center-based oriented rectangle (degrees clockwise from +X).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrientedRect {
    /// Center of the box.
    pub center: Point,
    /// Full width along the local X axis.
    pub width: f32,
    /// Full height along the local Y axis.
    pub height: f32,
    /// Rotation in degrees.
    pub angle_deg: f32,
}

impl OrientedRect {
    /// Creates an oriented rect when all numbers are finite and sizes non-negative.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::NonFinite`] for non-finite inputs.
    pub fn new(center: Point, width: f32, height: f32, angle_deg: f32) -> Result<Self, CoreError> {
        if !width.is_finite()
            || !height.is_finite()
            || !angle_deg.is_finite()
            || width < 0.0
            || height < 0.0
        {
            return Err(CoreError::NonFinite);
        }
        Ok(Self {
            center,
            width,
            height,
            angle_deg,
        })
    }

    /// Four corners in image coordinates (TL, TR, BR, BL in local frame).
    #[must_use]
    pub fn corners(self) -> [Point; 4] {
        let rad = self.angle_deg * (core::f32::consts::PI / 180.0);
        let (s, c) = sin_cos_approx(rad);
        let hw = self.width * 0.5;
        let hh = self.height * 0.5;
        let local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)];
        let mut out = [self.center; 4];
        for (i, (lx, ly)) in local.iter().enumerate() {
            let x = self.center.x() + lx * c - ly * s;
            let y = self.center.y() + lx * s + ly * c;
            out[i] = Point::new(x, y).unwrap_or(self.center);
        }
        out
    }

    /// Axis-aligned bounding box of this oriented rect.
    ///
    /// Falls back to a 1×1 box at the center if construction fails (should not
    /// happen for finite corners).
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn axis_aligned_bbox(self) -> Rect {
        let corners = self.corners();
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for p in corners {
            min_x = min_x.min(p.x());
            min_y = min_y.min(p.y());
            max_x = max_x.max(p.x());
            max_y = max_y.max(p.y());
        }
        if let Ok(r) = Rect::new(min_x, min_y, max_x, max_y) {
            r
        } else if let Ok(r) = Rect::new(
            self.center.x(),
            self.center.y(),
            self.center.x() + 1.0,
            self.center.y() + 1.0,
        ) {
            r
        } else {
            // Finite unit box at origin as absolute last resort.
            Rect::new(0.0, 0.0, 1.0, 1.0).unwrap_or_else(|_| {
                // Point/Rect construction cannot fail for these literals under current API.
                unreachable!("unit rect must be valid")
            })
        }
    }
}

/// Approximate oriented box overlap via AABB of each OBB (fast host prefilter).
///
/// Not exact OBB intersection; suitable for soft ranking / NMS gates.
#[must_use]
pub fn oriented_aabb_iou(a: OrientedRect, b: OrientedRect) -> f32 {
    sightloom_core::iou(a.axis_aligned_bbox(), b.axis_aligned_bbox())
}

/// Oriented detection for NMS (score + OBB + optional class).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrientedDetection {
    /// Box.
    pub rect: OrientedRect,
    /// Score.
    pub score: f32,
    /// Class.
    pub class_id: Option<u16>,
}

/// Hard NMS on oriented detections using AABB-IoU approximation.
///
/// Compacts kept detections to the front; returns kept count.
#[cfg(feature = "alloc")]
pub fn oriented_nms_aabb(dets: &mut [OrientedDetection], threshold: f32) -> usize {
    if dets.is_empty() || !threshold.is_finite() {
        return 0;
    }
    let mut order: Vec<usize> = (0..dets.len()).collect();
    order.sort_by(|&a, &b| {
        dets[b]
            .score
            .partial_cmp(&dets[a].score)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    let mut suppressed = vec![false; dets.len()];
    for (pi, &i) in order.iter().enumerate() {
        if suppressed[i] {
            continue;
        }
        for &j in order.iter().skip(pi + 1) {
            if suppressed[j] {
                continue;
            }
            if dets[i].class_id.is_some()
                && dets[j].class_id.is_some()
                && dets[i].class_id != dets[j].class_id
            {
                continue;
            }
            if oriented_aabb_iou(dets[i].rect, dets[j].rect) > threshold {
                suppressed[j] = true;
            }
        }
    }
    let mut kept = 0;
    for i in 0..dets.len() {
        if !suppressed[i] {
            dets[kept] = dets[i];
            kept += 1;
        }
    }
    kept
}

/// Portable sin/cos for `no_std` (Taylor series, angle reduced to `[-pi, pi]`).
fn sin_cos_approx(rad: f32) -> (f32, f32) {
    let mut x = rad;
    // wrap roughly into [-pi, pi]
    let two_pi = 2.0 * core::f32::consts::PI;
    while x > core::f32::consts::PI {
        x -= two_pi;
    }
    while x < -core::f32::consts::PI {
        x += two_pi;
    }
    // sin Taylor
    let x2 = x * x;
    let sin = x * (1.0 - x2 / 6.0 + x2 * x2 / 120.0);
    let cos = 1.0 - x2 / 2.0 + x2 * x2 / 24.0;
    (sin, cos)
}
