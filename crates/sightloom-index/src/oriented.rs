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

/// Exact-ish oriented box `IoU` via convex polygon intersection (Sutherland–Hodgman).
///
/// Both OBBs are convex quads; intersection area / union is computed in plane.
/// More accurate than [`oriented_aabb_iou`] for rotated boxes.
#[cfg(feature = "alloc")]
#[must_use]
pub fn oriented_iou(a: OrientedRect, b: OrientedRect) -> f32 {
    let poly_a = a.corners();
    let poly_b = b.corners();
    let a_pts: Vec<(f32, f32)> = poly_a.iter().map(|p| (p.x(), p.y())).collect();
    let b_pts: Vec<(f32, f32)> = poly_b.iter().map(|p| (p.x(), p.y())).collect();
    let inter = convex_intersection(&a_pts, &b_pts);
    let inter_area = polygon_area(&inter);
    if inter_area <= 0.0 {
        return 0.0;
    }
    let area_a = a.width * a.height;
    let area_b = b.width * b.height;
    let union = area_a + area_b - inter_area;
    if union <= 1e-12 {
        return 0.0;
    }
    (inter_area / union).clamp(0.0, 1.0)
}

#[cfg(feature = "alloc")]
fn polygon_area(pts: &[(f32, f32)]) -> f32 {
    if pts.len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0_f32;
    for i in 0..pts.len() {
        let (x1, y1) = pts[i];
        let (x2, y2) = pts[(i + 1) % pts.len()];
        sum += x1 * y2 - x2 * y1;
    }
    (sum * 0.5).abs()
}

#[cfg(feature = "alloc")]
fn convex_intersection(subject: &[(f32, f32)], clip: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if subject.is_empty() || clip.len() < 3 {
        return Vec::new();
    }
    let mut output = subject.to_vec();
    for i in 0..clip.len() {
        if output.is_empty() {
            break;
        }
        let a = clip[i];
        let b = clip[(i + 1) % clip.len()];
        let input = output;
        output = Vec::new();
        if input.is_empty() {
            break;
        }
        let mut s = input[input.len() - 1];
        for &e in &input {
            if inside(e, a, b) {
                if !inside(s, a, b)
                    && let Some(x) = line_intersect(s, e, a, b)
                {
                    output.push(x);
                }
                output.push(e);
            } else if inside(s, a, b)
                && let Some(x) = line_intersect(s, e, a, b)
            {
                output.push(x);
            }
            s = e;
        }
    }
    output
}

#[cfg(feature = "alloc")]
fn inside(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> bool {
    // Left of directed edge a->b (CCW clip poly).
    (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0) >= -1e-6
}

#[cfg(feature = "alloc")]
fn line_intersect(
    start: (f32, f32),
    end: (f32, f32),
    clip_a: (f32, f32),
    clip_b: (f32, f32),
) -> Option<(f32, f32)> {
    let d1x = end.0 - start.0;
    let d1y = end.1 - start.1;
    let d2x = clip_b.0 - clip_a.0;
    let d2y = clip_b.1 - clip_a.1;
    let den = d1x * d2y - d1y * d2x;
    if den.abs() < 1e-12 {
        return None;
    }
    let t = ((clip_a.0 - start.0) * d2y - (clip_a.1 - start.1) * d2x) / den;
    Some((start.0 + t * d1x, start.1 + t * d1y))
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

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    #[test]
    fn oriented_iou_high_for_aligned_overlap() {
        let a = OrientedRect::new(Point::new(10.0, 10.0).unwrap(), 8.0, 4.0, 0.0).unwrap();
        let b = OrientedRect::new(Point::new(11.0, 10.0).unwrap(), 8.0, 4.0, 0.0).unwrap();
        let iou = oriented_iou(a, b);
        assert!(iou > 0.5, "iou={iou}");
        let aabb = oriented_aabb_iou(a, b);
        assert!((iou - aabb).abs() < 0.15, "iou={iou} aabb={aabb}");
    }

    #[test]
    fn oriented_iou_zero_when_disjoint() {
        let a = OrientedRect::new(Point::new(0.0, 0.0).unwrap(), 2.0, 2.0, 0.0).unwrap();
        let b = OrientedRect::new(Point::new(50.0, 50.0).unwrap(), 2.0, 2.0, 0.0).unwrap();
        assert!(oriented_iou(a, b) < 1e-6);
    }
}
