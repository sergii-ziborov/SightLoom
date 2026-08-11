//! Rectangle overlap metrics used by matching and suppression algorithms.

use crate::Rect;

/// Returns the area shared by two rectangles.
#[must_use]
pub fn intersection_area(a: Rect, b: Rect) -> f32 {
    a.intersection(b).area()
}

/// Returns intersection over union (`IoU`) for two rectangles.
///
/// The result is zero when the union has zero area.
#[must_use]
pub fn iou(a: Rect, b: Rect) -> f32 {
    let intersection = intersection_area(a, b);
    let union = a.area() + b.area() - intersection;

    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

/// Returns intersection over the smaller area (`IoS`) for two rectangles.
///
/// The result is zero when either rectangle has zero area.
#[must_use]
pub fn ios(a: Rect, b: Rect) -> f32 {
    let smaller = a.area().min(b.area());

    if smaller > 0.0 {
        intersection_area(a, b) / smaller
    } else {
        0.0
    }
}
