//! Moore-neighborhood outer contour tracing for dense binary masks.

use super::{DenseMask, MaskError};
use sightloom_core::Point;

/// Clockwise Moore neighborhood offsets (8-connected), starting east of the
/// current pixel and walking counter-clockwise (standard border following).
const NEIGH: [(i32, i32); 8] = [
    (1, 0),   // E
    (1, -1),  // NE
    (0, -1),  // N
    (-1, -1), // NW
    (-1, 0),  // W
    (-1, 1),  // SW
    (0, 1),   // S
    (1, 1),   // SE
];

/// Traces the outer contour of the first foreground component found.
///
/// Writes ordered boundary pixel centers into `output` and returns the count.
/// Uses Moore neighborhood tracing; stops when returning to the start with the
/// same entry direction, or when `output` is full.
///
/// # Errors
///
/// - [`MaskError::EmptyPolygon`] — no foreground
/// - [`MaskError::InsufficientCapacity`] — `output` empty
/// - [`MaskError::NonFinite`] — point construction failure
pub fn dense_to_contour(
    mask: DenseMask<'_>,
    output: &mut [Point],
) -> Result<usize, MaskError> {
    if output.is_empty() {
        return Err(MaskError::InsufficientCapacity);
    }
    let start = find_start(mask).ok_or(MaskError::EmptyPolygon)?;
    // Enter from the left of start (west → east).
    let mut prev_dir = 4_usize; // came from W looking for E
    let mut cur = start;
    let mut count = 0_usize;
    let max_steps = (mask.width() as usize)
        .saturating_mul(mask.height() as usize)
        .saturating_mul(8)
        .max(8);

    for _ in 0..max_steps {
        if count >= output.len() {
            return Err(MaskError::InsufficientCapacity);
        }
        output[count] = Point::new(cur.0 as f32 + 0.5, cur.1 as f32 + 0.5)
            .map_err(|_| MaskError::NonFinite)?;
        count += 1;

        // Search starts at (prev_dir + 6) % 8 = back-right of entry (Moore rule).
        let begin = (prev_dir + 6) % 8;
        let mut found = None;
        for k in 0..8 {
            let d = (begin + k) % 8;
            let nx = cur.0 as i32 + NEIGH[d].0;
            let ny = cur.1 as i32 + NEIGH[d].1;
            if nx < 0 || ny < 0 {
                continue;
            }
            let (ux, uy) = (nx as u32, ny as u32);
            if mask.is_set(ux, uy) {
                found = Some(((ux, uy), d));
                break;
            }
        }
        let Some((next, dir)) = found else {
            // Isolated pixel.
            break;
        };
        // Stop condition: back at start with same direction after at least 2 points.
        if next == start && dir == prev_dir && count > 2 {
            break;
        }
        // Also stop if we closed the loop by returning to start (relaxed).
        if next == start && count > 3 {
            break;
        }
        cur = next;
        prev_dir = dir;
    }

    if count < 3 {
        // Degenerate: fall back to bbox corners if possible.
        return dense_contour_bbox_fallback(mask, output);
    }
    Ok(count)
}

/// All external contours (one per 4-connected component, outer only).
///
/// Each contour is a run of points; contours are separated by writing them
/// sequentially into `output` and returning lengths in `lengths`.
///
/// # Errors
///
/// Capacity / empty errors.
#[cfg(feature = "alloc")]
pub fn dense_to_contours(
    mask: DenseMask<'_>,
    output: &mut [Point],
    lengths: &mut alloc::vec::Vec<usize>,
) -> Result<usize, MaskError> {
    lengths.clear();
    let w = mask.width() as usize;
    let h = mask.height() as usize;
    let mut visited = alloc::vec![false; w.saturating_mul(h)];
    let mut total = 0_usize;
    let mut scratch = alloc::vec![Point::new(0.0, 0.0).map_err(|_| MaskError::NonFinite)?; 4096];

    for y in 0..mask.height() {
        for x in 0..mask.width() {
            if !mask.is_set(x, y) {
                continue;
            }
            let idx = (y as usize) * w + (x as usize);
            if visited[idx] {
                continue;
            }
            // Only start on left-edge of component (background to the left).
            let left_bg = x == 0 || !mask.is_set(x - 1, y);
            if !left_bg {
                continue;
            }
            match dense_to_contour_from(mask, (x, y), &mut scratch) {
                Ok(n) if n >= 3 => {
                    if total + n > output.len() {
                        return Err(MaskError::InsufficientCapacity);
                    }
                    output[total..total + n].copy_from_slice(&scratch[..n]);
                    // Mark visited along contour
                    for p in &scratch[..n] {
                        let px = p.x() as u32;
                        let py = p.y() as u32;
                        if px < mask.width() && py < mask.height() {
                            visited[(py as usize) * w + (px as usize)] = true;
                        }
                    }
                    lengths.push(n);
                    total += n;
                }
                _ => {
                    visited[idx] = true;
                }
            }
        }
    }
    if total == 0 {
        return Err(MaskError::EmptyPolygon);
    }
    Ok(total)
}

fn dense_to_contour_from(
    mask: DenseMask<'_>,
    start: (u32, u32),
    output: &mut [Point],
) -> Result<usize, MaskError> {
    if output.is_empty() {
        return Err(MaskError::InsufficientCapacity);
    }
    let mut prev_dir = 4_usize;
    let mut cur = start;
    let mut count = 0_usize;
    let max_steps = output.len().saturating_mul(2).max(8);
    for _ in 0..max_steps {
        if count >= output.len() {
            return Ok(count);
        }
        output[count] = Point::new(cur.0 as f32 + 0.5, cur.1 as f32 + 0.5)
            .map_err(|_| MaskError::NonFinite)?;
        count += 1;
        let begin = (prev_dir + 6) % 8;
        let mut found = None;
        for k in 0..8 {
            let d = (begin + k) % 8;
            let nx = cur.0 as i32 + NEIGH[d].0;
            let ny = cur.1 as i32 + NEIGH[d].1;
            if nx < 0 || ny < 0 {
                continue;
            }
            let (ux, uy) = (nx as u32, ny as u32);
            if mask.is_set(ux, uy) {
                found = Some(((ux, uy), d));
                break;
            }
        }
        let Some((next, dir)) = found else {
            break;
        };
        if next == start && count > 3 {
            break;
        }
        cur = next;
        prev_dir = dir;
    }
    Ok(count)
}

fn find_start(mask: DenseMask<'_>) -> Option<(u32, u32)> {
    for y in 0..mask.height() {
        for x in 0..mask.width() {
            if mask.is_set(x, y) {
                return Some((x, y));
            }
        }
    }
    None
}

fn dense_contour_bbox_fallback(
    mask: DenseMask<'_>,
    output: &mut [Point],
) -> Result<usize, MaskError> {
    if output.len() < 4 {
        return Err(MaskError::InsufficientCapacity);
    }
    let bbox = mask.bbox().ok_or(MaskError::EmptyPolygon)?;
    output[0] = Point::new(bbox.left(), bbox.top()).map_err(|_| MaskError::NonFinite)?;
    output[1] = Point::new(bbox.right(), bbox.top()).map_err(|_| MaskError::NonFinite)?;
    output[2] = Point::new(bbox.right(), bbox.bottom()).map_err(|_| MaskError::NonFinite)?;
    output[3] = Point::new(bbox.left(), bbox.bottom()).map_err(|_| MaskError::NonFinite)?;
    Ok(4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DenseMask;

    #[test]
    fn traces_filled_rectangle() {
        // 5x5 with 3x3 filled block
        let mut data = [0_u8; 25];
        for y in 1..4 {
            for x in 1..4 {
                data[y * 5 + x] = 1;
            }
        }
        let mask = DenseMask::new(5, 5, &data).unwrap();
        let mut out = [Point::new(0.0, 0.0).unwrap(); 64];
        let n = dense_to_contour(mask, &mut out).unwrap();
        assert!(n >= 4, "n={n}");
        // All contour points should be near the filled region.
        for p in &out[..n] {
            assert!(p.x() >= 0.5 && p.x() <= 4.5);
            assert!(p.y() >= 0.5 && p.y() <= 4.5);
        }
    }
}
