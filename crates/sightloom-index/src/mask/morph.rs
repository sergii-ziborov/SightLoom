//! Binary morphology helpers for dense masks.

use super::{DenseMask, MaskError};

/// Dilates a dense mask with a square odd kernel of `radius` pixels.
///
/// # Errors
///
/// Returns [`MaskError::InsufficientCapacity`] when `output` is shorter than
/// the source, and [`MaskError::EmptyDimensions`] when radius is zero.
pub fn dilate(mask: DenseMask<'_>, radius: u32, output: &mut [u8]) -> Result<(), MaskError> {
    morph(mask, radius, output, true)
}

/// Erodes a dense mask with a square odd kernel of `radius` pixels.
///
/// # Errors
///
/// Returns capacity or radius errors like [`dilate`].
pub fn erode(mask: DenseMask<'_>, radius: u32, output: &mut [u8]) -> Result<(), MaskError> {
    morph(mask, radius, output, false)
}

fn morph(
    mask: DenseMask<'_>,
    radius: u32,
    output: &mut [u8],
    dilate_mode: bool,
) -> Result<(), MaskError> {
    if radius == 0 {
        return Err(MaskError::EmptyDimensions);
    }
    let needed = mask.data().len();
    if output.len() < needed {
        return Err(MaskError::InsufficientCapacity);
    }
    let width = mask.width();
    let height = mask.height();
    let r = radius as i32;

    for y in 0..height {
        for x in 0..width {
            let mut value = !dilate_mode;
            'kernel: for dy in -r..=r {
                for dx in -r..=r {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                        if !dilate_mode {
                            value = false;
                            break 'kernel;
                        }
                        continue;
                    }
                    let set = mask.is_set(nx as u32, ny as u32);
                    if dilate_mode {
                        if set {
                            value = true;
                            break 'kernel;
                        }
                    } else if !set {
                        value = false;
                        break 'kernel;
                    }
                }
            }
            let index = (y as usize) * (width as usize) + (x as usize);
            output[index] = u8::from(value);
        }
    }
    Ok(())
}

/// Soft feathering: distance-like alpha falloff inside `radius` of the edge.
///
/// Foreground pixels near the boundary receive values in `1..=254`; interior
/// stays `255` and background stays `0`. Uses Chebyshev distance.
///
/// # Errors
///
/// Returns capacity or radius errors like [`dilate`].
pub fn feather(mask: DenseMask<'_>, radius: u32, output: &mut [u8]) -> Result<(), MaskError> {
    if radius == 0 {
        return Err(MaskError::EmptyDimensions);
    }
    let needed = mask.data().len();
    if output.len() < needed {
        return Err(MaskError::InsufficientCapacity);
    }
    let width = mask.width();
    let height = mask.height();
    let r = radius as i32;

    for y in 0..height {
        for x in 0..width {
            let index = (y as usize) * (width as usize) + (x as usize);
            if !mask.is_set(x, y) {
                output[index] = 0;
                continue;
            }
            let mut min_dist = radius;
            for dy in -r..=r {
                for dx in -r..=r {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let outside = nx < 0
                        || ny < 0
                        || nx >= width as i32
                        || ny >= height as i32
                        || !mask.is_set(nx as u32, ny as u32);
                    if outside {
                        let dist = dx.unsigned_abs().max(dy.unsigned_abs());
                        min_dist = min_dist.min(dist);
                    }
                }
            }
            output[index] = if min_dist >= radius {
                255
            } else {
                let alpha = ((min_dist as f32 / radius as f32 * 255.0) + 0.5) as u8;
                alpha.max(1)
            };
        }
    }
    Ok(())
}

/// Fills enclosed background holes entirely surrounded by foreground.
///
/// Uses a flood-fill of background pixels reachable from the image border;
/// unreachable background becomes foreground.
///
/// `queue_scratch` must hold at least `width * height` `(x, y)` slots.
///
/// # Errors
///
/// Returns capacity errors when buffers are too small.
pub fn fill_holes(
    mask: DenseMask<'_>,
    output: &mut [u8],
    visited: &mut [bool],
    queue_scratch: &mut [(u32, u32)],
) -> Result<(), MaskError> {
    let width = mask.width();
    let height = mask.height();
    let needed = mask.data().len();
    if output.len() < needed || visited.len() < needed || queue_scratch.len() < needed {
        return Err(MaskError::InsufficientCapacity);
    }
    for (dst, src) in output.iter_mut().zip(mask.data().iter()) {
        *dst = u8::from(*src != 0);
    }
    for value in visited.iter_mut().take(needed) {
        *value = false;
    }

    let mut q_len = 0_usize;
    for x in 0..width {
        try_enqueue(mask, width, x, 0, visited, queue_scratch, &mut q_len);
        if height > 1 {
            try_enqueue(
                mask,
                width,
                x,
                height - 1,
                visited,
                queue_scratch,
                &mut q_len,
            );
        }
    }
    for y in 0..height {
        try_enqueue(mask, width, 0, y, visited, queue_scratch, &mut q_len);
        if width > 1 {
            try_enqueue(
                mask,
                width,
                width - 1,
                y,
                visited,
                queue_scratch,
                &mut q_len,
            );
        }
    }

    let mut head = 0_usize;
    while head < q_len {
        let (x, y) = queue_scratch[head];
        head += 1;
        let neighbors = [
            (x.wrapping_sub(1), y),
            (x + 1, y),
            (x, y.wrapping_sub(1)),
            (x, y + 1),
        ];
        for (nx, ny) in neighbors {
            if nx >= width || ny >= height {
                continue;
            }
            try_enqueue(mask, width, nx, ny, visited, queue_scratch, &mut q_len);
        }
    }

    for y in 0..height {
        for x in 0..width {
            let index = (y as usize) * (width as usize) + (x as usize);
            if !mask.is_set(x, y) && !visited[index] {
                output[index] = 1;
            }
        }
    }
    Ok(())
}

fn try_enqueue(
    mask: DenseMask<'_>,
    width: u32,
    x: u32,
    y: u32,
    visited: &mut [bool],
    queue_scratch: &mut [(u32, u32)],
    q_len: &mut usize,
) {
    let index = (y as usize) * (width as usize) + (x as usize);
    if visited[index] || mask.is_set(x, y) || *q_len >= queue_scratch.len() {
        return;
    }
    visited[index] = true;
    queue_scratch[*q_len] = (x, y);
    *q_len += 1;
}
