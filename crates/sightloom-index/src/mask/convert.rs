//! Conversions between masks, polygons, and bounding boxes.

use super::{CroppedMask, DenseMask, MaskError, PolygonMask, RleMask};
use sightloom_core::{Point, Rect};

/// Converts a dense mask to its tight axis-aligned bounding box.
#[must_use]
pub fn dense_to_bbox(mask: DenseMask<'_>) -> Option<Rect> {
    mask.bbox()
}

/// Rasterizes a polygon mask into a dense buffer of size `width * height`.
///
/// # Errors
///
/// Returns capacity errors when `output` is too small.
pub fn polygon_to_dense(
    polygon: &PolygonMask<'_>,
    width: u32,
    height: u32,
    output: &mut [u8],
) -> Result<(), MaskError> {
    if width == 0 || height == 0 {
        return Err(MaskError::EmptyDimensions);
    }
    let needed = (width as usize).saturating_mul(height as usize);
    if output.len() < needed {
        return Err(MaskError::InsufficientCapacity);
    }
    for y in 0..height {
        for x in 0..width {
            let point =
                Point::new(x as f32 + 0.5, y as f32 + 0.5).map_err(|_| MaskError::NonFinite)?;
            let index = (y as usize) * (width as usize) + (x as usize);
            output[index] = u8::from(polygon.contains(point));
        }
    }
    Ok(())
}

/// Extracts an ordered boundary-ish polygon from a cropped mask's outer bbox corners.
///
/// Compact approximation (tight AABB corners). Prefer
/// [`crate::dense_to_contour`] on a dense raster for Moore boundary tracing.
///
/// Writes up to four points into `output` and returns the count.
///
/// # Errors
///
/// Returns [`MaskError::EmptyPolygon`] when the crop has no foreground, and
/// [`MaskError::InsufficientCapacity`] when fewer than four output slots exist.
pub fn cropped_to_polygon_approx(
    mask: CroppedMask<'_>,
    output: &mut [Point],
) -> Result<usize, MaskError> {
    if output.len() < 4 {
        return Err(MaskError::InsufficientCapacity);
    }
    let mut min_x = mask.width();
    let mut min_y = mask.height();
    let mut max_x = 0_u32;
    let mut max_y = 0_u32;
    let mut found = false;
    for y in 0..mask.height() {
        for x in 0..mask.width() {
            if !mask.is_set(mask.origin_x() + x, mask.origin_y() + y) {
                continue;
            }
            found = true;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x + 1);
            max_y = max_y.max(y + 1);
        }
    }
    if !found {
        return Err(MaskError::EmptyPolygon);
    }
    let left = (mask.origin_x() + min_x) as f32;
    let top = (mask.origin_y() + min_y) as f32;
    let right = (mask.origin_x() + max_x) as f32;
    let bottom = (mask.origin_y() + max_y) as f32;
    output[0] = Point::new(left, top).map_err(|_| MaskError::NonFinite)?;
    output[1] = Point::new(right, top).map_err(|_| MaskError::NonFinite)?;
    output[2] = Point::new(right, bottom).map_err(|_| MaskError::NonFinite)?;
    output[3] = Point::new(left, bottom).map_err(|_| MaskError::NonFinite)?;
    Ok(4)
}

/// Decodes RLE into dense bytes (see [`RleMask::decode_into`]).
///
/// # Errors
///
/// Propagates decode capacity errors.
pub fn rle_to_dense(mask: RleMask<'_>, output: &mut [u8]) -> Result<(), MaskError> {
    mask.decode_into(output)
}

/// Encodes dense bytes into RLE (see [`RleMask::encode_from_dense`]).
///
/// # Errors
///
/// Propagates encode errors.
pub fn dense_to_rle(
    width: u32,
    height: u32,
    data: &[u8],
    output: &mut [u32],
) -> Result<usize, MaskError> {
    RleMask::encode_from_dense(width, height, data, output)
}

/// Bounding-box rectangle as a trivial four-corner polygon.
///
/// # Errors
///
/// Returns [`MaskError::InsufficientCapacity`] when `output` has fewer than
/// four slots.
pub fn bbox_to_polygon(rect: Rect, output: &mut [Point]) -> Result<usize, MaskError> {
    if output.len() < 4 {
        return Err(MaskError::InsufficientCapacity);
    }
    output[0] = Point::new(rect.left(), rect.top()).map_err(|_| MaskError::NonFinite)?;
    output[1] = Point::new(rect.right(), rect.top()).map_err(|_| MaskError::NonFinite)?;
    output[2] = Point::new(rect.right(), rect.bottom()).map_err(|_| MaskError::NonFinite)?;
    output[3] = Point::new(rect.left(), rect.bottom()).map_err(|_| MaskError::NonFinite)?;
    Ok(4)
}
