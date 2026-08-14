//! Inference tiling helpers for large frames (4K/8K / small objects).
//!
//! Generates overlapping tile windows; hosts run detectors per tile and map
//! boxes back with [`tile_to_global`].
#![allow(clippy::cast_precision_loss)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::{CoreError, Rect};

/// One tile window inside a full-resolution frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileWindow {
    /// Tile column index.
    pub col: u32,
    /// Tile row index.
    pub row: u32,
    /// Left pixel (inclusive).
    pub x0: u32,
    /// Top pixel (inclusive).
    pub y0: u32,
    /// Tile width in pixels.
    pub width: u32,
    /// Tile height in pixels.
    pub height: u32,
}

/// Generates overlapping tiles covering `frame_w` × `frame_h`.
///
/// `tile` is the tile edge length; `overlap` is the overlap in pixels
/// (`0..tile`). Returns tiles in row-major order.
///
/// # Errors
///
/// Returns [`CoreError::InvalidThreshold`] when sizes are zero or overlap ≥ tile.
pub fn generate_tiles(
    frame_w: u32,
    frame_h: u32,
    tile: u32,
    overlap: u32,
) -> Result<Vec<TileWindow>, CoreError> {
    if frame_w == 0 || frame_h == 0 || tile == 0 || overlap >= tile {
        return Err(CoreError::InvalidThreshold);
    }
    let stride = tile - overlap;
    let mut out = Vec::new();
    let mut row = 0_u32;
    let mut y = 0_u32;
    loop {
        let y0 = y.min(frame_h.saturating_sub(tile));
        let height = tile.min(frame_h.saturating_sub(y0));
        let mut col = 0_u32;
        let mut x = 0_u32;
        loop {
            let x0 = x.min(frame_w.saturating_sub(tile));
            let width = tile.min(frame_w.saturating_sub(x0));
            out.push(TileWindow {
                col,
                row,
                x0,
                y0,
                width,
                height,
            });
            if x0 + width >= frame_w {
                break;
            }
            x = x.saturating_add(stride);
            col = col.saturating_add(1);
        }
        if y0 + height >= frame_h {
            break;
        }
        y = y.saturating_add(stride);
        row = row.saturating_add(1);
    }
    Ok(out)
}

/// Maps a box from tile-local coordinates to full-frame coordinates.
///
/// # Errors
///
/// Returns geometry errors from [`Rect::new`].
pub fn tile_to_global(tile: TileWindow, local: Rect) -> Result<Rect, CoreError> {
    let dx = tile.x0 as f32;
    let dy = tile.y0 as f32;
    Rect::new(
        local.left() + dx,
        local.top() + dy,
        local.right() + dx,
        local.bottom() + dy,
    )
    .map_err(|_| CoreError::NonFinite)
}
