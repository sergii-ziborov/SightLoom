//! Dense binary masks over a full image frame.

use crate::MaskError;
use sightloom_core::Rect;

/// A dense binary mask with one byte per pixel (`0` or non-zero).
#[derive(Clone, Copy, Debug)]
pub struct DenseMask<'a> {
    width: u32,
    height: u32,
    data: &'a [u8],
}

impl<'a> DenseMask<'a> {
    /// Borrows a dense mask when the buffer length matches `width * height`.
    ///
    /// # Errors
    ///
    /// Returns [`MaskError::EmptyDimensions`] when either edge is zero, and
    /// [`MaskError::LengthMismatch`] when `data` is the wrong length.
    pub fn new(width: u32, height: u32, data: &'a [u8]) -> Result<Self, MaskError> {
        if width == 0 || height == 0 {
            return Err(MaskError::EmptyDimensions);
        }
        let expected = (width as usize).saturating_mul(height as usize);
        if data.len() != expected {
            return Err(MaskError::LengthMismatch);
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Returns the mask width in pixels.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Returns the mask height in pixels.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.height
    }

    /// Returns the raw mask bytes.
    #[must_use]
    pub const fn data(self) -> &'a [u8] {
        self.data
    }

    /// Returns whether pixel `(x, y)` is foreground.
    #[must_use]
    pub fn is_set(self, x: u32, y: u32) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        let index = (y as usize) * (self.width as usize) + (x as usize);
        self.data.get(index).is_some_and(|value| *value != 0)
    }

    /// Returns the axis-aligned bounds of all foreground pixels, if any.
    #[must_use]
    pub fn bbox(self) -> Option<Rect> {
        let mut min_x = self.width;
        let mut min_y = self.height;
        let mut max_x = 0_u32;
        let mut max_y = 0_u32;
        let mut found = false;

        for y in 0..self.height {
            for x in 0..self.width {
                if !self.is_set(x, y) {
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
            return None;
        }

        Rect::new(min_x as f32, min_y as f32, max_x as f32, max_y as f32).ok()
    }

    /// Counts foreground pixels.
    #[must_use]
    pub fn area(self) -> u32 {
        self.data.iter().filter(|value| **value != 0).count() as u32
    }
}
