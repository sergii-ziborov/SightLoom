//! Cropped compact masks stored only over their bounding box.

use crate::{DenseMask, MaskError};
use sightloom_core::Rect;

/// A binary mask cropped to a bounding rectangle in full-frame coordinates.
#[derive(Clone, Copy, Debug)]
pub struct CroppedMask<'a> {
    origin_x: u32,
    origin_y: u32,
    width: u32,
    height: u32,
    data: &'a [u8],
}

impl<'a> CroppedMask<'a> {
    /// Borrows a cropped mask when the buffer matches the crop size.
    ///
    /// # Errors
    ///
    /// Returns [`MaskError::EmptyDimensions`] or [`MaskError::LengthMismatch`]
    /// for invalid geometry or buffer size.
    pub fn new(
        origin_x: u32,
        origin_y: u32,
        width: u32,
        height: u32,
        data: &'a [u8],
    ) -> Result<Self, MaskError> {
        if width == 0 || height == 0 {
            return Err(MaskError::EmptyDimensions);
        }
        let expected = (width as usize).saturating_mul(height as usize);
        if data.len() != expected {
            return Err(MaskError::LengthMismatch);
        }
        Ok(Self {
            origin_x,
            origin_y,
            width,
            height,
            data,
        })
    }

    /// Creates a cropped mask by extracting the dense mask's foreground bbox.
    ///
    /// Writes cropped bytes into `output` and returns the view over the used
    /// prefix of `output`.
    ///
    /// # Errors
    ///
    /// Returns [`MaskError::EmptyPolygon`] when the dense mask is empty, and
    /// [`MaskError::InsufficientCapacity`] when `output` is too small.
    pub fn from_dense<'out>(
        dense: DenseMask<'_>,
        output: &'out mut [u8],
    ) -> Result<CroppedMask<'out>, MaskError> {
        let bbox = dense.bbox().ok_or(MaskError::EmptyPolygon)?;
        let origin_x = bbox.left() as u32;
        let origin_y = bbox.top() as u32;
        let width = (bbox.right() - bbox.left()) as u32;
        let height = (bbox.bottom() - bbox.top()) as u32;
        let needed = (width as usize).saturating_mul(height as usize);
        if output.len() < needed {
            return Err(MaskError::InsufficientCapacity);
        }

        for y in 0..height {
            for x in 0..width {
                let src_x = origin_x + x;
                let src_y = origin_y + y;
                let dst = (y as usize) * (width as usize) + (x as usize);
                output[dst] = u8::from(dense.is_set(src_x, src_y));
            }
        }

        CroppedMask::new(origin_x, origin_y, width, height, &output[..needed])
    }

    /// Returns the crop origin X in full-frame coordinates.
    #[must_use]
    pub const fn origin_x(self) -> u32 {
        self.origin_x
    }

    /// Returns the crop origin Y in full-frame coordinates.
    #[must_use]
    pub const fn origin_y(self) -> u32 {
        self.origin_y
    }

    /// Returns the crop width.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Returns the crop height.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.height
    }

    /// Returns the cropped mask bytes.
    #[must_use]
    pub const fn data(self) -> &'a [u8] {
        self.data
    }

    /// Returns the crop rectangle in full-frame coordinates.
    ///
    /// # Panics
    ///
    /// Panics only if internal crop bounds become inverted, which constructors
    /// prevent.
    #[must_use]
    pub fn bbox(self) -> Rect {
        Rect::new(
            self.origin_x as f32,
            self.origin_y as f32,
            (self.origin_x + self.width) as f32,
            (self.origin_y + self.height) as f32,
        )
        .expect("validated crop bounds")
    }

    /// Returns whether full-frame pixel `(x, y)` is foreground.
    #[must_use]
    pub fn is_set(self, x: u32, y: u32) -> bool {
        if x < self.origin_x || y < self.origin_y {
            return false;
        }
        let local_x = x - self.origin_x;
        let local_y = y - self.origin_y;
        if local_x >= self.width || local_y >= self.height {
            return false;
        }
        let index = (local_y as usize) * (self.width as usize) + (local_x as usize);
        self.data.get(index).is_some_and(|value| *value != 0)
    }

    /// Counts foreground pixels inside the crop.
    #[must_use]
    pub fn area(self) -> u32 {
        self.data.iter().filter(|value| **value != 0).count() as u32
    }
}
