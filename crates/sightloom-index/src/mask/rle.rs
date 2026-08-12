//! Run-length encoded binary masks.

use super::MaskError;

/// A COCO-style row-major RLE mask.
///
/// `counts` stores alternating background/foreground run lengths starting with
/// a background run (which may be zero).
#[derive(Clone, Copy, Debug)]
pub struct RleMask<'a> {
    width: u32,
    height: u32,
    counts: &'a [u32],
}

impl<'a> RleMask<'a> {
    /// Borrows an RLE mask when dimensions are non-empty and runs cover exactly
    /// `width * height` pixels.
    ///
    /// # Errors
    ///
    /// Returns [`MaskError::EmptyDimensions`] or [`MaskError::LengthMismatch`].
    pub fn new(width: u32, height: u32, counts: &'a [u32]) -> Result<Self, MaskError> {
        if width == 0 || height == 0 {
            return Err(MaskError::EmptyDimensions);
        }
        let total = u64::from(width).saturating_mul(u64::from(height));
        let covered: u64 = counts.iter().map(|value| u64::from(*value)).sum();
        if covered != total {
            return Err(MaskError::LengthMismatch);
        }
        Ok(Self {
            width,
            height,
            counts,
        })
    }

    /// Returns the mask width.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Returns the mask height.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.height
    }

    /// Returns the RLE run counts.
    #[must_use]
    pub const fn counts(self) -> &'a [u32] {
        self.counts
    }

    /// Counts foreground pixels from the odd-indexed runs.
    #[must_use]
    pub fn area(self) -> u32 {
        self.counts.iter().skip(1).step_by(2).copied().sum()
    }

    /// Decodes RLE into a dense byte buffer (`0`/`1`).
    ///
    /// # Errors
    ///
    /// Returns [`MaskError::InsufficientCapacity`] when `output` is shorter
    /// than `width * height`.
    pub fn decode_into(self, output: &mut [u8]) -> Result<(), MaskError> {
        let needed = (self.width as usize).saturating_mul(self.height as usize);
        if output.len() < needed {
            return Err(MaskError::InsufficientCapacity);
        }
        let mut offset = 0_usize;
        let mut foreground = false;
        for &run in self.counts {
            let end = offset + run as usize;
            let fill = u8::from(foreground);
            for slot in output.iter_mut().take(end).skip(offset) {
                *slot = fill;
            }
            offset = end;
            foreground = !foreground;
        }
        Ok(())
    }

    /// Encodes a dense `0`/`1` buffer into RLE counts written to `output`.
    ///
    /// Returns the number of run values written.
    ///
    /// # Errors
    ///
    /// Returns [`MaskError::LengthMismatch`] when `data` is the wrong length,
    /// and [`MaskError::InsufficientCapacity`] when `output` cannot hold the
    /// runs (worst case `width * height + 1`).
    pub fn encode_from_dense(
        width: u32,
        height: u32,
        data: &[u8],
        output: &mut [u32],
    ) -> Result<usize, MaskError> {
        if width == 0 || height == 0 {
            return Err(MaskError::EmptyDimensions);
        }
        let expected = (width as usize).saturating_mul(height as usize);
        if data.len() != expected {
            return Err(MaskError::LengthMismatch);
        }
        if expected == 0 {
            return Ok(0);
        }

        let mut written = 0_usize;
        let mut current = data[0] != 0;
        // RLE starts with a background run.
        if current {
            if output.is_empty() {
                return Err(MaskError::InsufficientCapacity);
            }
            output[0] = 0;
            written = 1;
        }
        let mut run = 1_u32;
        for &value in &data[1..] {
            let is_fg = value != 0;
            if is_fg == current {
                run = run.saturating_add(1);
            } else {
                if written >= output.len() {
                    return Err(MaskError::InsufficientCapacity);
                }
                output[written] = run;
                written += 1;
                current = is_fg;
                run = 1;
            }
        }
        if written >= output.len() {
            return Err(MaskError::InsufficientCapacity);
        }
        output[written] = run;
        written += 1;
        Ok(written)
    }
}
