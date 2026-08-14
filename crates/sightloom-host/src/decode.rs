//! Optional JPEG/PNG decode for encoded [`PhotoView`]s (feature `image-decode`).

use crate::error::HostError;
use sightloom::{FrameView, PhotoView, PixelFormat};

/// Decoded RGB8 raster ready for preprocess / embed.
#[derive(Clone, Debug)]
pub struct DecodedRgb {
    /// Packed RGB8.
    pub rgb: Vec<u8>,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
}

impl DecodedRgb {
    /// Borrows as a [`FrameView`].
    #[must_use]
    pub fn as_frame_view(&self) -> FrameView<'_> {
        FrameView::new(
            self.width,
            self.height,
            (self.width as usize).saturating_mul(3),
            PixelFormat::Rgb8,
            &self.rgb,
        )
    }
}

/// Decodes a photo into RGB8.
///
/// Preference: existing `frame` → copy RGB; else decode `encoded` JPEG/PNG when
/// the `image-decode` feature is enabled.
///
/// # Errors
///
/// Missing buffers / decode failures / unsupported format without feature.
pub fn decode_photo_rgb(photo: &PhotoView<'_>) -> Result<DecodedRgb, HostError> {
    if let Some(frame) = photo.frame {
        let rgb = crate::reference::frame_to_rgb8(&frame)?;
        return Ok(DecodedRgb {
            rgb,
            width: frame.width,
            height: frame.height,
        });
    }
    let Some(encoded) = photo.encoded else {
        return Err(HostError::Runtime(
            "PhotoView has neither frame nor encoded bytes".into(),
        ));
    };
    decode_encoded_rgb(encoded)
}

/// Decodes JPEG/PNG bytes to RGB8.
///
/// # Errors
///
/// Without `image-decode`, always errors. With the feature, image crate errors.
#[cfg(feature = "image-decode")]
pub fn decode_encoded_rgb(encoded: &[u8]) -> Result<DecodedRgb, HostError> {
    let img = image::load_from_memory(encoded)
        .map_err(|e| HostError::Preprocess(format!("image decode: {e}")))?
        .to_rgb8();
    let width = img.width();
    let height = img.height();
    Ok(DecodedRgb {
        rgb: img.into_raw(),
        width,
        height,
    })
}

/// Stub when `image-decode` is off.
///
/// # Errors
///
/// Always returns [`HostError::Preprocess`] unless the feature is enabled.
#[cfg(not(feature = "image-decode"))]
pub fn decode_encoded_rgb(_encoded: &[u8]) -> Result<DecodedRgb, HostError> {
    Err(HostError::Preprocess(
        "encoded photo requires feature `image-decode` (JPEG/PNG) or provide PhotoView::frame"
            .into(),
    ))
}

#[cfg(all(test, feature = "image-decode"))]
mod tests {
    use super::*;
    use image::{ImageBuffer, ImageFormat, Rgb};
    use std::io::Cursor;

    fn tiny_png() -> Vec<u8> {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_pixel(1, 1, Rgb([200, 10, 30]));
        let mut buf = Cursor::new(Vec::new());
        img.write_to(&mut buf, ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    #[test]
    fn decodes_png() {
        let d = decode_encoded_rgb(&tiny_png()).unwrap();
        assert_eq!((d.width, d.height), (1, 1));
        assert_eq!(d.rgb, vec![200, 10, 30]);
    }

    #[test]
    fn decode_photo_prefers_frame_over_encoded() {
        let rgb = [10_u8, 20, 30, 40, 50, 60];
        let frame = FrameView::new(2, 1, 6, PixelFormat::Rgb8, &rgb);
        let enc = tiny_png();
        let photo = PhotoView {
            frame: Some(frame),
            encoded: Some(&enc),
        };
        let d = decode_photo_rgb(&photo).unwrap();
        assert_eq!((d.width, d.height), (2, 1));
        assert_eq!(d.rgb, rgb);
    }

    #[test]
    fn decode_photo_from_encoded_png() {
        let enc = tiny_png();
        let photo = PhotoView::from_encoded(&enc);
        let d = decode_photo_rgb(&photo).unwrap();
        assert_eq!((d.width, d.height), (1, 1));
        assert_eq!(d.rgb, vec![200, 10, 30]);
    }
}

#[cfg(all(test, not(feature = "image-decode")))]
mod tests_no_decode {
    use super::*;

    #[test]
    fn encoded_errors_without_feature() {
        let err = decode_encoded_rgb(&[0xFF, 0xD8, 0xFF]).unwrap_err();
        assert!(matches!(err, HostError::Preprocess(_)));
    }
}
