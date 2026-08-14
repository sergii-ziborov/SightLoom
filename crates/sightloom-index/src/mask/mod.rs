//! Compact mask representations and morphology.

mod contour;
mod convert;
mod cropped;
mod dense;
mod error;
mod morph;
mod ops;
mod polygon_mask;
mod rle;

pub use contour::dense_to_contour;
#[cfg(feature = "alloc")]
pub use contour::dense_to_contours;
pub use convert::{
    bbox_to_polygon, cropped_to_polygon_approx, dense_to_bbox, dense_to_rle, polygon_to_dense,
    rle_to_dense,
};
pub use cropped::CroppedMask;
pub use dense::DenseMask;
pub use error::MaskError;
pub use morph::{dilate, erode, feather, fill_holes};
pub use ops::{
    cropped_mask_iou, dense_mask_difference, dense_mask_iou, dense_mask_union, mask_nms_by_iou,
};
pub use polygon_mask::PolygonMask;
pub use rle::RleMask;
