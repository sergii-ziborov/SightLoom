//! Mask representation and operation tests.

use sightloom_index::{
    CroppedMask, DenseMask, RleMask, cropped_mask_iou, dense_mask_difference, dense_mask_iou,
    dense_mask_union, dense_to_rle, dilate, erode, fill_holes, rle_to_dense,
};

#[test]
fn dense_bbox_and_area() {
    // 4x4 with a 2x2 block at (1,1)
    let data = [
        0, 0, 0, 0, //
        0, 1, 1, 0, //
        0, 1, 1, 0, //
        0, 0, 0, 0,
    ];
    let mask = DenseMask::new(4, 4, &data).expect("dense");
    assert_eq!(mask.area(), 4);
    let bbox = mask.bbox().expect("bbox");
    assert_eq!(
        (bbox.left(), bbox.top(), bbox.right(), bbox.bottom()),
        (1.0, 1.0, 3.0, 3.0)
    );
}

#[test]
fn rle_roundtrip() {
    let data = [0, 1, 1, 0, 0, 1, 0, 0, 1];
    let mut counts = [0_u32; 16];
    let n = dense_to_rle(3, 3, &data, &mut counts).expect("encode");
    let rle = RleMask::new(3, 3, &counts[..n]).expect("rle");
    let mut decoded = [0_u8; 9];
    rle_to_dense(rle, &mut decoded).expect("decode");
    assert_eq!(decoded, data);
    assert_eq!(rle.area(), 4);
}

#[test]
fn dense_iou_union_difference() {
    let a = [1, 1, 0, 0];
    let b = [0, 1, 1, 0];
    let da = DenseMask::new(2, 2, &a).unwrap();
    let db = DenseMask::new(2, 2, &b).unwrap();
    let iou = dense_mask_iou(da, db).unwrap();
    assert!((iou - 1.0 / 3.0).abs() < 1e-6);

    let mut uni = [0_u8; 4];
    dense_mask_union(da, db, &mut uni).unwrap();
    assert_eq!(uni, [1, 1, 1, 0]);

    let mut diff = [0_u8; 4];
    dense_mask_difference(da, db, &mut diff).unwrap();
    assert_eq!(diff, [1, 0, 0, 0]);
}

#[test]
fn cropped_iou() {
    let a_data = [1, 1, 1, 1];
    let b_data = [1, 1, 0, 0];
    let a = CroppedMask::new(0, 0, 2, 2, &a_data).unwrap();
    let b = CroppedMask::new(1, 0, 2, 2, &b_data).unwrap();
    // a covers (0,0)-(2,2), b covers local fg at (1,0) and (2,0)
    let iou = cropped_mask_iou(a, b);
    assert!(iou > 0.0);
}

#[test]
fn dilate_and_erode() {
    let data = [
        0, 0, 0, //
        0, 1, 0, //
        0, 0, 0,
    ];
    let mask = DenseMask::new(3, 3, &data).unwrap();
    let mut out = [0_u8; 9];
    dilate(mask, 1, &mut out).unwrap();
    assert_eq!(out.iter().filter(|v| **v != 0).count(), 9);

    let full = [1_u8; 9];
    let full_mask = DenseMask::new(3, 3, &full).unwrap();
    erode(full_mask, 1, &mut out).unwrap();
    // Only center survives radius-1 erode on 3x3 full
    assert_eq!(out, [0, 0, 0, 0, 1, 0, 0, 0, 0]);
}

#[test]
fn hole_fill_closes_interior() {
    // ring with hole in center
    let data = [
        1, 1, 1, //
        1, 0, 1, //
        1, 1, 1,
    ];
    let mask = DenseMask::new(3, 3, &data).unwrap();
    let mut out = [0_u8; 9];
    let mut visited = [false; 9];
    let mut queue = [(0_u32, 0_u32); 9];
    fill_holes(mask, &mut out, &mut visited, &mut queue).unwrap();
    assert_eq!(out, [1; 9]);
}
