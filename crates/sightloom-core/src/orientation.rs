//! Exact orientation signs for finite `f32` coordinates.

use core::cmp::Ordering;

use crate::Point;

const LIMBS: usize = 9;
const MIN_PRODUCT_EXPONENT: i16 = -298;
const FILTER_ERROR_FACTOR: f64 = 8.0 * f64::EPSILON;

/// Returns the exact ordering of the orientation determinant against zero.
pub(crate) fn orientation_sign(start: Point, end: Point, point: Point) -> Ordering {
    let start_x = f64::from(start.x());
    let start_y = f64::from(start.y());
    let end_x = f64::from(end.x());
    let end_y = f64::from(end.y());
    let point_x = f64::from(point.x());
    let point_y = f64::from(point.y());
    let terms = [
        end_x * point_y,
        -(end_x * start_y),
        -(start_x * point_y),
        -(end_y * point_x),
        end_y * start_x,
        start_y * point_x,
    ];

    let mut estimate = 0.0;
    let mut magnitude = 0.0;
    for term in terms {
        estimate += term;
        magnitude += term.abs();
    }

    // Each product is exact: two f32 significands need at most 48 of f64's
    // 53 bits. This factor comfortably bounds rounding in both six-term sums.
    let error_bound = magnitude * FILTER_ERROR_FACTOR;
    if estimate > error_bound {
        Ordering::Greater
    } else if estimate < -error_bound {
        Ordering::Less
    } else {
        exact_orientation_sign(start, end, point)
    }
}

fn exact_orientation_sign(start: Point, end: Point, point: Point) -> Ordering {
    let mut positive = [0_u64; LIMBS];
    let mut negative = [0_u64; LIMBS];

    accumulate_product(&mut positive, &mut negative, end.x(), point.y(), false);
    accumulate_product(&mut positive, &mut negative, end.x(), start.y(), true);
    accumulate_product(&mut positive, &mut negative, start.x(), point.y(), true);
    accumulate_product(&mut positive, &mut negative, end.y(), point.x(), true);
    accumulate_product(&mut positive, &mut negative, end.y(), start.x(), false);
    accumulate_product(&mut positive, &mut negative, start.y(), point.x(), false);

    compare_magnitudes(&positive, &negative)
}

fn accumulate_product(
    positive: &mut [u64; LIMBS],
    negative: &mut [u64; LIMBS],
    left: f32,
    right: f32,
    subtract: bool,
) {
    let left = decompose(left);
    let right = decompose(right);
    if left.mantissa == 0 || right.mantissa == 0 {
        return;
    }

    let product = left.mantissa * right.mantissa;
    let shift = usize::try_from(left.exponent + right.exponent - MIN_PRODUCT_EXPONENT)
        .expect("finite f32 product exponent is in range");
    let destination = if left.negative ^ right.negative ^ subtract {
        negative
    } else {
        positive
    };
    add_shifted(destination, product, shift);
}

#[derive(Clone, Copy)]
struct FloatParts {
    negative: bool,
    mantissa: u64,
    exponent: i16,
}

fn decompose(value: f32) -> FloatParts {
    let bits = value.to_bits();
    let fraction = bits & 0x007f_ffff;
    let exponent_bits = (bits >> 23) & 0xff;
    let (mantissa, exponent) = if exponent_bits == 0 {
        (u64::from(fraction), -149)
    } else {
        let exponent_bits = i16::try_from(exponent_bits).expect("masked f32 exponent fits in i16");
        (u64::from(fraction | 0x0080_0000), exponent_bits - 150)
    };

    FloatParts {
        negative: bits >> 31 != 0,
        mantissa,
        exponent,
    }
}

fn add_shifted(magnitude: &mut [u64; LIMBS], value: u64, shift: usize) {
    let limb = shift / u64::BITS as usize;
    let offset = shift % u64::BITS as usize;
    add_at(magnitude, limb, value << offset);
    if offset != 0 {
        add_at(magnitude, limb + 1, value >> (u64::BITS as usize - offset));
    }
}

fn add_at(magnitude: &mut [u64; LIMBS], index: usize, value: u64) {
    let mut carry = value;
    for limb in &mut magnitude[index..] {
        if carry == 0 {
            return;
        }
        let (sum, overflow) = limb.overflowing_add(carry);
        *limb = sum;
        carry = u64::from(overflow);
    }
    debug_assert_eq!(carry, 0);
}

fn compare_magnitudes(left: &[u64; LIMBS], right: &[u64; LIMBS]) -> Ordering {
    for index in (0..LIMBS).rev() {
        match left[index].cmp(&right[index]) {
            Ordering::Equal => {}
            ordering => return ordering,
        }
    }
    Ordering::Equal
}
