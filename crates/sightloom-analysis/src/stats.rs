//! Portable descriptive statistics for miners and anomaly detectors.

/// Arithmetic mean of a non-empty finite sample.
#[must_use]
pub fn mean(values: &[f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let mut sum = 0.0_f32;
    for value in values {
        if !value.is_finite() {
            return None;
        }
        sum += *value;
    }
    Some(sum / values.len() as f32)
}

/// Population standard deviation (`sqrt(mean((x-mu)^2))`).
#[must_use]
pub fn stddev(values: &[f32]) -> Option<f32> {
    let mu = mean(values)?;
    if values.len() < 2 {
        return Some(0.0);
    }
    let mut acc = 0.0_f32;
    for value in values {
        let d = *value - mu;
        acc += d * d;
    }
    let var = acc / values.len() as f32;
    Some(sqrt_approx(var))
}

/// Absolute z-score `|x - mu| / sigma` (0 when sigma is ~0).
#[must_use]
pub fn z_score(value: f32, mu: f32, sigma: f32) -> Option<f32> {
    if !value.is_finite() || !mu.is_finite() || !sigma.is_finite() {
        return None;
    }
    if sigma <= 1e-6 {
        return Some(if (value - mu).abs() <= 1e-6 {
            0.0
        } else {
            f32::INFINITY
        });
    }
    Some((value - mu).abs() / sigma)
}

/// Median absolute deviation from the median (`median(|x - med|)`).
///
/// Uses `scratch` of length ≥ `values.len()` (and reuses it for median).
#[must_use]
pub fn mad(values: &[f32], scratch: &mut [f32]) -> Option<f32> {
    let med = median(values, scratch)?;
    if scratch.len() < values.len() {
        return None;
    }
    for (i, v) in values.iter().enumerate() {
        scratch[i] = (*v - med).abs();
    }
    // Second median over absolute deviations — needs a second buffer half or in-place.
    let n = values.len();
    let abs = &mut scratch[..n];
    for i in 1..abs.len() {
        let mut j = i;
        while j > 0
            && abs[j]
                .partial_cmp(&abs[j - 1])
                .unwrap_or(core::cmp::Ordering::Equal)
                .is_lt()
        {
            abs.swap(j, j - 1);
            j -= 1;
        }
    }
    let mid = n / 2;
    if n.is_multiple_of(2) {
        Some((abs[mid - 1] + abs[mid]) * 0.5)
    } else {
        Some(abs[mid])
    }
}

/// Robust z-score using median and MAD (`0.6745 * |x-med| / MAD`).
#[must_use]
pub fn robust_z_score(value: f32, med: f32, mad_value: f32) -> Option<f32> {
    if !value.is_finite() || !med.is_finite() || !mad_value.is_finite() {
        return None;
    }
    if mad_value <= 1e-6 {
        return Some(if (value - med).abs() <= 1e-6 {
            0.0
        } else {
            f32::INFINITY
        });
    }
    Some(0.6745 * (value - med).abs() / mad_value)
}

/// Simple CUSUM-style change-point score: max running deviation of prefix means.
///
/// Returns `(index_after_change, score)` for the strongest split, or `None`.
#[must_use]
pub fn change_point_cusum(values: &[f32]) -> Option<(usize, f32)> {
    if values.len() < 4 {
        return None;
    }
    let total_mean = mean(values)?;
    let mut best_i = 0_usize;
    let mut best_score = 0.0_f32;
    let mut prefix = 0.0_f32;
    for (i, v) in values.iter().enumerate().skip(1) {
        if !v.is_finite() {
            return None;
        }
        prefix += values[i - 1];
        let left_n = i as f32;
        let left_mean = prefix / left_n;
        let right_n = (values.len() - i) as f32;
        if right_n < 1.0 {
            break;
        }
        let right_sum: f32 = values[i..].iter().sum();
        let right_mean = right_sum / right_n;
        let score = (left_mean - total_mean).abs() * left_n.sqrt()
            + (right_mean - total_mean).abs() * right_n.sqrt();
        if score > best_score {
            best_score = score;
            best_i = i;
        }
    }
    (best_score > 0.0).then_some((best_i, best_score))
}

/// Median of a non-empty sample (sorts a copy into `scratch`).
///
/// # Errors
///
/// Returns `None` when `values` is empty or `scratch` is shorter than `values`.
pub fn median(values: &[f32], scratch: &mut [f32]) -> Option<f32> {
    if values.is_empty() || scratch.len() < values.len() {
        return None;
    }
    scratch[..values.len()].copy_from_slice(values);
    let slice = &mut scratch[..values.len()];
    // Insertion sort keeps this portable on no_std without `alloc` sort helpers.
    for i in 1..slice.len() {
        let mut j = i;
        while j > 0
            && slice[j]
                .partial_cmp(&slice[j - 1])
                .unwrap_or(core::cmp::Ordering::Equal)
                .is_lt()
        {
            slice.swap(j, j - 1);
            j -= 1;
        }
    }
    let mid = slice.len() / 2;
    if slice.len().is_multiple_of(2) {
        Some((slice[mid - 1] + slice[mid]) * 0.5)
    } else {
        Some(slice[mid])
    }
}

/// Hour of day `0..24` from nanoseconds since an arbitrary epoch.
#[must_use]
pub fn hour_of_day_ns(time_ns: i64) -> u8 {
    let day_ns = 86_400_i64.saturating_mul(1_000_000_000);
    let tod = time_ns.rem_euclid(day_ns);
    let hour = tod / 3_600_000_000_000;
    u8::try_from(hour.clamp(0, 23)).unwrap_or(0)
}

/// Day of week `0..7` (0 = epoch day alignment) from nanoseconds.
#[must_use]
pub fn day_of_week_ns(time_ns: i64) -> u8 {
    let day = time_ns.div_euclid(86_400_i64.saturating_mul(1_000_000_000));
    u8::try_from(day.rem_euclid(7)).unwrap_or(0)
}

fn sqrt_approx(value: f32) -> f32 {
    if value <= 0.0 || !value.is_finite() {
        return 0.0;
    }
    let mut y = value;
    for _ in 0..8 {
        y = 0.5 * (y + value / y);
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_std_z_and_median() {
        let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&values).unwrap() - 3.0).abs() < 1e-5);
        assert!(stddev(&values).unwrap() > 1.0);
        assert!((z_score(3.0, 3.0, 1.0).unwrap()).abs() < 1e-5);
        let mut scratch = [0.0_f32; 8];
        assert!((median(&values, &mut scratch).unwrap() - 3.0).abs() < 1e-5);
    }
}
