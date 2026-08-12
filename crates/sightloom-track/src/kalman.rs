//! Constant-velocity Kalman filter for axis-aligned boxes in xyah space.

use sightloom_core::Rect;

/// Eight-dimensional Kalman state: center, aspect, height, and velocities.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanState {
    /// State mean: center-x, center-y, aspect, height, then velocities.
    pub mean: [f32; 8],
    /// Diagonal covariance approximation (full 8x8 is unnecessary for `ByteTrack`).
    pub variance: [f32; 8],
}

impl KalmanState {
    /// Initializes a filter from a measurement rectangle.
    #[must_use]
    pub fn initiate(rect: Rect) -> Self {
        let (cx, cy, a, h) = rect_to_xyah(rect);
        Self {
            mean: [cx, cy, a, h, 0.0, 0.0, 0.0, 0.0],
            variance: [
                2.0 * h,
                2.0 * h,
                1e-2,
                2.0 * h,
                10.0 * h,
                10.0 * h,
                1e-5,
                10.0 * h,
            ]
            .map(|v| (v * v).max(1e-6)),
        }
    }

    /// Predicts the next state under a constant-velocity model.
    pub fn predict(&mut self) {
        for i in 0..4 {
            self.mean[i] += self.mean[i + 4];
        }
        // Process noise scales with current height.
        let h = self.mean[3].abs().max(1.0);
        let motion = [h * h, h * h, 1e-4, h * h, h * h, h * h, 1e-8, h * h];
        for (variance, noise) in self.variance.iter_mut().zip(motion.iter()) {
            *variance = (*variance + *noise).max(1e-6);
        }
    }

    /// Incorporates a new rectangle measurement.
    pub fn update(&mut self, rect: Rect) {
        let (cx, cy, a, h) = rect_to_xyah(rect);
        let measurement = [cx, cy, a, h];
        let h_noise = (h * 0.1) * (h * 0.1);
        let noise = [h_noise.max(1e-6), h_noise.max(1e-6), 1e-2, h_noise.max(1e-6)];

        for i in 0..4 {
            let innov = measurement[i] - self.mean[i];
            let s = self.variance[i] + noise[i];
            let k = self.variance[i] / s;
            self.mean[i] += k * innov;
            self.variance[i] = ((1.0 - k) * self.variance[i]).max(1e-6);
            // Velocity residual feedback.
            let vk = self.variance[i + 4] / (self.variance[i + 4] + s);
            self.mean[i + 4] += vk * innov;
            self.variance[i + 4] = ((1.0 - vk) * self.variance[i + 4]).max(1e-6);
        }
    }

    /// Projected bounding box for the current mean.
    #[must_use]
    pub fn to_rect(self) -> Rect {
        xyah_to_rect(self.mean[0], self.mean[1], self.mean[2], self.mean[3])
    }
}

fn rect_to_xyah(rect: Rect) -> (f32, f32, f32, f32) {
    let w = rect.width().max(1e-3);
    let h = rect.height().max(1e-3);
    let cx = rect.left() + w * 0.5;
    let cy = rect.top() + h * 0.5;
    (cx, cy, w / h, h)
}

fn xyah_to_rect(cx: f32, cy: f32, a: f32, h: f32) -> Rect {
    let h = h.abs().max(1e-3);
    let w = (a.abs().max(1e-3)) * h;
    let left = cx - w * 0.5;
    let top = cy - h * 0.5;
    Rect::new(left, top, left + w, top + h)
        .unwrap_or_else(|_| Rect::new(cx, cy, cx + 1.0, cy + 1.0).expect("unit fallback is valid"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sightloom_core::Rect;

    #[test]
    fn predict_moves_with_velocity_after_update() {
        let r0 = Rect::new(0.0, 0.0, 10.0, 20.0).unwrap();
        let r1 = Rect::new(5.0, 0.0, 15.0, 20.0).unwrap();
        let mut kf = KalmanState::initiate(r0);
        kf.update(r0);
        kf.predict();
        kf.update(r1);
        kf.predict();
        let predicted = kf.to_rect();
        assert!(predicted.center().x() > r1.center().x() - 1.0);
    }
}
