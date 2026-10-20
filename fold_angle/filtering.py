"""Adaptive low-pass angle filtering (One Euro), with angles in degrees."""
import math


class AdaptiveAngleFilter:
    def __init__(self, min_cutoff_hz=.8, beta=.10, derivative_cutoff_hz=1.):
        if min_cutoff_hz <= 0 or beta < 0 or derivative_cutoff_hz <= 0:
            raise ValueError('Filter cutoffs must be positive and beta nonnegative')
        self.min_cutoff = float(min_cutoff_hz)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff_hz)
        self.reset()

    def reset(self):
        self.previous_raw = self.previous_filtered = self.previous_time = None
        self.derivative = 0.

    @staticmethod
    def _alpha(cutoff, dt):
        value = 2 * math.pi * cutoff * dt
        return value / (1 + value)

    def update(self, angle, timestamp):
        if not math.isfinite(angle) or not math.isfinite(timestamp):
            raise ValueError('Angle and timestamp must be finite')
        angle, timestamp = float(angle), float(timestamp)
        if self.previous_time is None:
            filtered = angle
        else:
            dt = timestamp - self.previous_time
            if dt <= 0:
                raise ValueError('Filter timestamps must increase')
            velocity = (angle - self.previous_raw) / dt
            alpha_d = self._alpha(self.derivative_cutoff, dt)
            self.derivative += alpha_d * (velocity - self.derivative)
            cutoff = self.min_cutoff + self.beta * abs(self.derivative)
            alpha = self._alpha(cutoff, dt)
            filtered = self.previous_filtered + alpha * (angle - self.previous_filtered)
        self.previous_time = timestamp
        self.previous_raw = angle
        self.previous_filtered = filtered
        return filtered
