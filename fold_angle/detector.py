"""LSD side-edge detection with a manually initialized lower-body tracker.

This is a scene-specific classical vision baseline, not a trained object
detector. Scores express heuristic image support, not calibrated probability.
"""
from dataclasses import dataclass
import cv2
import numpy as np

from .geometry import directed_angle, orient_edge
from .filtering import AdaptiveAngleFilter


@dataclass
class Candidate:
    edge: np.ndarray
    angle_deg: float
    score: float
    dark_support: float
    length: float


@dataclass
class Measurement:
    valid: bool
    angle_deg: float | None
    score: float
    reason: str
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    hinge: np.ndarray | None = None
    base_score: float = 0.0
    candidate_count: int = 0
    filtered_angle_deg: float | None = None
    selected_edge_angle_deg: float | None = None
    observed_lower: np.ndarray | None = None
    observed_upper: np.ndarray | None = None
    registration_offset: np.ndarray | None = None
    fixed_geometry: bool = False
    fusion_count: int = 1


def _dark_support(frame, a, b):
    vector = b - a
    length = np.linalg.norm(vector)
    normal = np.array([-vector[1], vector[0]]) / length
    points = a + np.linspace(.12, .90, 50)[:, None] * vector
    support = []
    for offset in [-4, -2, 0, 2, 4]:
        xy = np.rint(points + offset * normal).astype(int)
        xy[:, 0] = np.clip(xy[:, 0], 0, frame.shape[1] - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, frame.shape[0] - 1)
        pixels = frame[xy[:, 1], xy[:, 0]].astype(float)
        luminance = pixels @ np.array([.114, .587, .299])
        support.append((luminance < 95) & (np.ptp(pixels, axis=1) < 55))
    return float(np.any(support, axis=0).mean())


def edge_candidates(frame, hinge, body_length, lines=None):
    """Find dark long edges extending from the hinge into the opening half-plane."""
    hinge = np.asarray(hinge, dtype=float)
    if lines is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE).detect(gray)[0]
    if lines is None:
        return []
    candidates = []
    for line in lines.reshape(-1, 4):
        edge = orient_edge(line.reshape(2, 2), hinge)
        a, b = edge
        length = float(np.linalg.norm(b - a))
        if not .55 * body_length <= length <= 1.18 * body_length:
            continue
        direction = (b - a) / length
        delta = a - hinge
        perpendicular = abs(delta[0] * direction[1] - delta[1] * direction[0])
        along = float(delta @ direction)
        if perpendicular > .11 * body_length or not -.18 * body_length < along < .26 * body_length:
            continue
        if b[1] > hinge[1] + .12 * body_length:
            continue
        support = _dark_support(frame, a, b)
        if support < .86:
            continue
        angle = directed_angle([-1, 0], direction)
        proximity = max(0, 1 - np.linalg.norm(delta) / (.35 * body_length))
        score = (.5 * min(length / body_length, 1.) + .35 * support + .15 * proximity)
        candidates.append(Candidate(edge, angle, float(score), support, length))
    return candidates


def lower_edge_from_lines(lines, expected, hinge, body_length):
    """Match a visible rail segment by its direction and line location.

    LSD may split a rail at a reflection; its endpoints need not span the
    whole initialized body. Retain the observed segment without extrapolation.
    """
    expected = np.asarray(expected, dtype=float)
    options = []
    if lines is None:
        return None
    for line in lines.reshape(-1, 4):
        edge = orient_edge(line.reshape(2, 2), hinge)
        a, b = edge
        vector = b - a
        length = np.linalg.norm(vector)
        if not .60 * body_length < length < 1.18 * body_length or abs(vector[1]) > .06 * length:
            continue
        if b[0] > a[0] or abs(a[0] - expected[0, 0]) > 18 or b[0] > expected[1, 0] + .3 * body_length:
            continue
        y_at_hinge = a[1] + (expected[0, 0] - a[0]) * vector[1] / vector[0]
        error = abs(y_at_hinge - expected[0, 1])
        if error < 12:
            options.append((error + .02 * abs(a[0] - expected[0, 0]), edge))
    return min(options, key=lambda item: item[0])[1] if options else None


class SideEdgeDetector:
    def __init__(self, config, first_frame):
        self.mirrored_detector = None
        if config.get('mirror_input', False):
            # Reuse the existing left-pointing-base geometry in canonical
            # coordinates, then reflect measurements back to source pixels.
            width = first_frame.shape[1]
            canonical = dict(config, mirror_input=False)
            x0,y0,x1,y1 = config['base_template_box']
            canonical['base_template_box'] = [width-x1,y0,width-x0,y1]
            canonical['lower_edge'] = [[width-1-x,y] for x,y in config['lower_edge']]
            canonical['hinge'] = [width-1-config['hinge'][0],config['hinge'][1]]
            self.mirrored_detector = SideEdgeDetector(canonical, cv2.flip(first_frame,1))
            self.fixed_geometry = self.mirrored_detector.fixed_geometry
            self.hinge0 = np.asarray(config['hinge'],dtype=float)
            self.fixed_lower_ray = self.mirrored_detector.fixed_lower_ray.copy()
            self.fixed_lower_ray[:,0] = width-1-self.fixed_lower_ray[:,0]
            self.mirror_width = width
            return
        self.config = config
        width, height = config['reference_size']
        if first_frame.shape[:2] != (height, width):
            raise ValueError('Frame size differs from reference_size; adjust the initialization config')
        x0, y0, x1, y1 = map(int, config['base_template_box'])
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError('base_template_box must lie inside the frame')
        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.template = gray[y0:y1, x0:x1].copy()
        if self.template.std() < 2:
            raise ValueError('Base template has too little texture')
        self.origin = np.array([x0, y0])
        self.position = self.origin.copy()
        self.hinge0 = np.asarray(config['hinge'], dtype=float)
        self.lower0 = orient_edge(config['lower_edge'], self.hinge0)
        self.length = float(config['body_length'])
        if self.length <= 0:
            raise ValueError('body_length must be positive')
        self.previous = None
        self.gap = 0
        # Standard refinement truncates these thin reflective rails into short
        # pieces; keep the unrefined support and apply our physical constraints.
        self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)
        self.fixed_geometry = bool(config.get('fixed_geometry', False))
        self.frame_index = -1
        self.fps = float(config.get('fps', 30.))
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('fps must be positive')
        self.angle_filter = AdaptiveAngleFilter(**config.get('angle_filter', {}))
        if self.fixed_geometry:
            initialized = lower_edge_from_lines(self.lsd.detect(gray)[0], self.lower0, self.hinge0, self.length)
            if initialized is not None:
                self.lower0 = initialized.copy()
        self.fixed_direction = self.lower0[1] - self.lower0[0]
        if not np.isfinite(self.fixed_direction).all() or np.linalg.norm(self.fixed_direction) < 1:
            raise ValueError('Lower reference edge must have finite nonzero length')
        self.fixed_direction /= np.linalg.norm(self.fixed_direction)
        self.fixed_lower_ray = np.array([self.hinge0, self.hinge0 + self.length * self.fixed_direction])

    def _invalid(self, reason, base_score=0., lower=None, hinge=None, count=0):
        self.gap += 1
        self.angle_filter.reset()
        if self.fixed_geometry:
            return Measurement(False, None, 0., reason, lower=self.fixed_lower_ray.copy(),
                               hinge=self.hinge0.copy(), base_score=base_score, candidate_count=count,
                               observed_lower=lower, registration_offset=(self.position - self.origin).copy(),
                               fixed_geometry=True)
        return Measurement(False, None, 0., reason, lower=lower, hinge=hinge,
                           base_score=base_score, candidate_count=count)

    def detect(self, frame):
        if self.mirrored_detector is not None:
            result = self.mirrored_detector.detect(cv2.flip(frame,1))
            for name in ('lower','upper','observed_lower','observed_upper'):
                points = getattr(result,name)
                if points is not None:
                    points = points.copy()
                    points[:,0] = self.mirror_width-1-points[:,0]
                    setattr(result,name,points)
            if result.hinge is not None:
                result.hinge = result.hinge.copy()
                result.hinge[0] = self.mirror_width-1-result.hinge[0]
            if result.registration_offset is not None:
                result.registration_offset = result.registration_offset.copy()
                result.registration_offset[0] *= -1
            return result
        self.frame_index += 1
        if frame.shape[1::-1] != tuple(self.config['reference_size']):
            return self._invalid('frame_size_changed')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        th, tw = self.template.shape
        rx, ry = self.config.get('template_search_radius', [12, 10])
        # Enlarge search after loss to permit recovery without adapting the template.
        expansion = min(self.gap, 5) + 1
        x, y = self.position
        xa, ya = max(0, x - rx * expansion), max(0, y - ry * expansion)
        xb, yb = min(gray.shape[1], x + tw + rx * expansion), min(gray.shape[0], y + th + ry * expansion)
        scores = cv2.matchTemplate(gray[ya:yb, xa:xb], self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(scores)
        if not np.isfinite(score) or score < self.config.get('min_template_score', .65):
            return self._invalid('base_template_not_found', score)
        self.position = np.array([xa + location[0], ya + location[1]])
        offset = self.position - self.origin
        hinge, lower = self.hinge0 + offset, self.lower0 + offset
        lines = self.lsd.detect(gray)[0]
        if lines is None:
            return self._invalid('no_lines', score, hinge=hinge)
        # Refine the lower direction with a currently observed long near-horizontal edge.
        observed_lower = lower_edge_from_lines(lines, lower, hinge, self.length)
        if observed_lower is None and not self.fixed_geometry:
            return self._invalid('lower_edge_not_found', score, hinge=hinge)
        if not self.fixed_geometry:
            lower = observed_lower
        invalid_lower = observed_lower if self.fixed_geometry else lower
        candidates = edge_candidates(frame, hinge, self.length, lines)
        if not candidates:
            return self._invalid('upper_edge_not_found', score, invalid_lower, hinge)
        tilted = any(4 < c.angle_deg < 175 and c.score > .73 for c in candidates)
        ranked = []
        for candidate in candidates:
            angle = directed_angle(lower[1] - lower[0], candidate.edge[1] - candidate.edge[0])
            # The stationary lower rail is always present. A near-closed upper
            # rail must be a separately visible edge above it, never the same
            # segment or its lower silhouette/reflection.
            if candidate.angle_deg < 4:
                cx, cy = candidate.edge[0]
                exclusion_lower = observed_lower if observed_lower is not None else lower
                slope = (exclusion_lower[1, 1] - exclusion_lower[0, 1]) / (exclusion_lower[1, 0] - exclusion_lower[0, 0])
                lower_y = exclusion_lower[0, 1] + (cx - exclusion_lower[0, 0]) * slope
                if cy >= lower_y - 2:
                    continue
            quality = candidate.score
            if tilted and candidate.angle_deg < 4:
                quality -= .55  # A visible raised edge must not be replaced by the stationary lower edge.
            if candidate.angle_deg < 4 and candidate.edge[0, 1] > hinge[1] + 4:
                quality -= .12
            if self.previous is not None and self.gap <= self.config.get('max_reacquire_gap', 8):
                change = abs(angle - self.previous)
                allowed = self.config.get('max_angle_step_deg', 24) * (1 + min(self.gap, 3))
                if change > allowed:
                    continue
                quality -= .10 * min(change / 15, 2)
            ranked.append((quality, angle, candidate))
        if not ranked:
            return self._invalid('angle_discontinuity', score, invalid_lower, hinge, len(candidates))
        quality, angle, chosen = max(ranked, key=lambda item: item[0])
        if quality < self.config.get('min_edge_score', .66):
            return self._invalid('ambiguous_or_weak_upper_edge', score, invalid_lower, hinge, len(candidates))
        if self.fixed_geometry:
            selected_angle = angle
            compatible = [(q, a, c) for q, a, c in ranked
                          if abs(a - selected_angle) <= 2.5 and q >= self.config.get('min_edge_score', .66)]
            weights = [c.length * q * q for q, a, c in compatible]
            angle = float(np.average([a for q, a, c in compatible], weights=weights))
            filtered = self.angle_filter.update(angle, self.frame_index / self.fps)
            radians = np.radians(filtered)
            rotation = np.array([[np.cos(radians), -np.sin(radians)],
                                 [np.sin(radians), np.cos(radians)]])
            upper_ray = np.array([self.hinge0, self.hinge0 + self.length * (rotation @ self.fixed_direction)])
            self.previous, self.gap = angle, 0
            return Measurement(True, angle, float(min(quality, score)), '',
                               self.fixed_lower_ray.copy(), upper_ray, self.hinge0.copy(), float(score),
                               len(candidates), filtered_angle_deg=filtered,
                               selected_edge_angle_deg=selected_angle, observed_lower=observed_lower,
                               observed_upper=chosen.edge.copy(), registration_offset=offset.copy(),
                               fixed_geometry=True, fusion_count=len(compatible))
        self.previous = angle
        self.gap = 0
        return Measurement(True, angle, float(min(quality, score)), '', lower, chosen.edge,
                           hinge, float(score), len(candidates))
