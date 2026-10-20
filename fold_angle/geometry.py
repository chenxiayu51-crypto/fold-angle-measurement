import numpy as np


def directed_angle(lower, upper):
    """Unsigned angle of two hinge-to-free-end vectors, in [0, 180]."""
    u, v = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if u.shape != (2,) or v.shape != (2,):
        raise ValueError('Expected two 2D vectors')
    if not np.isfinite([u, v]).all() or min(np.linalg.norm(u), np.linalg.norm(v)) < 1e-8:
        raise ValueError('Edge directions must be finite and nonzero')
    cross = u[0] * v[1] - u[1] * v[0]
    return float(np.degrees(np.arctan2(abs(cross), np.dot(u, v))))


def orient_edge(edge, hinge):
    """Put the endpoint nearest the configured hinge first."""
    points = np.asarray(edge, dtype=float).reshape(2, 2)
    if np.linalg.norm(points[0] - hinge) > np.linalg.norm(points[1] - hinge):
        points = points[::-1]
    return points.copy()
