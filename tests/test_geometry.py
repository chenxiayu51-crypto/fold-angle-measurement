import unittest
import numpy as np

from fold_angle.geometry import directed_angle, orient_edge


class AngleTests(unittest.TestCase):
    def test_closed_right_obtuse_and_flat(self):
        for degrees in [0, 30, 90, 150, 180]:
            with self.subTest(degrees=degrees):
                radians = np.radians(degrees)
                upper = [-np.cos(radians), -np.sin(radians)]
                self.assertAlmostEqual(directed_angle([-1, 0], upper), degrees)

    def test_edge_endpoint_order_does_not_reverse_angle(self):
        hinge = [200, 180]
        a = orient_edge([[60, 180], [198, 180]], hinge)
        b = orient_edge([[330, 105], [202, 178]], hinge)
        self.assertGreater(directed_angle(a[1] - a[0], b[1] - b[0]), 140)
        self.assertTrue(np.array_equal(b, orient_edge(b[::-1], hinge)))

    def test_reject_degenerate_or_nonfinite_vectors(self):
        for v in ([0, 0], [np.nan, 0], [np.inf, 1]):
            with self.subTest(v=v):
                with self.assertRaises(ValueError):
                    directed_angle([1, 0], v)
