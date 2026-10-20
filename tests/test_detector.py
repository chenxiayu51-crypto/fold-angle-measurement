import unittest
from pathlib import Path
import cv2
import numpy as np

from fold_angle.detector import edge_candidates, SideEdgeDetector, lower_edge_from_lines


class DetectionTests(unittest.TestCase):
    def test_real_upright_rail_is_not_lost_by_segment_refinement(self):
        image = cv2.imread(str(Path(__file__).parent / 'fixtures/upright_frame_297.png'))
        candidates = edge_candidates(image, [244, 180], 150)
        self.assertTrue(any(89 < c.angle_deg < 98 for c in candidates))

    def test_lower_body_alone_must_not_be_reported_as_closed_phone(self):
        image = np.full((240, 320, 3), 220, np.uint8)
        cv2.rectangle(image, (40, 180), (140, 190), (20, 20, 20), -1)
        config = dict(reference_size=[320, 240], base_template_box=[40, 180, 140, 193],
                      lower_edge=[[140, 181], [40, 181]], hinge=[140, 180], body_length=100)
        for fixed in [False, True]:
            with self.subTest(fixed=fixed):
                result = SideEdgeDetector(dict(config, fixed_geometry=fixed), image).detect(image)
                self.assertFalse(result.valid)
                self.assertIsNone(result.angle_deg)

    def test_partial_lower_edge_is_enough_to_measure_its_direction(self):
        lines = np.array([[[243.1, 184., 115.6, 182.1]],
                          [[238.1, 178.1, 124.3, 176.2]]], dtype=np.float32)
        lower = lower_edge_from_lines(lines, [[241, 179], [92, 177]], [242, 175], 150)
        self.assertIsNotNone(lower)
        self.assertLess(abs(lower[1, 1] - lower[0, 1]), 3)

    def test_blank_image_has_no_candidates(self):
        image = np.full((256, 480, 3), 220, dtype=np.uint8)
        self.assertEqual(edge_candidates(image, [200, 180], 150), [])

    def test_dark_side_edges_at_acute_and_obtuse_angles(self):
        for degrees in [30, 90, 150]:
            with self.subTest(degrees=degrees):
                image = np.full((256, 480, 3), 220, dtype=np.uint8)
                hinge = np.array([210, 190])
                free = hinge + 145 * np.array([-np.cos(np.radians(degrees)), -np.sin(np.radians(degrees))])
                cv2.line(image, tuple(hinge), tuple(free.astype(int)), (20, 20, 20), 7)
                candidates = edge_candidates(image, hinge, 150)
                self.assertTrue(candidates)
                self.assertLess(min(abs(c.angle_deg - degrees) for c in candidates), 2)

    def test_far_edge_does_not_count_as_hinge_edge(self):
        image = np.full((256, 480, 3), 220, dtype=np.uint8)
        cv2.line(image, (20, 35), (160, 35), (20, 20, 20), 7)
        self.assertEqual(edge_candidates(image, [210, 190], 150), [])

    def test_blank_frame_after_initialization_is_invalid(self):
        first = np.full((256, 480, 3), 220, dtype=np.uint8)
        cv2.rectangle(first, (60, 180), (210, 190), (20, 20, 20), -1)
        config = dict(reference_size=[480, 256], base_template_box=[60, 179, 210, 192],
                      lower_edge=[[60, 185], [210, 185]], hinge=[210, 180],
                      body_length=150)
        detector = SideEdgeDetector(config, first)
        result = detector.detect(np.full_like(first, 220))
        self.assertFalse(result.valid)
        self.assertIsNone(result.angle_deg)


if __name__ == '__main__':
    unittest.main()
