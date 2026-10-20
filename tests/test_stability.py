import unittest
import cv2
import numpy as np

from fold_angle.filtering import AdaptiveAngleFilter
from fold_angle.detector import SideEdgeDetector
from fold_angle.geometry import directed_angle


class FilterTests(unittest.TestCase):
    def test_suppresses_stationary_noise_without_snapping_to_a_preset_angle(self):
        smoother = AdaptiveAngleFilter()
        raw = 43.7 + np.tile([-1., 1.], 90)
        filtered = [smoother.update(value, i / 30) for i, value in enumerate(raw)]
        self.assertLess(np.std(filtered[60:]), .25 * np.std(raw[60:]))
        self.assertLess(abs(np.mean(filtered[60:]) - 43.7), .1)

    def test_tracks_fast_rotation_and_resets_after_a_gap(self):
        smoother = AdaptiveAngleFilter()
        raw = np.arange(0., 120., 3.)
        values = [smoother.update(value, i / 30) for i, value in enumerate(raw)]
        self.assertLess(np.mean(raw[-20:] - values[-20:]), 3.)
        smoother.reset()
        self.assertEqual(smoother.update(15., 10.), 15.)


class FixedGeometryTests(unittest.TestCase):
    def test_mirrored_scene_keeps_angle_and_draws_ray_above_base(self):
        config = dict(reference_size=[320,240],base_template_box=[180,180,280,193],
                      lower_edge=[[179,181],[279,181]],hinge=[179,180],body_length=100,
                      fixed_geometry=True,mirror_input=True,fps=30)
        image=cv2.flip(self.scene(),1)
        result=SideEdgeDetector(config,image).detect(image)
        self.assertTrue(result.valid,result.reason)
        self.assertAlmostEqual(result.angle_deg,60.,delta=3.)
        self.assertLess(result.upper[1,1],result.upper[0,1])
        self.assertGreater(result.lower[1,0],result.lower[0,0])

    def test_invalid_texture_only_frame_has_no_invented_observed_lower(self):
        image = np.random.default_rng(17).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        config = dict(reference_size=[320, 240], base_template_box=[40, 180, 140, 193],
                      lower_edge=[[140, 181], [40, 181]], hinge=[140, 180], body_length=100,
                      fixed_geometry=True)
        result = SideEdgeDetector(config, image).detect(image)
        self.assertFalse(result.valid)
        self.assertIsNone(result.observed_lower)
        self.assertIsNone(result.observed_upper)
        np.testing.assert_array_equal(result.hinge, [140, 180])

    @staticmethod
    def scene(dx=0, dy=0):
        image = np.full((240, 320, 3), 220, np.uint8)
        cv2.rectangle(image, (40 + dx, 180 + dy), (140 + dx, 190 + dy), (20, 20, 20), -1)
        cv2.line(image, (140 + dx, 176 + dy), (90 + dx, 90 + dy), (20, 20, 20), 7)
        return image

    def test_rays_share_constant_pivot_and_lower_direction_across_translation(self):
        config = dict(reference_size=[320, 240], base_template_box=[40, 180, 140, 193],
                      lower_edge=[[140, 181], [40, 181]], hinge=[140, 180], body_length=100,
                      fixed_geometry=True, fps=30)
        detector = SideEdgeDetector(config, self.scene())
        reference = None
        for dx, dy in [(0, 0), (3, 2), (5, 4), (4, 3)]:
            result = detector.detect(self.scene(dx, dy))
            self.assertTrue(result.valid, result.reason)
            np.testing.assert_allclose(result.hinge, [140, 180])
            np.testing.assert_array_equal(result.lower[0], result.upper[0])
            np.testing.assert_array_equal(result.hinge, result.lower[0])
            if reference is None:
                reference = result.lower.copy()
            np.testing.assert_array_equal(result.lower, reference)
            self.assertAlmostEqual(directed_angle(result.lower[1] - result.lower[0],
                                                  result.upper[1] - result.upper[0]),
                                   result.filtered_angle_deg)
            self.assertIsNotNone(result.observed_upper)
            np.testing.assert_allclose(result.registration_offset, [dx, dy], atol=1)
        blank = detector.detect(np.full((240, 320, 3), 220, np.uint8))
        self.assertFalse(blank.valid)
        self.assertIsNone(blank.filtered_angle_deg)
