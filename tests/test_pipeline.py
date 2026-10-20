import csv
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np

from measure_video import run_video


class ExportTests(unittest.TestCase):
    def test_fixed_pivot_export_keeps_common_origin_and_separates_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'shifted.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 15, (320, 240))
            self.assertTrue(writer.isOpened())
            for dx, dy in [(0, 0), (3, 2), (5, 3)]:
                image = np.full((240, 320, 3), 220, np.uint8)
                cv2.rectangle(image, (40+dx, 180+dy), (140+dx, 190+dy), (20, 20, 20), -1)
                cv2.line(image, (140+dx, 176+dy), (90+dx, 90+dy), (20, 20, 20), 7)
                writer.write(image)
            writer.release()
            config = dict(reference_size=[320, 240], base_template_box=[40, 180, 140, 193],
                          lower_edge=[[140, 181], [40, 181]], hinge=[140, 180], body_length=100,
                          fixed_geometry=True)
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config))
            summary = run_video(source, config_path, root / 'out')
            with (root/'out/angles.csv').open(encoding='utf-8-sig') as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                self.assertEqual(row['valid'], '1')
                self.assertEqual(float(row['lower_hinge_x']), 140)
                self.assertEqual(float(row['upper_hinge_x']), 140)
                self.assertEqual(float(row['upper_hinge_y']), 180)
                self.assertTrue(row['observed_upper_hinge_x'])
                self.assertEqual(row['geometry_coordinate_system'], 'stabilized_fixed_rays')
            self.assertGreater(float(rows[-1]['registration_dy']), 1)
            self.assertEqual(summary['fps'], 15)
            self.assertIn('One Euro', summary['smoothing'])

    def test_csv_video_frames_and_missing_angle_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'synthetic.avi'
            image = np.full((240, 320, 3), 220, np.uint8)
            cv2.rectangle(image, (40, 180), (140, 190), (20, 20, 20), -1)
            cv2.line(image, (42, 175), (138, 175), (20, 20, 20), 5)
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10, (320, 240))
            self.assertTrue(writer.isOpened())
            for i in range(5):
                writer.write(np.full_like(image, 220) if i == 2 else image)
            writer.release()
            config = dict(reference_size=[320, 240], base_template_box=[40, 180, 140, 193],
                          lower_edge=[[140, 181], [40, 181]], hinge=[140, 180], body_length=100)
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config))
            summary = run_video(source, config_path, root / 'out')
            with (root / 'out' / 'angles.csv').open(encoding='utf-8-sig') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(summary['processed_frames'], 5)
            self.assertEqual(len(rows), 5)
            self.assertEqual(float(rows[-1]['timestamp_s']), .4)
            self.assertEqual(rows[2]['valid'], '0')
            self.assertEqual(rows[2]['projected_angle_deg'], '')
            self.assertEqual(rows[2]['smoothed_projected_angle_deg'], '')
            self.assertTrue(rows[2]['failure_reason'])
            self.assertEqual(rows[0]['valid'], '1')
            cap = cv2.VideoCapture(str(root / 'out' / 'annotated.mp4'))
            count = 0
            while cap.read()[0]:
                count += 1
            cap.release()
            self.assertEqual(count, 5)

    def test_missing_input_fails_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                run_video(root / 'absent.mp4', root / 'config.json', root / 'out')
            self.assertFalse((root / 'out').exists())
