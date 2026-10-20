# Fold Angle Implementation Plan

> Execute inline with the executing-plans workflow. The user has approved the two-side-edge design and requested that it be built.

**Goal:** Measure the supplied video and deliver reproducible code, annotated video, per-frame CSV and a plot.

**Architecture:** Configured local base tracking anchors a constrained LSD upper-side-edge detector. Directed edge geometry produces projected angles; exports preserve invalid states and raw/smoothed values.

**Tech Stack:** Python, OpenCV, NumPy, matplotlib, unittest; existing vggt310 environment.

## Tasks

1. Create tests/test_geometry.py before implementation. Verify 0, 30, 90, 150 and 180 degrees, opposite endpoint ordering, zero-length rejection, and blank-image rejection. Run `python -m unittest discover -s tests -v`; confirm missing implementation fails. Implement fold_angle/geometry.py and fold_angle/detector.py; rerun to green.
2. Inspect line candidates and base-template tracking on the supplied frames. Put first-frame initialization and physical constraints in configs/sample_video.json. Add synthetic image detection tests before detector implementation. Implement image-supported candidate ranking and temporal selection in fold_angle/detector.py; preserve invalid outputs. Run the same unittest command.
3. Implement measure_video.py to decode sequentially and export outputs/fold_angle/{angles.csv,annotated.mp4,angle_curve.png,diagnostics.jpg,summary.json}. Reject unreadable input or unusable metadata and check writer initialization. Record frame index, frame timestamp, raw and smooth projected angle, validity, score, endpoints and failure reason. Include README.md and requirements.txt.
4. Run the full video. Inspect diagnostic overlays and the curve, investigate failure modes, and compare held-out manual projected directions on representative frames. Verify CSV/video frame counts and invalid-row semantics. Request one final independent code review as required by the review skill, fix substantive findings, rerun relevant checks, and deliver artifacts.

## Execution rulings

- Work in the existing initially empty repository on a codex/ branch; no commits or remote publication requested. Preserve prior inspection artifacts.
- Treat this as a current-video prototype, not a pretrained object detector. No angle ground truth is available.
- Do not repeatedly ask approval for design/plan files: the user has explicitly selected the proposed measurement design.
