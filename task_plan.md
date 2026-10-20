# Fold angle video assessment

## Synchronized experiment 20261016_110453
User supplied a complete execution specification and authorized implementation, actual runs, fixes, tests and a Chinese report; no new design-approval gate is needed.
16. Audit new files and inspect scene/streams. Status: complete. 1421 decoded frames; 1280x720/30fps. Main appears fixed; new lower ray points right. Actual decoder playback times agree with frame/fps, but are not exposure timestamps. New IMU18306raw/17605unique records. Input hashes unchanged.
17. Adapt visual initialization without changing historical outputs. Status: complete. Optional mirrored canonical processing fixes direction assumptions;1118/1421valid, invalid angles remain blank. Old default behavior passes regression tests.
18. Implement/test synchronization with gaps, event matching, held-out validation and ambiguity audit. Status: complete. Offset b=-1.217160899s, a=1; correlation .17319 to .97982,9training/4withheld events; no competing strong offset. No claim of independent millisecond precision.
19. Reconstruct IMU from verified static reference and axis; compare on common valid support; mask each gyro with leakage tests. Status: complete. Initial33.253755deg from synchronized raw video; first1s reference.1051common video frames: fixed complementary MAE.66946deg, provisional dual1.04880deg, retained moving gyro.90957deg. Only fixed gyro gravity baseline covers45.86%of available reference frames. No arbitrary-handheld claim.
20. Generate diagnostics, test and verify deliverables, write final Chinese report. Status: complete.22required artifacts verified;29tests/13subtests pass. Independent trapezoidal difference1.21e-12deg checks numerical consistency only. See outputs/experiment_20261016_110453/report.md and verification.json; verification entry verify_synchronized_experiment.py.

Date: 2026-10-15
Input: data/raw/243eca009e4449f611a92a5437cc4d66.mp4
Angle convention: closed = 0 degrees; flat = 180 degrees.

## Scope
Inspect the supplied video and propose an evidence-based measurement design using two physical edges. No trained detector or physical-angle accuracy is assumed.

## Phases
1. Inspect workspace and available video tooling. Status: complete
2. Extract and visually inspect representative frames. Status: complete
3. Record geometric assumptions, detection options, and a recommended design. Status: complete
4. Clarify intended edge selection before implementation. Status: complete
5. Implement geometry, constrained side-edge detection, and video exports. Status: complete
6. Verify real-video overlays, data integrity, and independent review. Status: complete

## Result
- outputs/fold_angle/: annotated video,535valid/545frame CSV, curve, diagnostics, summary, verification record and6frame manual projection sanity check.
-12geometry/detection/export tests pass; all545annotated frames decode; no identical upper/lower segments in valid rows.
- Physical-angle accuracy remains uncalibrated; no trained object detector was produced or claimed.

## Fixed-pivot stability revision
7. Freeze initialized lower direction and common virtual pivot; compensate input translation in display coordinates. Status: complete
8. Fuse compatible upper-rail directions and apply adaptive angle filtering; retain raw measurements and invalid states. Status: complete
9. Run video to a separate output folder; compare static-window jitter and verify identical pivots across frames. Status: complete

Revision result: outputs/fold_angle_fixed contains the default stable result.544/545valid; fixed pivot(240,148) has zero coordinate variation; lower ray identical across all rows.17tests pass. Static frame-change RMS falls20.4%closed and87.1%flat relative to the prior median-filtered result.90deg crossing response delays11–23ms in this clip. Prior output remains available.

## Dual-IMU fusion extension
10. Confirm sensor configuration and document the fusion model, synchronization protocol, calibration and offline validation sequence. Status: complete (design only).

User confirms one accelerometer and one gyroscope on each phone half. Proposed first prototype: single-DOF angle plus differential gyro bias, with timestamped vision and gated gravity updates. See docs/fusion/dual_imu_visual_fusion_plan.md. No IMU logs or synchronized frame timestamps received; fusion implementation and performance validation are future work.

## Tooling issues

## Progress summary documentation (2026-10-16)
14. Check code, recorded results and assumptions; write a detailed Chinese Markdown summary, as requested instead of Word. Status: complete.
15. Verify document encoding, artifact links, key numbers and completed-versus-planned claims. Status: complete. Report contains 16 numbered sections; all 26 linked files/images exist; numeric IMU window statistics match saved JSON, UTF-8 and code fences checked. README entry added. No experiment rerun.

Documentation tooling note: a resumed JavaScript tool call initially rejected unescaped Windows paths; corrected to forward-slash paths before any filesystem mutation.

## IMU log analysis (2026-10-16)
11. Audit supplied sensor CSV: duplicates, clocks, sampling and motion. Status: complete.
12. Estimate hinge angle from accelerometers and gyroscopes with explicit calibration assumptions; export curves and data. Status: complete (preliminary fixed-base reconstruction; provisional dual-IMU comparison).
13. Verify synthetic geometry/timing cases and actual results; document limitations and acquisition fixes. Status: complete.

Input: data/raw/sensor_log_2026-10-16_09-12-25.csv. User describes initially flat, closing, holding and repeated opening/closing. Initial inspection finds four streams and different clock domains, with many duplicate records on acc1/gyro1. Preserve the input.

Result: outputs/imu_20261016_091225 contains 4,354 angle rows, curves, audit and report. 22 tests plus 13 subtests pass. Independent gyro reintegration agrees within 4e-13 degrees; source SHA-256 preserved. User confirms rad/s and main fixed / secondary moving. Clock alignment and directed installed axes remain provisional; physical accuracy not established. A Windows path literal initially caused a unicodeescape syntax error in the new CLI report string; fixed with a raw string before producing final results.
- Default pip failed with proxy/TLS error; bundled pip also encountered TLS/download interruptions. Resolved video decoding through an existing OpenCV environment; cancelled installation.

## Principles walkthrough (2026-10-16)
- Explain each implemented stage and distinguish implemented filters from proposed Kalman/learning methods. Status: complete.
- Recompute real examples and verify links and metrics. Status: complete; 21 document links exist and five groups of numerical checks pass.

## CSV-first model explanation
- Rebuild explanation starting with raw columns and actual records; explain parameters, recursive state and reuse on future CSVs. Complete.
- Recompute every fixed-base step and verify tutorial. Complete; see csv_tutorial_verification.json.

## Six-question visual explanation (2026-10-18)
- Recompute CSV row-order clock pairs and inspect the true hinge-axis selection source. Complete.
- Explain h, body/world coordinates, e1/e2, gyro projection, trapezoidal integration and all Acc gates with phone figures and real numerical samples. Complete.
- Verify generated diagrams, raw-fit coefficients, basis geometry and documentation links. Complete; evidence in outputs/explanation_20261018/verification.json.
