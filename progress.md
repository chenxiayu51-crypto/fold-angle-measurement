# Progress

## Synchronized experiment 20261016_110453
- Final rerun completed with 22 required artifacts verified. verify_synchronized_experiment.py independently recomputed CSV metrics, compared all17605cleaned samples with raw keep-last values, reintegrated gyro within1.21e-12deg, checked ray angles, PNG encodings, input/source hashes and unique ordered peak matches, and reran all29tests/13subtests successfully. report.md includes final test record. Original inputs and earlier experiment outputs were preserved.
- Final offset -1.217160899s, a=1;9training/4withheld events; normalized correlation .17319→.97982.1051common valid reference frames; fixed complementary MAE.66946deg, provisionalAB1.04880deg, retainedB.90957deg. A-only gravity baseline45.86%coverage. Report explicitly limits these to uncalibrated projected reference/fixed-base data.
- Read the complete user specification and required project files. New video: 1280x720, 30fps,1421 decoded frames. Current scene's lower half points right; initial angle is around34degrees. Kept all historical outputs intact.
- Initial scene probe without mirror adaptation exposed rendered upper rays below the table and reduced coverage. Added a failing mirror regression, then optional canonical mirror support with output-coordinate reflection. Final visual coverage1118/1421, with explicit invalid frames and late camera-motion loss.
- Initial direct reuse of old IMU CLI was unsuitable because it assumes180degrees/five static seconds; preserved those diagnostics only under this experiment's _probes/invalid_legacy_initialization. Replaced the first synchronization probe with modular, tested synchronization/comparison and a reproducible driver. Initial code probing also exposed an incorrect correlation-tuple max call and accel/gyro pairing; those probe outputs are superseded.
- New tests cover known synthetic offset, missing visual data, rejected flat signals, periodic timing ambiguity, complementary angles, gyro masking invariance and mirrored geometry.29tests/13subtests passed before final experiment rerun.
- Driver uses integer native timestamps, gap-aware interpolation, raw-visual derivative, training-only offset fit with withheld event windows, explicit initial visual anchor, first1s IMU reference, common-support residuals and retained-input-only ablation. Acc-only gravity-plane normal independently supportsY so the A-only axis input does not depend on masked gyroB.
- Local sensitivity/quality checks include initial-bias window choice, one-frame timing perturbations, direction/speed/angle bins, repeated static holds and original-input hashes. No neural model or continuous visual feedback into IMU estimates.
- An attempted unnecessary cleanup command was rejected automatically; proceeded safely by overwriting only this experiment's generated files, preserving original inputs and all historical results. No cleanup permission is needed.

## 2026-10-16 Markdown progress report
- User requested a detailed project summary and explicitly chose Markdown instead of Word. Applied the research reporting and file-planning workflow; checked implementation, saved metrics, audit and verification records.
- Created docs/2026-10-16-fold-angle-progress-summary.md with 16 sections: goals, visual processing, fixed-pivot stability, IMU audit, raw-to-angle equations and gates, measured results, discrete/real-time distinction, handheld/fusion design, gyro-reduction research and experiment plan, artifact index and reproduction commands. Added a README entry.
- Kept completed fixed-scene experiments separate from proposed arbitrary-handheld, synchronized fusion and learned sensor reduction. Original inputs, code and experiment outputs unchanged. Existing test results are cited as historical evidence; no models or tests rerun for documentation.
- Documentation checks: the initial tool wrapper rejected Windows backslash escapes before execution, fixed using forward-slash paths. A numeric text check initially rejected typographic negative signs; adjusted comparison normalization without altering the reported values.
- Final documentation verification passed: UTF-8 document, 16 sections, all 26 linked files/images exist, all five windows' primary median/std and comparison medians agree with saved JSON, code fences balanced and README index present. Document is 41,617 bytes. No experiment tests rerun for this documentation-only change.

## 2026-10-15
- Read workflow and brainstorming skills; current work is input inspection and design assessment.
- Verified source video path and empty project state.
- Preparing video decoding for representative-frame inspection.
- Found existing OpenCV in the vggt310 environment after pip networking failed; cancelled the download.
- Decoded metadata and extracted 12 representative frames; inspected contact sheet plus individual frames at 4.93 and 6.60 s.
- Proposed two near-camera side edges and requested clarification asynchronously; no automated measurement implementation started.
- Prepared manual edge-selection illustration for geometric review.
- Wrote outputs/video_inspection/assessment.md with three candidate approaches, directed-angle definition, perspective limitations, and independent validation requirements.
- Verified 12 extracted frames, metadata, and both image artifacts. Awaiting edge-selection clarification; no trained detector, per-frame angle CSV, or accuracy claim produced.
- User confirmed the proposed two side edges and requested proceeding. Saved design and implementation plan; continuing inline without additional design approval gates.
- Ruling: use the existing empty repository on a codex/ branch rather than a new checkout because there is no base commit and all current artifacts belong to this task.
- Geometry and synthetic-edge tests failed before implementation, then passed. Added video-export integration tests and implemented the full pipeline; 10 tests pass (removed a test file-handle warning afterward).
- First real-video probe rejected 308 frames because LSD splits long lower rails at reflections. Root cause: matching full endpoints rather than the observed line location. Added a regression for partial lower rails and switched to matching line position and direction.
- User clarified that camera and lower body should be treated as fixed; local translation only compensates accidental recording movement. Updated runtime summary assumptions accordingly.
- A temporary diagnostic command had an f-string escaping syntax error; corrected the command, without changing detector behavior.
- Real-video second probe exposed two root causes: standard LSD refinement removes reflective rail support, and the same lower line could be selected twice. Added failing raw-frame and lower-only regression tests, fixed both, and reran successfully.
- Final rerun:535/545valid,10invalid. Inspected24overlays and full curve; independent reviewer reran12tests, decoded545output video frames, checked invalid semantics and confirmed no remaining blockers within the prototype scope.
- Documentation patch initially failed because a findings context did not match; reissued with corrected context.
- Final verification checks545CSV rows against545decoded annotated frames,30fps timestamps, recomputed edge angles, blank invalid values and no duplicate rails. Saved verification.json and6coarse manual projection checks. No physical-angle accuracy or held-out evaluation claimed.
- User requests stronger mechanical constraints: fixed lower direction and fixed common intersection, reduced jitter. Revision will preserve earlier results and export to outputs/fold_angle_fixed; raw observations and virtual rays will be explicitly separated.
- Fixed geometry and One Euro filter implemented after failing regression tests.16tests passed, including constant shared origin under translation, stationary noise suppression and quick rotation response, invalid reset and source-versus-stabilized coordinate export.
- First fixed-pivot run:544/545valid; all valid rows have identical pivot and constant lower ray. Inspected24overlays and full curve. Closed0–3s frame-change RMS: previous median .02264deg, new adaptive .01803deg; flat15.6–18.1s: .06030deg to .00779deg. Window standard deviations also retained in the stability report; these are stability diagnostics, not accuracy estimates.
- Independent review caught initialized lower geometry mislabeled as an observation on invalid fixed-mode frames. Added a failing texture-only image regression; corrected invalid paths to export the actual observation or None. Rerunning final outputs.
- Final fixed revision verified:17tests pass;545CSV rows and545decoded video frames;544valid, frame426invalid. Fixed pivot has zero coordinate variation and lower ray is identical; actual selected upper angles and filtered model-ray angles independently recomputed from exported coordinates.
- Saved stability_comparison.json/.png with both raw/filtered window means, standard deviations and frame-change RMS, plus90degree crossing response. Independent reviewer confirmed the observation export fix; no outstanding review findings.
- User confirmed upper and lower phone halves each provide three-axis acc and gyro. Wrote docs/fusion/dual_imu_visual_fusion_plan.md with the one-DOF fusion state, calibrated hinge-axis projections, observability limits, delayed visual updates, asynchronous sample schemas and independent validation sequence. This is a design artifact; no IMU data or fusion performance results exist yet.

## 2026-10-16
- Received sensor_log_2026-10-16_09-12-25.csv; 52,947 rows spanning 52.71 s, four sensor streams. Acc1/gyro1 have 26,471/6,180 duplicate timestamp records and 67/23 conflicting timestamps. Kept the last value per timestamp, retaining an explicit audit.
- User confirmed gyro rad/s and fixed main IMU / moving secondary IMU. Signal behavior supports acc1/gyro1 as secondary, Chinese-named streams as main. Two distinct timestamp domains mapped approximately using log row order; never represented as hardware synchronization.
- Added audit, tested one-DOF estimator and offline CLI. Five meaningful new tests initially failed for missing implementation, then passed; all prior visual tests also pass (22 tests, 13 subtests total). Fixed one raw Windows string syntax error before CLI execution.
- Generated primary fixed-base reconstruction using secondary IMU gyro plus gated gravity corrections, and a provisional dual-IMU comparison. Initial 180 degrees supplied by user's description; no endpoint snapping, scale fitting or visual data used.
- Visually checked angle_curve.png, independently reintegrated gyro to 4e-13 deg agreement, verified 4,354 ascending original integer timestamps and finite output angles, and verified unchanged input SHA-256. No physical truth or calibrated dual-IMU accuracy claimed.

## Principles walkthrough (2026-10-16)
- User requires concrete principles for every step, not just outputs. Added a 16-section Chinese Markdown walkthrough (21,428 characters), linked from README and the experiment report.
- Added experiment-local principles_examples.py/.json: recomputes a raw conflicting timestamp, selected line angle, One Euro, accepted/rejected complementary updates, frame-300 interpolation/residual and all fixed-base common-support metrics. Assertions passed against raw logs and saved CSVs.
- Verified UTF-8, balanced code/math fences, all 21 links, sync offset and rounded main metrics. Evidence: principles_documentation_verification.json. No algorithm parameter changes or experiment rerun.

## CSV-first explanation correction
- User could not follow prior report. Created docs/从原始CSV到下一条IMU数据_完整计算教程.md from exact raw header/lines to split streams, time conversion, interpolation, per-record calibration, parameter/state separation, three recursive updates and fresh-session initialization.
- Added trace_csv_to_angle.py, csv_to_angle_trace.csv and csv_to_angle_parameters.json. Recomputed all4085Acc/gyro/fused points and gate flags from raw CSV; max fused difference2.84e-13deg. This is explanation evidence, not a trained model or deployable universal calibration.

## Six-question diagram explanation (2026-10-18)
- Added docs/手机铰链轴与IMU角度计算_六个问题图解.md and seven reproducible figures, including real phone annotation, body/world axes, XZ basis, actual Acc projections, trapezoid area and vertical-hinge observability.
- Recomputed 9,417 raw-row clock pairs; 9,241 pass the <25ms bracket criterion. Coefficients reproduce infer_clock_map exactly. Documented four boundary-clamped pairs and diagnostic-only refit, without changing the historical estimator.
- Verified orthogonal basis and handedness e1 cross e2 = -h, Acc vector reconstruction and exact first worked phase; inspected all seven figures. Linked from README and prior CSV tutorial. Final document checks are recorded alongside numerical verification.
