# Findings

## Synchronized experiment 20261016_110453 (final)
- Camera geometry is horizontally reversed relative to the original scene. Explicit mirror canonicalization preserves directed 0–180degree geometry and correct ray rendering. New pivot(540,305),1118/1421valid. Late camera movement causes missing observations; no generic camera-rotation tracker is claimed.
- New raw CSV18306rows,17605unique sensor events; gyro1 duplicates375/conflicts9,acc1 duplicates326/conflicts6. Largest acc1 gap91ms and gyro1 gap52ms; native integer times retained. Two clock domains remain approximately software-mapped.
- AccB-only gravity plane normal[.002375,.999995,-.001836] and gyro PCA[.002046,.999965,-.008168] independently supportY; gyro PC energy99.9319%. Candidate A-only axis input is based on retained acceleration, without deriving it from the missing gyro.
- Synchronization uses raw-visual local polynomial velocity versus bias-corrected axial gyro; training windows exclude held-out windows even in coarse search.9training/4validation events, b=-1.217160899s,a=1,common-support correlation .17319→.97982. Strong alternate cycle offsets are lower; no scale/drift or calibrated time-uncertainty claim.
- First1s biasB[.00345,.0002,-.00085]rad/s; video-derived initial33.253755deg. Fixed complementary MAE.66946deg/RMSE1.04049deg; gyro.92694/1.22182; provisionalAB1.04880/1.42307; retainedB.90957/1.27811 on1051identical raw visual frames. Visual is uncalibrated projection, and initialization shares that visual anchor.
- OnlyA gyro baseline is acc gravity phase under conservative low dynamics, not zero-filled missing-B gyro integration.482valid reference frames,45.86%coverage; all-config common comparison reported separately. Arbitrary-handheld sensor removal remains unvalidated.
- Report, all22requested artifacts, supplemental direction/static/bias/time sensitivity diagnostics, manifest and verification generated.29tests/13subtests passed; input hashes unchanged and native cleaned values independently matched to last raw record. No neural training or continuous visual-IMU fusion performed.

- Workspace initially contains only .git; no existing implementation or AGENTS.md found in workspace/ancestors.
- Supplied MP4 exists, size 1,736,651 bytes.
- Neither the default Python nor bundled Python initially includes OpenCV.
- Default Python pip failed with a proxy/TLS error. Trying bundled Python in an isolated temporary dependency directory.
- Two image lines alone do not generally identify a physical hinge angle under arbitrary perspective. Edge identity, direction, camera pose, and visibility must be checked.
- Successfully decoded using existing python; no new package installation needed. Bundled pip attempt was interrupted after TLS/download problems.
- Video metadata: 480x256, 30 fps, 545 frames, nominal duration 18.1667 s; last-frame timestamp 18.1333 s.
- Twelve uniformly sampled frames show a near-side camera, lower body resting on a table, hinge toward the right, upper body swinging from left (closed) through upright to right (flat).
- Hands occlude portions of the upper body; protective case, reflections, thickness, and background edges can confuse generic line detectors.
- Use two directed vectors from hinge toward free ends; an undirected acute line-angle calculation folds angles above 90 degrees and is unsuitable.
- Full closure has finite body thickness and distinct rail offsets; a shared virtual hinge/centerline and endpoint-state handling are preferable to literal line intersection.
- Saved contact sheet and original sampled frames under outputs/video_inspection/. The edge-selection illustration uses manual approximate points and is not an automatic detector result or angle ground truth.
- Final implementation uses separate observed rail directions, not a forced common intersection. User confirmed fixed-camera/fixed-base assumptions; small accidental translation is compensated locally.
- Raw-frame evidence showed LSD standard refinement truncating/removing reflective upper rails (e.g. frame297 from134px to76px). Using unrefined LSD plus physical constraints restored supported upper candidates.
- Independent review reproduced duplicate lower/upper selection producing false0deg. Added a lower-only regression and require a nearclosed upper rail at least2px above the lower rail. No identical observed upper/lower segments remain in final valid rows.
- Final run:545frames,535valid,10invalid. Nearclosed~0deg, final flat~179.4–179.7deg. Valid coverage is98.17%, not an accuracy estimate.
- Fixed-pivot revision: fixed_geometry=true, frozen lower direction initialized from first-frame rail, constant virtual pivot(240,148), upper-direction fusion within2.5deg and adaptive One Euro filter(mincutoff .8Hz,beta .1). Frame registration removes observed translation before visualization; observed segments remain separately recorded in original coordinates.
- Revision outputs/fold_angle_fixed:544/545valid; only frame426invalid. Modeled lower ray identical in all545rows; common origin identical on all544valid frames; modeled ray angles reproduce filtered CSV; selected observed upper directions reproduce single-edge angle CSV.
- Stationary-window frame-change RMS against prior median output: closed0–3s .02264→.01803deg(20.4%reduction); flat15.6–18.1s .06030→.00779deg(87.1%). Full-window standard deviations and means also saved; these include slow drift and do not all improve. No physical accuracy claim.
- New raw-to-filtered90deg crossing delays11.4–22.6ms across5opening/closing crossings; this measures filter response within the clip, not timing against external ground truth.
# Dual-IMU fusion design extension (2026-10-15)

- Confirmed: both halves have acc + gyro. First fusion prototype can use angle and differential gyro bias under the one-DOF hinge constraint.
- Needs synchronized samples, calibrated hinge axes and absolute initialization. Gravity becomes uninformative when the hinge aligns with gravity; dynamic acceleration requires gating.
- Existing visual measurements are uncalibrated projected angles. Use geometrically calibrated, temporally unfiltered measurements for fusion; retain One Euro filtered output for display.
- No IMU logs, physical reference measurements or fusion evaluation were provided. Design and collection protocol are recorded in docs/fusion/dual_imu_visual_fusion_plan.md.

## IMU capture findings (2026-10-16)
- Supplied 52.71 s capture shows two substantial closing/opening cycles. Strong motions approximately 15.1–17.5 s, 30.3–32.1 s, 35.9–38.6 s and 47.6–50.3 s; brief repositioning occurs separately.
- Gyro unit rad/s confirmed. Main IMU fixed / secondary moving confirmed. Acc1/gyro1 inferred secondary; Chinese names inferred main. Motion PCA axis approximately [0.0046, 0.99996, -0.00715], first component 99.84% gyro motion energy. This supports +Y but is not installation calibration.
- Distinct timestamp domains: ~1.79e18 ns versus ~4.75e13 ns. Software row-order fit residual 5th/95th percentiles about -9/+12 ms; this is not an independently established time-error bound.
- Deduplicate acc1 from 31,517 to 5,046 timestamps; gyro1 from 10,536 to 4,356. 67/23 timestamps contain different values, losing sample timing. Main IMU acc/gyro counts 5,627/5,267 are unique.
- Fixed-base fused median angles in chosen windows: initial flat 179.996°, first closed 0.795°, middle flat 180.205°, second closed 0.811°, final flat 178.936°. The initial angle is imposed from user information; no independent accuracy conclusion.
- Pure gyro second closed median 2.588° versus fused 0.811°, but this is an algorithm comparison, not a ground-truth error reduction measurement. Base motion and uncalibrated gravity references shift the provisional dual estimate by about a degree in closed holds; keep it separate from primary fixed-base result.


## Principles walkthrough evidence (2026-10-16)
- At IMU0.013s: raw Gyro-B Y .0418/.0349, bias .0002rad/s, dt .013s; gyro prediction33.225339158deg, gravity32.882110594deg, gain.012915865; fused33.220906064deg.
- At2.003s corrected gyro norm.424376rad/s exceeds.25; Acc correction rejected and gyro prediction36.277015203deg retained.
- Video frame300 at10s maps toIMU8.782839101s; fixed estimate67.277953519deg minus raw projection67.61271deg yields-.334756481deg residual.
- Current estimator is gated complementary correction, not Kalman; vision provides shared initialization and evaluation, not ongoing fusion. Future arbitrary-handheld gyro removal remains unproven.

## Clock and coordinate explanation evidence (2026-10-18)
- Clock map uses raw write-order neighboring B rows, not hardware matched samples: 9,417 pairs, 9,241 selected, 176 excluded, four boundary-clamped initial A rows included. Excluding these four changes offset by about -0.00930ms (diagnostic only).
- Synced experiment chooses B axis from the dominant coordinate of the Acc rotation-plane normal (0.0023750,0.9999955,-0.0018363), then checks Gyro principal direction. This is offline axis inference and coordinate-axis simplification, not photographic installation calibration. A axis remains provisional.
- e1/e2 are fixed triples in B body coordinates and move with B in the world. Static Acc is specific force, opposite downward physical gravity. With plot +X right and +Z up, +Y is into the page. Existing basis has e1 cross e2 = -h.
