# Fixed-pivot stability implementation

User request: lower body is mechanically fixed; reduce jitter, preferably fix the intersection of the two angle rays. This explicitly authorizes implementing the revision to the existing prototype.

1. Add regression tests for fixed lower direction, identical ray origins across translation, angle filter static noise suppression and fast motion response, invalid reset, and absence of false closure.
2. Extend the detector with a fixed-geometry mode. The configured lower rail initializes a fixed direction; observed lower rail is used for image checks only. Keep the virtual hinge and ray lengths fixed in stabilized coordinates. Raw LSD observations remain separate from modeled rays. Do not force finite-thickness physical rail offsets to intersect in image space.
3. Add a One Euro angle filter: derivative-adaptive cutoff for strong stationary smoothing and fast motion response. Preserve raw and filtered angles separately; reset after invalid frames. Fuse only directions compatible with the selected upper rail.
4. In the video pipeline, stabilize by the observed base translation before drawing the constant pivot, show filtered modeled rays and the observed upper segment, and export the coordinate/angle distinctions. Default to outputs/fold_angle_fixed, preserving previous outputs/fold_angle.
5. Run the tests and all545frames, inspect overlays, verify constant lower ray and shared pivot, and compare jitter in closed0–3s and flat15.6–18.1s windows. These are image stability diagnostics, not mechanical-angle accuracy metrics.

Use the existing workspace and codex/fold-angle branch; no new checkout, publication or additional approval is needed. Independent final review follows the requesting-code-review skill.
