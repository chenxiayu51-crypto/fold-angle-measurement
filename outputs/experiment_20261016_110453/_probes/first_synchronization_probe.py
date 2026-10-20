"""Analyze one independently started video/dual-IMU experiment.

This script reuses the project's timestamp audit and one-DOF estimator, but
does not assume the previous experiment's initial angle or sensor axis.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from analyze_imu_log import infer_clock_map, interpolate
from imu_angle import estimate_angle, read_log


def write_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_video_angles(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    t = np.array([float(r["timestamp_s"]) for r in rows], dtype=float)
    raw = np.array([float(r["projected_angle_deg"]) if r["projected_angle_deg"] else np.nan for r in rows])
    filt = np.array([float(r["smoothed_projected_angle_deg"]) if r["smoothed_projected_angle_deg"] else np.nan for r in rows])
    valid = np.isfinite(raw) & (np.array([int(r["valid"]) for r in rows]) > 0)
    return t, raw, filt, valid


def fill_short_gaps(t, values, valid, max_gap_s=0.55):
    out = np.full_like(values, np.nan, dtype=float)
    out[valid] = values[valid]
    good = np.flatnonzero(valid)
    for left, right in zip(good[:-1], good[1:]):
        if t[right] - t[left] <= max_gap_s:
            out[left : right + 1] = np.interp(t[left : right + 1], [t[left], t[right]], [values[left], values[right]])
    return out


def video_motion_signal(t, raw, valid):
    filled = fill_short_gaps(t, raw, valid)
    deriv = np.full_like(filled, np.nan)
    smooth = np.full_like(filled, np.nan)
    good = np.flatnonzero(np.isfinite(filled))
    if len(good) == 0:
        return smooth, deriv, filled
    breaks = np.flatnonzero(np.diff(good) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, len(good)]
    for start, end in zip(starts, ends):
        idx = good[start:end]
        if len(idx) < 7:
            continue
        n = len(idx)
        window = min(31, n if n % 2 else n - 1)
        window = max(7, window)
        if window > n:
            continue
        segment = filled[idx]
        smooth[idx] = savgol_filter(segment, window, 2, mode="interp")
        deriv[idx] = savgol_filter(segment, window, 2, deriv=1, delta=float(np.median(np.diff(t[idx]))), mode="interp")
    return smooth, deriv, filled


def resample_signal(t, values, query):
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.full_like(query, np.nan, dtype=float)
    return np.interp(query, t[finite], values[finite], left=np.nan, right=np.nan)


def resample_piecewise(t, values, query, max_gap_s=0.55):
    """Interpolate only within observed segments; do not bridge visual outages."""
    finite = np.flatnonzero(np.isfinite(values))
    out = np.full_like(query, np.nan, dtype=float)
    if len(finite) < 2:
        return out
    breaks = np.flatnonzero(np.diff(finite) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, len(finite)]
    for start, end in zip(starts, ends):
        ids = finite[start:end]
        if len(ids) < 2:
            continue
        m = (query >= t[ids[0]]) & (query <= t[ids[-1]]) & ((t[ids[-1]] - t[ids[0]]) <= max_gap_s or len(ids) > 8)
        if m.any():
            out[m] = np.interp(query[m], t[ids], values[ids])
    return out


def synchronization(t_video, video_rate, t_imu, imu_rate):
    # t_imu = a*t_video + b. Search clock scale only within a narrow range;
    # the evidence is insufficient for a free drift model.
    grid = np.arange(0.995, 1.00501, 0.00025)
    best = None
    for a in grid:
        for b in np.arange(-4.0, 4.001, 0.01):
            ti = a * t_video + b
            y = np.interp(ti, t_imu, imu_rate, left=np.nan, right=np.nan)
            m = np.isfinite(video_rate) & np.isfinite(y)
            if m.sum() < 300:
                continue
            x = video_rate[m]
            z = -y[m]
            x = (x - x.mean()) / (x.std() + 1e-12)
            z = (z - z.mean()) / (z.std() + 1e-12)
            corr = float(np.mean(x * z))
            if best is None or corr > best[0]:
                best = (corr, a, b, int(m.sum()))
    if best is None:
        raise RuntimeError("Unable to find a video/IMU overlap for synchronization")
    corr, a, b, count = best
    # Compare with a=1, b=0 and with the best offset but fixed scale.
    def corr_for(scale, offset):
        y = np.interp(scale * t_video + offset, t_imu, imu_rate, left=np.nan, right=np.nan)
        m = np.isfinite(video_rate) & np.isfinite(y)
        if m.sum() < 2:
            return None
        x = video_rate[m]; z = -y[m]
        return float(np.corrcoef(x, z)[0, 1]), int(m.sum())
    before = corr_for(1.0, 0.0)
    fixed_scale = corr_for(1.0, b) or (-1.0, 0)
    return {
        "model": "t_imu = a * t_video + b",
        "scale_estimated": False,
        "a_search_best": float(a),
        "a_used": 1.0,
        "offset_s": float(b),
        "sign_used": -1,
        "correlation_before": before[0] if before else None,
        "correlation_after": fixed_scale[0],
        "overlap_samples": int(fixed_scale[1]),
        "scale_search_best_correlation": float(corr),
        "scale_search_best_overlap_samples": count,
        "interpretation": "The fixed-scale offset is used because the short independently started capture does not support a reliable clock-drift estimate. The narrow scale search is retained as a diagnostic only.",
    }


def event_residuals(t_video, video_rate, t_imu, imu_rate, offset):
    def peaks(t, rate, threshold):
        finite = np.isfinite(rate)
        x = np.nan_to_num(rate, nan=0.0)
        distance = max(1, int(round(0.8 / np.median(np.diff(t)))))
        p_pos, _ = find_peaks(x, height=threshold, distance=distance)
        p_neg, _ = find_peaks(-x, height=threshold, distance=distance)
        return [(float(t[i]), 1) for i in p_pos] + [(float(t[i]), -1) for i in p_neg]
    vthr = max(35.0, float(np.nanpercentile(np.abs(video_rate), 78)))
    ithr = max(0.35, float(np.nanpercentile(np.abs(imu_rate), 78)))
    vp = sorted(peaks(t_video, video_rate, vthr))
    ip = sorted(peaks(t_imu, -imu_rate, ithr))
    used = set(); matches = []
    for tv, sign in vp:
        target = tv + offset
        candidates = [(abs(ti - target), j, ti, si) for j, (ti, si) in enumerate(ip) if j not in used and si == sign and abs(ti - target) <= 1.0]
        if candidates:
            _, j, ti, si = min(candidates)
            used.add(j)
            matches.append({"video_time_s": tv, "imu_time_s": ti, "direction": "opening" if sign > 0 else "closing", "residual_ms": (ti - target) * 1000.0})
    return {"video_threshold_deg_s": vthr, "imu_threshold_rad_s": ithr, "matches": matches, "match_count": len(matches)}


def local_offsets(t_video, video_rate, t_imu, imu_rate, centers):
    """Estimate offset independently around separated motion events."""
    grid = np.arange(-2.0, 0.001, 0.01)
    out = []
    for center in centers:
        best = None
        window = (t_video >= center - 1.5) & (t_video <= center + 1.5) & np.isfinite(video_rate)
        for b in grid:
            y = np.interp(t_video[window] + b, t_imu, -imu_rate, left=np.nan, right=np.nan)
            m = np.isfinite(y)
            if m.sum() < 20:
                continue
            corr = float(np.corrcoef(video_rate[window][m], y[m])[0, 1])
            if best is None or corr > best[0]:
                best = (corr, b)
        if best:
            out.append({"video_event_center_s": center, "local_offset_s": best[1], "correlation": best[0]})
    return out


def metrics(ref, estimate):
    m = np.isfinite(ref) & np.isfinite(estimate)
    if not m.any():
        return {"samples": 0, "mae_deg": None, "rmse_deg": None, "p95_abs_deg": None, "max_abs_deg": None, "median_bias_deg": None}
    e = estimate[m] - ref[m]
    ae = np.abs(e)
    return {"samples": int(m.sum()), "mae_deg": float(np.mean(ae)), "rmse_deg": float(np.sqrt(np.mean(e * e))), "p95_abs_deg": float(np.percentile(ae, 95)), "max_abs_deg": float(np.max(ae)), "median_bias_deg": float(np.median(e))}


def plot(path: Path, x, ys, labels, xlabel, ylabel, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 5), dpi=150)
    for y, label in zip(ys, labels):
        ax.plot(x, y, linewidth=1.0 if "raw" in label.lower() else 1.6, label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(alpha=0.22); ax.legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-angles", required=True)
    parser.add_argument("--imu", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output); (out / "synchronization").mkdir(parents=True, exist_ok=True); (out / "comparison").mkdir(parents=True, exist_ok=True); (out / "ablation").mkdir(parents=True, exist_ok=True); (out / "imu").mkdir(parents=True, exist_ok=True)
    tv, vr, vf, vvalid = load_video_angles(Path(args.video_angles))
    vs, vd, vfilled = video_motion_signal(tv, vr, vvalid)
    streams = read_log(args.imu)
    # Infer moving stream and axis from this recording, rather than reusing old labels.
    gyro_names = [name for name in streams if "gyro" in name.lower() or "陀螺" in name]
    scores = {name: float(np.max(np.std(streams[name].values, axis=0))) for name in gyro_names}
    moving_name = max(scores, key=scores.get)
    fixed_names = [name for name in gyro_names if name != moving_name]
    fixed_name = fixed_names[0] if fixed_names else None
    moving = streams[moving_name]
    axis_index = int(np.argmax(np.std(moving.values, axis=0)))
    axis = np.zeros(3); axis[axis_index] = 1.0
    tm = (moving.timestamps_ns - moving.timestamps_ns[0]) / 1e9
    bias = np.median(moving.values[tm < min(2.0, tm[-1] / 4), axis_index])
    imu_rate = moving.values[:, axis_index] - bias
    # Map the other clock domain approximately for the physical estimators.
    names = list(streams)
    if fixed_name and fixed_name.startswith(("加速度", "陀螺")):
        sensor_base = min(int(streams[n].timestamps_ns[0]) for n in names if n.startswith(("加速度", "陀螺")))
        host_base = min(int(streams[n].timestamps_ns[0]) for n in names if n in ("acc1", "gyro1"))
        clock = infer_clock_map(args.imu, host_base, sensor_base)
        tf = clock["scale"] * ((streams[fixed_name].timestamps_ns - sensor_base) / 1e9) + clock["offset_s"]
    else:
        tf = (streams[fixed_name].timestamps_ns - streams[fixed_name].timestamps_ns[0]) / 1e9 if fixed_name else None
        clock = None
    if fixed_name:
        fixed_acc_name = next((n for n in streams if n != moving_name and "acc" in n.lower()), None)
        # Pair acc1 with gyro1 because they share the same timestamp domain;
        # retain this as an inference in the report rather than a hard-coded
        # physical identity.
        moving_acc_name = "acc1" if "acc1" in streams else next((n for n in streams if "acc" in n.lower()), None)
        # Main Chinese accelerometer is fixed in the existing naming pattern; verify by gravity norm.
        acc_candidates = [n for n in streams if "acc" in n.lower() or "加速度" in n]
        norms = {n: float(np.median(np.linalg.norm(streams[n].values, axis=1)) ) for n in acc_candidates}
        moving_acc_name = min(acc_candidates, key=lambda n: abs(norms[n] - 9.81)) if acc_candidates else None
        fixed_acc_candidates = [n for n in acc_candidates if n != moving_acc_name]
        fixed_acc_name = fixed_acc_candidates[0] if fixed_acc_candidates else None
    else:
        moving_acc_name = fixed_acc_name = None; norms = {}; clock = None
    # For the current recording, gyro1 is the moving signal and the Chinese gyro is the near-static base.
    if fixed_name and fixed_name.startswith(("加速度", "陀螺")):
        base_start = min(int(streams[n].timestamps_ns[0]) for n in streams if n.startswith(("加速度", "陀螺")))
        base_times = {n: clock["scale"] * ((streams[n].timestamps_ns - base_start) / 1e9) + clock["offset_s"] for n in streams if n.startswith(("加速度", "陀螺"))}
        # The Chinese gyro stream is the fixed gyro; base acceleration is the other acceleration stream.
        base_gyro_name = next((n for n in streams if "陀螺" in n), fixed_name)
        base_acc_name = next((n for n in streams if "加速度" in n), fixed_acc_name)
        base_gyro = np.column_stack([np.interp(tm, base_times[base_gyro_name], streams[base_gyro_name].values[:,j]) for j in range(3)])
        base_acc = np.column_stack([np.interp(tm, base_times[base_acc_name], streams[base_acc_name].values[:,j]) for j in range(3)])
    else:
        base_gyro = base_acc = None
    sync = synchronization(tv, vd, tm, imu_rate)
    b = sync["offset_s"]; a = sync["a_used"]
    sync["local_event_offsets"] = local_offsets(tv, vd, tm, imu_rate, [3.7, 7.7, 12.6, 14.8, 29.7, 35.3])
    tv_for_imu = (tm - b) / a
    visual_at_imu = resample_piecewise(tv, vf, tv_for_imu)
    nearest = np.interp(tv_for_imu, tv, np.where(vvalid, tv, np.nan), left=np.nan, right=np.nan) if False else None
    # Initial angle is taken from a static visual frame near the first IMU time; it is not assumed to be 180 degrees.
    initial_candidates = vf[np.isfinite(vf) & (tv >= max(0, tv_for_imu[0]))]
    initial_angle = float(initial_candidates[0]) if len(initial_candidates) else 0.0
    moving_acc = moving.values if moving_acc_name == moving_name else streams[moving_acc_name].values
    # Moving accelerometer shares the moving clock domain in this log.
    moving_acc_t = (streams[moving_acc_name].timestamps_ns - moving.timestamps_ns[0]) / 1e9
    moving_acc = interpolate(moving_acc_t, streams[moving_acc_name].values, tm)
    fixed = estimate_angle(tm, moving_acc, moving.values, axis=axis, calibration_s=min(2.0, tm[-1] / 4), initial_angle_deg=initial_angle)
    if base_gyro is not None:
        dual = estimate_angle(tm, moving.values, moving.values, base_acc=base_acc, base_gyro=base_gyro, axis=axis, base_axis=axis, calibration_s=min(2.0, tm[-1] / 4), initial_angle_deg=initial_angle)
        keep_b = estimate_angle(tm, moving.values, moving.values, base_acc=base_acc, axis=axis, calibration_s=min(2.0, tm[-1] / 4), initial_angle_deg=initial_angle)
        zero = np.zeros_like(moving.values)
        keep_a = estimate_angle(tm, moving.values, zero, base_acc=base_acc, base_gyro=base_gyro, axis=axis, base_axis=axis, calibration_s=min(2.0, tm[-1] / 4), initial_angle_deg=initial_angle)
    else:
        dual = fixed; keep_a = keep_b = fixed
    write_json(out / "synchronization" / "time_alignment.json", {**sync, "moving_gyro": moving_name, "fixed_gyro": fixed_name, "moving_axis_index": axis_index, "moving_axis_unit_vector": axis.tolist(), "initial_angle_deg_from_video": initial_angle, "event_alignment": event_residuals(tv, vd, tm, imu_rate, b), "clock_audit": clock})
    # Export cleaned moving-domain IMU and synchronized angle comparison.
    cleaned = out / "imu" / "cleaned_imu.csv"
    fields = ["time_imu_s", "time_video_s", "moving_acc_x", "moving_acc_y", "moving_acc_z", "moving_gyro_x", "moving_gyro_y", "moving_gyro_z", "base_acc_x", "base_acc_y", "base_acc_z", "base_gyro_x", "base_gyro_y", "base_gyro_z"]
    with cleaned.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(fields)
        for k in range(len(tm)):
            w.writerow([tm[k], tv_for_imu[k], *moving_acc[k], *moving.values[k], *(base_acc[k] if base_acc is not None else [np.nan]*3), *(base_gyro[k] if base_gyro is not None else [np.nan]*3)])
    # Common comparison table. Long visual invalid gaps are excluded from metrics, but rows are retained.
    estimates = {"visual_projected_deg": visual_at_imu, "gyro_integrated_deg": fixed["gyro_angle_deg"], "fixed_base_fused_deg": fixed["angle_deg"], "dual_imu_deg": dual["angle_deg"], "keep_fixed_gyro_deg": keep_a["angle_deg"], "keep_moving_gyro_deg": keep_b["angle_deg"]}
    with (out / "comparison" / "aligned_angles.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["time_imu_s", "time_video_s", *estimates.keys()])
        for k in range(len(tm)): w.writerow([tm[k], tv_for_imu[k], *[v[k] for v in estimates.values()]])
    metrics_raw = {name: metrics(visual_at_imu, vals) for name, vals in estimates.items() if name != "visual_projected_deg"}
    write_json(out / "comparison" / "metrics.json", {"reference": "visual_projected_deg, uncalibrated 2D projection; not mechanical truth", "metrics_raw": metrics_raw, "visual_valid_overlap_samples": int(np.isfinite(visual_at_imu).sum()), "note": "No angle shift, scale fit or endpoint snapping is applied."})
    # Plots.
    plot(out / "synchronization" / "angular_velocity_overlay.png", tv, [vd, np.interp(tv, (tm - b) / a, -imu_rate, left=np.nan, right=np.nan)], ["visual d(angle)/dt", "-moving gyro projected"], "Video time (s)", "Angular rate", "Data-driven video/IMU synchronization")
    plot(out / "comparison" / "angle_overlay.png", tv_for_imu, [visual_at_imu, fixed["gyro_angle_deg"], fixed["angle_deg"], dual["angle_deg"]], ["Visual projected angle", "Gyro integration", "Fixed-base fused", "Provisional dual IMU"], "Video-aligned time (s)", "Angle (deg)", "Synchronized angle comparison")
    plot(out / "comparison" / "angle_residuals.png", tv_for_imu, [fixed["gyro_angle_deg"] - visual_at_imu, fixed["angle_deg"] - visual_at_imu, dual["angle_deg"] - visual_at_imu], ["Gyro - visual", "Fixed fused - visual", "Dual - visual"], "Video-aligned time (s)", "Residual (deg)", "Residual relative to visual projection")
    # Error grouped by the visual opening angle and by visual angular speed.
    vis_speed = resample_piecewise(tv, vd, tv_for_imu)
    method_values = {"Gyro integration": fixed["gyro_angle_deg"], "Fixed-base fused": fixed["angle_deg"], "Dual IMU": dual["angle_deg"], "Keep moving gyro": keep_b["angle_deg"], "Keep fixed gyro": keep_a["angle_deg"]}
    angle_edges = np.arange(0, 181, 30); speed_edges = np.array([0, 10, 30, 60, 120, np.inf])
    def grouped(edges, variable):
        result = {"edges": edges.tolist(), "methods": {}}
        for label, vals in method_values.items():
            err = np.abs(vals - visual_at_imu)
            groups=[]
            for lo, hi in zip(edges[:-1], edges[1:]):
                m=np.isfinite(visual_at_imu)&np.isfinite(err)&(variable>=lo)&(variable<hi)
                groups.append({"low":float(lo),"high":None if not np.isfinite(hi) else float(hi),"samples":int(m.sum()),"mae_deg":None if not m.any() else float(np.mean(err[m]))})
            result["methods"][label]=groups
        return result
    angle_groups=grouped(angle_edges,visual_at_imu); speed_groups=grouped(speed_edges,np.abs(vis_speed))
    write_json(out / "comparison" / "error_by_angle.json", angle_groups); write_json(out / "comparison" / "error_by_speed.json", speed_groups)
    def grouped_plot(path, grouped_data, title, x_label):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels=list(grouped_data["methods"]); centers=np.arange(len(grouped_data["edges"])-1); width=.8/max(1,len(labels)); fig,ax=plt.subplots(figsize=(12,5),dpi=150)
        for j,label in enumerate(labels):
            vals=[g["mae_deg"] if g["mae_deg"] is not None else np.nan for g in grouped_data["methods"][label]]
            ax.bar(centers+(j-(len(labels)-1)/2)*width,vals,width,label=label)
        names=[f'{a:g}–{b:g}' if np.isfinite(b) else f'{a:g}+' for a,b in zip(grouped_data["edges"][:-1],grouped_data["edges"][1:])]
        ax.set_xticks(centers,names); ax.set(xlabel=x_label,ylabel='MAE relative to visual projection (deg)',title=title); ax.grid(axis='y',alpha=.2); ax.legend(fontsize=8,ncol=2); fig.tight_layout(); fig.savefig(path); plt.close(fig)
    grouped_plot(out / "comparison" / "error_by_angle.png",angle_groups,"Residual by opening-angle bin","Visual projected angle bin (deg)")
    grouped_plot(out / "comparison" / "error_by_speed.png",speed_groups,"Residual by opening-speed bin","Visual |d angle/dt| bin (deg/s)")
    baseline_rows=[]
    for label,vals in method_values.items(): baseline_rows.append([label,metrics(visual_at_imu,vals)["samples"],metrics(visual_at_imu,vals)["mae_deg"],metrics(visual_at_imu,vals)["rmse_deg"],metrics(visual_at_imu,vals)["p95_abs_deg"],metrics(visual_at_imu,vals)["max_abs_deg"]])
    with (out / "ablation" / "baseline_comparison.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["configuration","samples","mae_deg","rmse_deg","p95_abs_deg","max_abs_deg"]); w.writerows(baseline_rows)
    write_json(out / "ablation" / "metrics.json", {"reference":"visual projected angle; no mechanical truth","configurations":baseline_rows,"interpretation":"Keep-fixed-gyro is a fixed-base stress test, not evidence for arbitrary handheld operation."})
    plot(out / "ablation" / "baseline_comparison.png", np.arange(len(baseline_rows)), [np.array([r[2] for r in baseline_rows]),np.array([r[3] for r in baseline_rows])], ["MAE","RMSE"], "Configuration index", "Error (deg)", "Single-gyro ablation comparison")
    # A compact synchronization diagnostic with local offsets and correlations.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    local=sync["local_event_offsets"]; fig,ax1=plt.subplots(figsize=(10,4),dpi=150); xx=[d["video_event_center_s"] for d in local]; yy=[d["local_offset_s"] for d in local]; cc=[d["correlation"] for d in local]; ax1.plot(xx,yy,"o-",label="Local offset"); ax1.axhline(b,color="k",ls="--",label=f"Global {b:.2f}s"); ax1.set(xlabel="Video event center (s)",ylabel="t_imu - t_video (s)",title="Synchronization diagnostics"); ax2=ax1.twinx(); ax2.plot(xx,cc,"s--",color="#c44",label="Local correlation"); ax2.set_ylabel("Correlation"); ax1.grid(alpha=.2); fig.tight_layout(); fig.savefig(out / "synchronization" / "synchronization_diagnostics.png"); plt.close(fig)
    # Copy diagnostics under the requested PNG name without altering the prior detector output.
    diag = Path(args.video_angles).parent / "diagnostics.jpg"
    if diag.is_file(): shutil.copyfile(diag, out / "video" / "diagnostics.png") if (out / "video").exists() else None
    write_json(out / "imu" / "summary.json", {"source": str(Path(args.imu).resolve()), "moving_gyro": moving_name, "fixed_gyro": fixed_name, "moving_axis_index": axis_index, "initial_angle_deg": initial_angle, "acc_norm_medians": norms, "clock": clock, "output_rows": len(tm), "note": "Current physical estimator is initialized from the synchronized visual projection and is therefore not an independent angle truth."})
    print(json.dumps({"moving_gyro": moving_name, "fixed_gyro": fixed_name, "axis_index": axis_index, "initial_angle_deg": initial_angle, "sync": sync, "metrics": metrics_raw}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
