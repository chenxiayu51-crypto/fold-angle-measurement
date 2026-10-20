"""Explain this experiment by independently tracing raw CSV -> each fixed-base update.

This is an experiment audit, not a saved learned model or a new deployment API.
Only csv_to_angle_trace.csv and csv_to_angle_parameters.json are written.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[1]))
from imu_angle import read_log
from fold_angle.synchronization import gap_interp


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    meta = json.loads((OUT/'imu/summary.json').read_text(encoding='utf-8'))
    source = meta['source']
    streams = read_log(source)
    original = read_csv(source)
    last_line = {(r['sensor'], int(r['timestamp_ns'])): k+2 for k, r in enumerate(original)}
    clock = meta['clock_map']
    tg = (streams['gyro1'].timestamps_ns-clock['host_base_ns'])/1e9
    ta = (streams['acc1'].timestamps_ns-clock['host_base_ns'])/1e9
    saved = read_csv(OUT/'imu/angles.csv')
    ns = np.array([int(r['timestamp_ns']) for r in saved], dtype=np.int64)
    t = (ns-clock['host_base_ns'])/1e9
    g = streams['gyro1'].values[np.searchsorted(streams['gyro1'].timestamps_ns, ns)]
    a = gap_interp(ta, streams['acc1'].values, t, .1)
    bias = np.median(streams['gyro1'].values[(tg >= 0)&(tg < 1)], axis=0)
    assert np.allclose(bias, meta['bias_B_xyz_rad_s'], atol=1e-14)
    h = np.array(meta['axis_b'])
    ref = np.median(a[t < 1], axis=0)
    gref = np.linalg.norm(ref)
    p0 = ref-ref@h*h
    e1 = p0/np.linalg.norm(p0)
    e2 = -np.cross(h, e1)
    u = a@e1
    v = a@e2
    phase = np.arctan2(v, u)
    obs = np.linalg.norm(a-(a@h)[:, None]*h, axis=1)/gref
    corrected = g-bias
    rate = corrected@h
    jerk = np.r_[0, np.linalg.norm(np.diff(a, axis=0), axis=1)/np.diff(t)]
    initial = meta['initial_angle_deg']
    previous = initial
    integral = initial
    assert np.diff(t).max() <= .1, 'This worked trace expects no long gyro outage.'
    rows = []
    interpolation = None
    for k in range(len(t)):
        dt = t[k]-t[k-1] if k else 0.
        delta = -.5*(rate[k-1]+rate[k])*dt*180/np.pi if k else 0.
        predicted = previous+delta
        integral += delta
        observation = initial-phase[k]*180/np.pi
        observation += 360*np.round((predicted-observation)/360)
        innovation = observation-predicted
        norm_ok = abs(np.linalg.norm(a[k])-gref) < .6
        proj_ok = obs[k] > .2
        jerk_ok = jerk[k] < 30
        gyro_ok = np.linalg.norm(corrected[k]) < .25
        innovation_ok = abs(innovation) < 20
        quality = norm_ok and proj_ok and jerk_ok
        used = quality and gyro_ok and innovation_ok
        gain = 1-np.exp(-dt)
        correction = gain*innovation if used else 0.
        result = predicted+correction
        assert abs(result-float(saved[k]['fixed_base_imu_angle_deg'])) < 1e-9
        assert abs(integral-float(saved[k]['gyro_integrated_angle_deg'])) < 1e-9
        assert abs(observation-float(saved[k]['acc_angle_deg'])) < 1e-9
        assert int(used) == int(saved[k]['acc_update_used'])
        rows.append({'index': k, 'raw_gyro_last_csv_line': last_line['gyro1', int(ns[k])],
            'timestamp_ns': int(ns[k]), 'time_s': t[k], 'dt_s': dt,
            'gyro_x_rad_s': g[k, 0], 'gyro_y_rad_s': g[k, 1], 'gyro_z_rad_s': g[k, 2],
            'acc_x_m_s2': a[k, 0], 'acc_y_m_s2': a[k, 1], 'acc_z_m_s2': a[k, 2],
            'axial_rate_rad_s': rate[k], 'previous_fused_deg': previous,
            'gyro_increment_deg': delta, 'prediction_deg': predicted,
            'acc_dot_e1': u[k], 'acc_dot_e2': v[k], 'acc_phase_rad': phase[k],
            'acc_angle_deg': observation, 'acc_norm_m_s2': np.linalg.norm(a[k]),
            'projected_fraction': obs[k], 'jerk_m_s3': jerk[k],
            'corrected_gyro_norm_rad_s': np.linalg.norm(corrected[k]),
            'norm_gate': int(norm_ok), 'projection_gate': int(proj_ok), 'jerk_gate': int(jerk_ok),
            'gyro_gate': int(gyro_ok), 'innovation_gate': int(innovation_ok),
            'acc_used': int(used), 'innovation_deg': innovation, 'gain_if_used': gain,
            'applied_correction_deg': correction, 'final_fused_deg': result,
            'pure_gyro_deg': integral, 'saved_fused_deg': float(saved[k]['fixed_base_imu_angle_deg'])})
        previous = result
        j = np.searchsorted(ta, t[k])
        if interpolation is None and k > 0 and 0 < j < len(ta) and ta[j] != t[k]:
            interpolation = {'index': k, 'query_time_s': t[k], 'acc_bracket_time_s': [ta[j-1], ta[j]],
                'acc_bracket_xyz': streams['acc1'].values[[j-1, j]].tolist(),
                'weight': (t[k]-ta[j-1])/(ta[j]-ta[j-1]), 'interpolated_acc_xyz': a[k].tolist()}
    with (OUT/'csv_to_angle_trace.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    params = {'purpose': 'This-recording parameter snapshot and explanation evidence; not universal or a learned model.',
        'source': source, 'raw_columns': list(original[0]), 'raw_rows': len(original),
        'bias_B_rad_s': bias.tolist(), 'bias_native_sample_count': int(((tg >= 0)&(tg < 1)).sum()),
        'axis_B': h.tolist(), 'opening_sign': -1, 'initial_angle_deg': initial,
        'initial_angle_source': meta['initialization_source'], 'gravity_reference': ref.tolist(),
        'gravity_reference_count_on_gyro_grid': int((t < 1).sum()), 'gref': gref,
        'e1': e1.tolist(), 'e2': e2.tolist(), 'tau_s': 1., 'clock_this_recording_only': clock,
        'state_after_index_1': {key: rows[1][key] for key in ('time_s', 'axial_rate_rad_s', 'final_fused_deg', 'pure_gyro_deg', 'acc_x_m_s2', 'acc_y_m_s2', 'acc_z_m_s2')},
        'first_three_updates': rows[:3], 'first_acc_interpolation': interpolation,
        'verification': {'traced_rows': len(rows), 'max_abs_fused_difference_deg': max(abs(r['final_fused_deg']-r['saved_fused_deg']) for r in rows),
            'all_gyro_acc_fused_values_and_update_flags_checked': True}}
    (OUT/'csv_to_angle_parameters.json').write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(params, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
