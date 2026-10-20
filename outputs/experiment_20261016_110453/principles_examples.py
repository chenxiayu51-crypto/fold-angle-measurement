"""Recompute worked examples for the principles note, without rerunning experiments.

Run from the project root with the project's Python environment.
Only principles_examples.json is written; input and experiment CSVs stay intact.
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from fold_angle.synchronization import gap_interp
from fold_angle.filtering import AdaptiveAngleFilter
from imu_angle import read_log, _gravity_phase


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    out = Path(__file__).resolve().parent
    summary = json.loads((out/'imu/summary.json').read_text(encoding='utf-8'))
    raw = read_csv(Path(summary['source']))
    streams = read_log(summary['source'])
    grouped = {}
    for row in raw:
        grouped.setdefault((row['sensor'], row['timestamp_ns']), []).append(row)
    conflict_key, conflicts = next((key, rows) for key, rows in grouped.items()
                                  if key[0] == 'gyro1' and len({(r['x'], r['y'], r['z']) for r in rows}) > 1)
    conflict_time = int(conflict_key[1])
    idx = np.flatnonzero(streams['gyro1'].timestamps_ns == conflict_time)[0]
    evidence = {'duplicate_conflict': {'sensor': conflict_key[0], 'timestamp_ns': conflict_time,
                'raw_xyz_in_file_order': [[float(r[k]) for k in ('x', 'y', 'z')] for r in conflicts],
                'kept_xyz': streams['gyro1'].values[idx].tolist()}}

    imu = read_csv(out/'imu/angles.csv')
    t = np.array([float(r['time_s']) for r in imu])
    ns = np.array([int(r['timestamp_ns']) for r in imu], dtype=np.int64)
    gstream = streams['gyro1']
    g = gstream.values[np.searchsorted(gstream.timestamps_ns, ns)]
    bias = np.array(summary['bias_B_xyz_rad_s'])
    axis = np.array(summary['axis_b'])
    rates = (g-bias)@axis
    acc_t = (streams['acc1'].timestamps_ns-summary['clock_map']['host_base_ns'])/1e9
    acc = gap_interp(acc_t, streams['acc1'].values, t, .1)
    initial = t < 1.
    phase, obs, gref = _gravity_phase(acc, axis, initial)
    jerk = np.r_[0., np.linalg.norm(np.diff(acc, axis=0), axis=1)/np.diff(t)]

    def step(k):
        prev = float(imu[k-1]['fixed_base_imu_angle_deg'])
        dt = t[k]-t[k-1]
        increment = -.5*(rates[k-1]+rates[k])*dt*180/np.pi
        predicted = prev+increment
        observation = float(imu[k]['acc_angle_deg'])
        gain = 1-np.exp(-dt)
        used = bool(int(imu[k]['acc_update_used']))
        calculated = predicted+gain*(observation-predicted) if used else predicted
        saved = float(imu[k]['fixed_base_imu_angle_deg'])
        assert abs(calculated-saved) < 1e-9
        return {'index': int(k), 'times_s': [t[k-1], t[k]], 'dt_s': dt,
                'raw_gyro_xyz_rad_s': [g[k-1].tolist(), g[k].tolist()],
                'corrected_axial_rate_rad_s': [rates[k-1], rates[k]],
                'previous_fused_deg': prev, 'increment_deg': increment,
                'prediction_deg': predicted, 'interpolated_acc_xyz_m_s2': acc[k].tolist(),
                'gravity_reference_xyz': np.median(acc[initial], axis=0).tolist(),
                'gravity_reference_norm': gref, 'acc_norm': float(np.linalg.norm(acc[k])),
                'projected_gravity_fraction': obs[k], 'jerk_m_s3': jerk[k],
                'gyro_norm_rad_s': float(np.linalg.norm(g[k]-bias)),
                'gravity_phase_rad': phase[k], 'acc_observation_deg': observation,
                'innovation_deg': observation-predicted, 'gain': gain,
                'acc_used': used, 'calculated_deg': calculated, 'saved_deg': saved}

    evidence['accepted_step'] = step(1)
    rejected = next(k for k in range(1, len(imu)) if float(imu[k]['time_s']) > 2
                    and imu[k]['acc_update_used'] == '0' and imu[k]['low_motion'] == '0')
    evidence['rejected_step'] = step(rejected)
    vr = read_csv(out/'video/angles.csv')
    filt = AdaptiveAngleFilter()
    for row in vr[:2]:
        calculated = filt.update(float(row['projected_angle_deg']), float(row['timestamp_s']))
        assert abs(calculated-float(row['smoothed_projected_angle_deg'])) < 2e-5
    row = vr[0]
    lower = np.array([float(row['lower_free_'+c])-float(row['lower_hinge_'+c]) for c in ('x', 'y')])
    upper = np.array([float(row['observed_upper_free_'+c])-float(row['observed_upper_hinge_'+c]) for c in ('x', 'y')])
    angle = np.rad2deg(np.arctan2(abs(lower[0]*upper[1]-lower[1]*upper[0]), lower@upper))
    assert abs(angle-float(row['selected_edge_projected_angle_deg'])) < .002
    evidence['visual_example'] = {'frame': 0, 'lower_vector': lower.tolist(), 'observed_upper_vector': upper.tolist(),
        'angle_from_rounded_coordinates_deg': angle, 'selected_edge_angle_deg': float(row['selected_edge_projected_angle_deg']),
        'fused_raw_angle_deg': float(row['projected_angle_deg']), 'fusion_count': int(row['fusion_count']),
        'next_raw_angle_deg': float(vr[1]['projected_angle_deg']), 'next_smoothed_deg': calculated}

    aligned = read_csv(out/'comparison/aligned_angles.csv')
    example = min((r for r in aligned if r['core_comparison_valid'] == '1'),
                  key=lambda r: abs(float(r['video_time_s'])-10))
    query = float(example['imu_time_s'])
    right = int(np.searchsorted(t, query))
    weight = (query-t[right-1])/(t[right]-t[right-1])
    left_angle = float(imu[right-1]['fixed_base_imu_angle_deg'])
    right_angle = float(imu[right]['fixed_base_imu_angle_deg'])
    interpolated = left_angle+weight*(right_angle-left_angle)
    assert abs(interpolated-float(example['Fixed complementary'])) < 1e-9
    evidence['aligned_example'] = {'row': example, 'imu_bracket_s': [t[right-1], t[right]],
        'imu_bracket_fixed_angles_deg': [left_angle, right_angle], 'interpolation_weight': weight,
        'recomputed_estimate_deg': interpolated,
        'residual_deg': interpolated-float(example['visual_raw_projected_deg'])}
    common = [r for r in aligned if r['core_comparison_valid'] == '1']
    residual = np.array([float(r['Fixed complementary'])-float(r['visual_raw_projected_deg']) for r in common])
    metrics = json.loads((out/'comparison/metrics.json').read_text(encoding='utf-8'))['core_common_metrics']['Fixed complementary']
    recomputed = {'samples': len(residual), 'mae_deg': float(np.mean(abs(residual))),
                  'rmse_deg': float(np.sqrt(np.mean(residual**2))),
                  'p95_abs_deg': float(np.percentile(abs(residual), 95)),
                  'max_abs_deg': float(max(abs(residual))), 'mean_bias_deg': float(np.mean(residual))}
    assert all(abs(v-metrics[k]) < 1e-10 for k, v in recomputed.items())
    evidence['fixed_metrics_recomputed'] = recomputed
    evidence['checks'] = {'duplicate_keep_last_matches': bool(np.array_equal(
        streams['gyro1'].values[idx], np.array([float(conflicts[-1][c]) for c in ('x', 'y', 'z')]))),
        'accepted_and_rejected_updates_reproduced': True, 'one_euro_first_two_rows_reproduced': True,
        'observed_edge_angle_reproduced_with_csv_rounding_tolerance': True,
        'aligned_interpolation_and_metrics_reproduced': True}
    assert all(evidence['checks'].values())
    (out/'principles_examples.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
