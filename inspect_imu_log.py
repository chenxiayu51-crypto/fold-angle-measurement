"""Read-only sensor-log audit; write diagnostics, preserving the source CSV."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def inspect(source, output):
    with Path(source).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row['sensor']].append((index, int(row['timestamp_ns']),
                                     tuple(float(row[k]) for k in ('x', 'y', 'z'))))
    result = {'source': str(Path(source).resolve()), 'rows': len(rows), 'streams': {}}
    for name, records in groups.items():
        by_time = defaultdict(list)
        for index, timestamp, value in records:
            by_time[timestamp].append((index, value))
        times = np.array(sorted(by_time), dtype=np.int64)
        values = np.array([by_time[int(t)][-1][1] for t in times])
        t = (times - times[0]) / 1e9
        dt = np.diff(t)
        integral = np.sum((values[1:] + values[:-1]) * dt[:, None] / 2, axis=0)
        result['streams'][name] = {
            'records': len(records), 'unique_timestamps': len(times),
            'duplicate_timestamp_records': len(records) - len(times),
            'conflicting_timestamps': sum(len(set(v for _, v in vv)) > 1 for vv in by_time.values()),
            'timestamp_backsteps': sum(records[i][1] < records[i-1][1] for i in range(1,len(records))),
            'first_timestamp_ns': int(times[0]), 'last_timestamp_ns': int(times[-1]),
            'duration_s': float(t[-1]),
            'dt_percentiles_ms': np.percentile(dt * 1000, [0, 5, 50, 95, 100]).tolist(),
            'initial_5s_median_xyz': np.median(values[t < 5], axis=0).tolist(),
            'initial_5s_std_xyz': np.std(values[t < 5], axis=0).tolist(),
            'min_xyz': values.min(axis=0).tolist(), 'max_xyz': values.max(axis=0).tolist(),
            'trapezoid_integral_xyz_raw_units_s': integral.tolist(),
            'one_second_bins': [
                {'start_s': second, 'count': int(np.sum((t>=second)&(t<second+1))),
                 'median_xyz': np.median(values[(t>=second)&(t<second+1)],axis=0).tolist(),
                 'max_norm': float(np.max(np.linalg.norm(values[(t>=second)&(t<second+1)],axis=1)))}
                for second in range(int(t[-1])+1) if np.any((t>=second)&(t<second+1))]
        }
    out = Path(output)
    out.mkdir(parents=True,exist_ok=True)
    (out/'data_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for name, stream in result['streams'].items():
        print(json.dumps({'sensor':name,**{k:v for k,v in stream.items() if k!='one_second_bins'}},ensure_ascii=True))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    inspect(args.input,args.output)
