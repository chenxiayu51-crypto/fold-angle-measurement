"""Measure projected fold angles in a manually initialized fixed-view video."""
import argparse
from collections import Counter, deque
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from fold_angle.detector import SideEdgeDetector


def _json_write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def _point_fields(prefix, edge):
    names = [f'{prefix}_{end}_{axis}' for end in ('hinge', 'free') for axis in ('x', 'y')]
    values = [''] * 4 if edge is None else [round(float(value), 3) for value in edge.flat]
    return dict(zip(names, values))


def _overlay(frame, result, index, fps):
    if result.fixed_geometry and result.registration_offset is not None:
        dx, dy = result.registration_offset
        transform = np.array([[1., 0., -dx], [0., 1., -dy]])
        frame = cv2.warpAffine(frame, transform, frame.shape[1::-1],
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(22, 24, 28))
    view = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    if result.fixed_geometry and result.observed_upper is not None:
        observed = result.observed_upper - result.registration_offset
        a, b = np.rint(observed * 2).astype(int)
        cv2.line(view, tuple(a), tuple(b), (190, 190, 190), 1, cv2.LINE_AA)
    for edge, color in [(result.lower, (70, 235, 60)), (result.upper, (60, 70, 255))]:
        if edge is None:
            continue
        a, b = np.rint(edge * 2).astype(int)
        cv2.arrowedLine(view, tuple(a), tuple(b), color, 3, cv2.LINE_AA, tipLength=.08)
        cv2.circle(view, tuple(b), 5, color, -1, cv2.LINE_AA)
    if result.hinge is not None:
        cv2.circle(view, tuple(np.rint(result.hinge * 2).astype(int)), 5, (30, 230, 255), -1)
    canvas = cv2.copyMakeBorder(view, 64, 40, 0, 0, cv2.BORDER_CONSTANT, value=(22, 24, 28))
    shown_angle = result.filtered_angle_deg if result.fixed_geometry else result.angle_deg
    label = f'{shown_angle:6.1f} deg' if result.valid else 'INVALID'
    color = (240, 245, 240) if result.valid else (80, 175, 255)
    cv2.putText(canvas, f'Fold angle: {label} (projected)', (18, 29), cv2.FONT_HERSHEY_SIMPLEX, .73, color, 2, cv2.LINE_AA)
    detail = f't = {index/fps:.3f} s | frame {index} | support = {result.score:.2f}'
    if result.valid and result.fixed_geometry:
        detail += f' | raw = {result.angle_deg:.1f}'
    if not result.valid:
        detail = f't = {index/fps:.3f} s | {result.reason}'
    cv2.putText(canvas, detail, (18, 53), cv2.FONT_HERSHEY_SIMPLEX, .49, (205, 205, 205), 1, cv2.LINE_AA)
    legend = ('FIXED pivot + GREEN reference | RED filtered upper | GRAY observed' if result.fixed_geometry
              else 'GREEN: lower side   RED: upper side   0=closed / 180=flat')
    cv2.putText(canvas, legend,
                (18, canvas.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, .53, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def _plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times = np.array([row['timestamp_s'] for row in rows])
    raw = np.array([row['projected_angle_deg'] if row['valid'] else np.nan for row in rows])
    filtered = np.array([row['smoothed_projected_angle_deg'] if row['valid'] else np.nan for row in rows])
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=160)
    ax.plot(times, raw, color='#939eab', linewidth=.8, label='Detected angle')
    fixed = rows[0].get('geometry_coordinate_system') == 'stabilized_fixed_rays'
    ax.plot(times, filtered, color='#176ac4', linewidth=1.7,
            label='Adaptive One Euro filter' if fixed else 'Causal median (5 valid frames)')
    missing = ~np.isfinite(raw)
    if missing.any():
        ax.plot(times[missing], np.full(missing.sum(), -5), '|', color='#df573d', label='Invalid frame')
    ax.set(xlabel='Time (s)', ylabel='Projected opening angle (degrees)', ylim=(-8, 188),
           title=('Fixed lower direction + common pivot | 0 = closed, 180 = flat' if fixed
                  else 'Two-side-edge fold angle | 0 = closed, 180 = flat'))
    ax.set_yticks(np.arange(0, 181, 30))
    ax.grid(alpha=.2)
    ax.legend(loc='best', fontsize=9)
    fig.text(.12, .015, 'Uncalibrated image angle. Missing frames are not interpolated.', fontsize=9, color='#555555')
    fig.tight_layout(rect=[0, .035, 1, 1])
    fig.savefig(path)
    plt.close(fig)


def run_video(video_path, config_path, output_dir):
    source, config_path, output_dir = Path(video_path), Path(config_path), Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    if source.resolve() == (output_dir / 'annotated.mp4').resolve():
        raise ValueError('Output video would overwrite input video')
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f'Cannot decode video: {source}')
    writer = None
    started = False
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Video has no usable nominal FPS')
        declared_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, first = capture.read()
        if not ok:
            raise ValueError('Cannot decode first frame')
        detector = SideEdgeDetector(dict(config, fps=fps), first)
        output_dir.mkdir(parents=True, exist_ok=True)
        _json_write(output_dir / 'summary.json', {'status': 'running', 'source': str(source.resolve())})
        started = True
        results, rows = [], []
        median_window = deque(maxlen=5)
        frame, index = first, 0
        while True:
            result = detector.detect(frame)
            results.append(result)
            row = dict(frame_index=index, timestamp_s=round(index / fps, 6),
                       projected_angle_deg=round(result.angle_deg, 5) if result.valid else '',
                       smoothed_projected_angle_deg='', valid=int(result.valid),
                       support_score=round(result.score, 5), base_match_score=round(result.base_score, 5),
                       candidate_count=result.candidate_count, failure_reason=result.reason,
                       interpolated=0, stabilized_projected_angle_deg='',
                       selected_edge_projected_angle_deg='', fusion_count=result.fusion_count,
                       geometry_coordinate_system=('stabilized_fixed_rays' if result.fixed_geometry
                                                   else 'source_observed_segments'))
            if result.valid:
                if result.fixed_geometry:
                    smoothed = result.filtered_angle_deg
                else:
                    median_window.append(result.angle_deg)
                    smoothed = float(np.median(median_window))
                row['smoothed_projected_angle_deg'] = round(smoothed, 5)
                row['stabilized_projected_angle_deg'] = round(smoothed, 5)
                row['selected_edge_projected_angle_deg'] = round(
                    result.selected_edge_angle_deg if result.fixed_geometry else result.angle_deg, 5)
            else:
                median_window.clear()
            row.update(_point_fields('lower', result.lower))
            row.update(_point_fields('upper', result.upper))
            row.update(_point_fields('observed_lower', result.observed_lower if result.fixed_geometry else result.lower))
            row.update(_point_fields('observed_upper', result.observed_upper if result.fixed_geometry else result.upper))
            delta = result.registration_offset if result.registration_offset is not None else np.zeros(2)
            row['registration_dx'], row['registration_dy'] = map(float, delta)
            row['hinge_anchor_x'] = round(float(result.hinge[0]), 3) if result.hinge is not None else ''
            row['hinge_anchor_y'] = round(float(result.hinge[1]), 3) if result.hinge is not None else ''
            rows.append(row)
            index += 1
            ok, frame = capture.read()
            if not ok:
                break
        if declared_count > 0 and len(rows) != declared_count:
            raise RuntimeError(f'Decoded {len(rows)} frames, metadata declares {declared_count}; output is incomplete')
        with (output_dir / 'angles.csv').open('w', newline='', encoding='utf-8-sig') as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            csv_writer.writeheader()
            csv_writer.writerows(rows)
        _plot(rows, output_dir / 'angle_curve.png')
        # Decode again instead of keeping the whole video in RAM.
        capture.release()
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError('Cannot reopen source for annotation')
        picks = np.unique(np.linspace(0, len(rows) - 1, min(24, len(rows))).astype(int))
        picked = []
        for index, result in enumerate(results):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'Cannot decode annotation frame {index}')
            view = _overlay(frame, result, index, fps)
            if writer is None:
                writer = cv2.VideoWriter(str(output_dir / 'annotated.mp4'),
                                         cv2.VideoWriter_fourcc(*'mp4v'), fps, view.shape[1::-1])
                if not writer.isOpened():
                    raise RuntimeError('Cannot create MP4 writer')
            writer.write(view)
            if index in picks:
                picked.append(cv2.resize(view, (480, 308)))
        writer.release()
        writer = None
        columns = min(4, len(picked))
        sheet = np.full((((len(picked) + columns - 1) // columns) * 308, columns * 480, 3), 245, np.uint8)
        for k, view in enumerate(picked):
            y, x = k // columns * 308, k % columns * 480
            sheet[y:y + 308, x:x + 480] = view
        if not cv2.imwrite(str(output_dir / 'diagnostics.jpg'), sheet):
            raise RuntimeError('Cannot save diagnostics image')
        valid_values = [r.angle_deg for r in results if r.valid]
        failure_counts = Counter(r.reason for r in results if not r.valid)
        summary = dict(status='complete', source=str(source.resolve()),
                       method=('fixed lower direction/common pivot + compatible LSD angle fusion + One Euro filter'
                               if detector.fixed_geometry else 'manually initialized template tracking + constrained LSD side-edge detection'),
                       assumptions='fixed camera and fixed lower body; local translation compensates small accidental movement',
                       config=config, fps=fps, processed_frames=len(rows), declared_frames=declared_count,
                       duration_s=len(rows) / fps, last_frame_timestamp_s=rows[-1]['timestamp_s'],
                       timestamp_method='frame_index / nominal FPS; assumes constant frame rate',
                       valid_frames=len(valid_values), invalid_frames=len(rows) - len(valid_values),
                       valid_fraction=len(valid_values) / len(rows), invalid_reasons=dict(failure_counts),
                       min_projected_angle_deg=min(valid_values) if valid_values else None,
                       max_projected_angle_deg=max(valid_values) if valid_values else None,
                       angle_type='uncalibrated 2D projection; no mechanical-angle ground truth',
                       smoothing=('One Euro adaptive cutoff; angle filter resets after every invalid frame'
                                  if detector.fixed_geometry else 'causal median of up to 5 consecutive valid frames; resets after every invalid frame'),
                       support_score_type='heuristic image support; not probability or validated accuracy',
                       interpolation='none', audio='not retained in annotated video')
        if detector.fixed_geometry:
            summary.update(fixed_pivot=detector.hinge0.tolist(), fixed_lower_ray=detector.fixed_lower_ray.tolist(),
                           geometry_coordinate_system='stabilized_fixed_rays',
                           observation_coordinate_system='original video pixels',
                           raw_angle_definition='weighted fusion of compatible observed upper directions against one fixed lower direction',
                           display_angle_definition='One Euro filtered raw angle; two modeled rays share the fixed pivot',
                           frame_stabilization='translation of original image by negative base template offset; uncovered borders dark',
                           physical_rail_note='finite-thickness observed rails need not intersect; modeled rays represent directions at a common virtual pivot')
        _json_write(output_dir / 'summary.json', summary)
        return summary
    except Exception as exc:
        if started:
            _json_write(output_dir / 'summary.json', {'status': 'failed', 'error': str(exc),
                        'note': 'Output files may be partial or from an earlier run; do not treat them as a complete result.'})
        raise
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--config', default=str(Path(__file__).parent / 'configs/sample_video.json'))
    parser.add_argument('--output', default=str(Path(__file__).parent / 'outputs/fold_angle_fixed'))
    args = parser.parse_args()
    summary = run_video(args.video, args.config, args.output)
    print(json.dumps({key: summary[key] for key in ['status', 'processed_frames', 'valid_frames', 'invalid_frames', 'valid_fraction', 'invalid_reasons']}, indent=2))


if __name__ == '__main__':
    main()
