"""Check real experiment artifacts independently, then run all project tests."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import cv2
import numpy as np


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def number(row,key):return float(row[key]) if row[key] else np.nan


def verify(out):
    out=Path(out).resolve();project=Path(__file__).resolve().parent
    required=['video/annotated.mp4','video/angles.csv','video/diagnostics.png','video/summary.json',
              'imu/cleaned_imu.csv','imu/angles.csv','imu/data_audit.json','imu/summary.json',
              'synchronization/time_alignment.json','synchronization/angular_velocity_overlay.png','synchronization/synchronization_diagnostics.png',
              'comparison/aligned_angles.csv','comparison/angle_overlay.png','comparison/angle_residuals.png','comparison/error_by_angle.png','comparison/error_by_speed.png','comparison/metrics.json',
              'ablation/baseline_comparison.csv','ablation/baseline_comparison.png','ablation/metrics.json','report.md','verification.json']
    assert all((out/p).is_file() and (out/p).stat().st_size>0 for p in required)
    for p in out.rglob('*.png'):
        if '_probes' in p.parts:continue
        assert p.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n',p
        assert cv2.imread(str(p)) is not None,p
    manifest=json.loads((out/'experiment_manifest.json').read_text(encoding='utf-8'))
    for p,digest in manifest['inputs_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest,p
    for p,digest in manifest['new_code_sha256'].items():
        assert hashlib.sha256((project/p).read_bytes()).hexdigest()==digest,p
    # Check every clean value against the last raw record, without read_log.
    source={}
    for r in read_rows(manifest['imu']):source[(r['sensor'],int(r['timestamp_ns']))]=tuple(float(r[k]) for k in ('x','y','z'))
    cleaned=read_rows(out/'imu/cleaned_imu.csv')
    assert len(cleaned)==len(source)
    for r in cleaned:
        key=(r['sensor'],int(r['original_timestamp_ns']))
        assert tuple(float(r[k]) for k in ('x','y','z'))==source[key]
    # Independent trapezoidal integration using native original timestamps.
    imu=read_rows(out/'imu/angles.csv');si=json.loads((out/'imu/summary.json').read_text(encoding='utf-8'))
    ns=np.array([int(r['timestamp_ns']) for r in imu],dtype=np.int64)
    assert (np.diff(ns)>0).all()
    gyro=np.array([source[('gyro1',int(t))] for t in ns])
    rate=(gyro-np.array(si['bias_B_xyz_rad_s']))@np.array(si['axis_b'])
    dt=np.diff(ns)/1e9
    integrated=si['initial_angle_deg']-np.r_[0.,np.cumsum(.5*(rate[:-1]+rate[1:])*dt)]*180/np.pi
    actual=np.array([float(r['gyro_integrated_angle_deg']) for r in imu])
    integration_error=float(np.max(np.abs(integrated-actual)))
    assert integration_error<1e-8,integration_error
    video=read_rows(out/'video/angles.csv');aligned=read_rows(out/'comparison/aligned_angles.csv')
    assert len(video)==len(aligned)
    metrics=json.loads((out/'comparison/metrics.json').read_text(encoding='utf-8'))
    mask=np.array([r['core_comparison_valid']=='1' for r in aligned]);ref=np.array([number(r,'visual_raw_projected_deg') for r in aligned])
    for name,expected in metrics['core_common_metrics'].items():
        vals=np.array([number(r,name) for r in aligned]);err=vals[mask]-ref[mask]
        assert len(err)==expected['samples']
        assert abs(np.mean(np.abs(err))-expected['mae_deg'])<1e-10
        assert abs(np.sqrt(np.mean(err**2))-expected['rmse_deg'])<1e-10
    geometry_error=[]
    for row in video:
        if row['valid']=='0':
            assert not row['projected_angle_deg'] and not row['smoothed_projected_angle_deg']
            continue
        u=np.array([float(row['lower_free_x'])-float(row['lower_hinge_x']),float(row['lower_free_y'])-float(row['lower_hinge_y'])])
        v=np.array([float(row['upper_free_x'])-float(row['upper_hinge_x']),float(row['upper_free_y'])-float(row['upper_hinge_y'])])
        ang=np.degrees(np.arctan2(abs(u[0]*v[1]-u[1]*v[0]),u@v))
        geometry_error.append(abs(ang-float(row['smoothed_projected_angle_deg'])))
    assert max(geometry_error)<.002
    sync=json.loads((out/'synchronization/time_alignment.json').read_text(encoding='utf-8'))
    train=sync['training_events'];holdout=sync['validation_events']
    assert not (set(e['event_id'] for e in train)&set(e['event_id'] for e in holdout))
    assert len(holdout)>0
    matched=sorted([e for e in train+holdout if e.get('peak_imu_s') is not None],key=lambda e:e['peak_video_s'])
    assert (np.diff([e['peak_imu_s'] for e in matched])>0).all()
    report_path=out/'report.md';report=report_path.read_text(encoding='utf-8')
    assert '\ufffd' not in report
    for image_path in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report):
        image=Path(image_path)
        if not image.is_absolute():image=report_path.parent/image
        assert image.is_file(),image
    run=subprocess.run([sys.executable,'-m','pytest','tests','-q'],cwd=project,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (out/'test_results.txt').write_text(run.stdout+'\n'+run.stderr,encoding='utf-8')
    verification=json.loads((out/'verification.json').read_text(encoding='utf-8'))
    verification.update(required_artifacts_checked=required,required_artifacts_exist=True,
                        native_cleaned_values_equal_last_raw_records=True,
                        independent_gyro_trapezoid_max_difference_deg=integration_error,
                        modeled_ray_vs_csv_angle_max_difference_deg=max(geometry_error),
                        independent_csv_metric_recomputation=True,all_png_files_decode=True,
                        matched_events_unique_and_ordered=True,training_and_validation_events_disjoint=True,
                        tests={'exit_code':run.returncode,'command':[sys.executable,'-m','pytest','tests','-q'],'output':run.stdout.strip(),'log':'test_results.txt'})
    (out/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    assert run.returncode==0,run.stdout+run.stderr
    marker='\n### 本次最终核验记录\n'
    if marker in report:report=report.split(marker)[0]
    report+=marker+'\n'+run.stdout.strip()+'\n\n已独立核对全部必需交付文件、PNG格式与解码、清理数据等于原始同时间戳最后记录、整数时间单调性、CSV残差指标、模型射线几何、原始文件哈希及事件匹配顺序。独立gyro再积分最大数值差为'+f'{integration_error:.3e}'+'°；该差值只验证数值实现一致，不是物理精度。\n'
    (out/'report.md').write_text(report,encoding='utf-8')
    print(json.dumps({'required_artifacts':len(required),'independent_integration_difference_deg':integration_error,'test_exit_code':run.returncode,'test_result':run.stdout.strip()},ensure_ascii=True,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    verify(parser.parse_args().output)
