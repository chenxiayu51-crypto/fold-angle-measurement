"""One-DOF IMU hinge estimator. No endpoint clipping or endpoint scale fitting."""
from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np


@dataclass
class Stream:
    timestamps_ns: np.ndarray
    values: np.ndarray
    input_records: int
    duplicate_records: int
    conflicting_timestamps: int


def read_log(path):
    """Keep the last logged value per stream/timestamp; never float-cast ns."""
    grouped = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            name = row['sensor']
            timestamp = int(row['timestamp_ns'])
            value = tuple(float(row[key]) for key in ('x', 'y', 'z'))
            records = grouped.setdefault(name, [])
            if records and timestamp < records[-1][0]:
                raise ValueError(f'Timestamp backstep in {name!r}; segment the recording first')
            records.append((timestamp, value))
    result = {}
    for name, records in grouped.items():
        unique, seen, conflicts = {}, {}, set()
        for timestamp, value in records:
            if timestamp in seen and value != seen[timestamp]:
                conflicts.add(timestamp)
            seen[timestamp] = value
            unique[timestamp] = value
        times = np.array(sorted(unique), dtype=np.int64)
        values = np.array([unique[int(t)] for t in times], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite sample in {name!r}')
        result[name] = Stream(times, values, len(records), len(records)-len(times), len(conflicts))
    return result


def _gravity_phase(acc, axis, initial):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    reference = np.median(acc[initial], axis=0)
    g_ref = float(np.linalg.norm(reference))
    in_plane = reference - np.dot(reference, axis)*axis
    if np.linalg.norm(in_plane) < .05*g_ref:
        return np.full(len(acc), np.nan), np.zeros(len(acc)), g_ref
    e1 = in_plane/np.linalg.norm(in_plane)
    e2 = -np.cross(axis, e1)
    parallel_removed = acc-(acc@axis)[:, None]*axis
    observability = np.linalg.norm(parallel_removed,axis=1)/g_ref
    phase = np.arctan2(acc@e2, acc@e1)
    return phase, observability, g_ref


def estimate_angle(t, acc, gyro, *, base_acc=None, base_gyro=None,
                   axis=(0.,1.,0.), base_axis=(0.,1.,0.), calibration_s=5.,
                   initial_angle_deg=180., correction_tau_s=1., max_gap_s=.1,
                   gyro_bias_xyz=None, base_gyro_bias_xyz=None):
    """Input samples must already share timestamps, SI units and directed axes.

    The model is 1-DOF. Positive axial rate closes the device in this
    recording convention. References are initialized from the first static
    window at known opening angle. Updates are causal after initialization;
    preprocessing/regridding, when used by callers, may be offline.
    """
    t=np.asarray(t,dtype=float);acc=np.asarray(acc,dtype=float);gyro=np.asarray(gyro,dtype=float)
    if len(t)<2 or np.any(np.diff(t)<=0):
        raise ValueError('Need strictly increasing sample times')
    if acc.shape!=(len(t),3) or gyro.shape!=acc.shape:
        raise ValueError('Expected Nx3 acceleration and gyro arrays')
    initial=t<t[0]+calibration_s
    if np.sum(initial)<2:
        raise ValueError('Insufficient calibration samples')
    axis=np.asarray(axis,dtype=float);axis/=np.linalg.norm(axis)
    ba=np.asarray(base_axis,dtype=float);ba/=np.linalg.norm(ba)
    bias=(np.median(gyro[initial],axis=0) if gyro_bias_xyz is None
          else np.asarray(gyro_bias_xyz,dtype=float))
    corrected=gyro-bias
    rate=corrected@axis
    phase,obs,g_ref=_gravity_phase(acc,axis,initial)
    quality=(np.abs(np.linalg.norm(acc,axis=1)-g_ref)<.6)&(obs>.2)
    low_motion=np.linalg.norm(corrected,axis=1)<.25
    jerk=np.r_[0.,np.linalg.norm(np.diff(acc,axis=0),axis=1)/np.diff(t)]
    quality &= (jerk<30.)
    base_bias=np.zeros(3)
    if base_acc is not None:
        base_acc=np.asarray(base_acc,dtype=float)
        bp,bo,bg=_gravity_phase(base_acc,ba,initial)
        phase=np.arctan2(np.sin(phase-bp),np.cos(phase-bp))
        quality &= (np.abs(np.linalg.norm(base_acc,axis=1)-bg)<.6)&(bo>.2)
        base_jerk=np.r_[0.,np.linalg.norm(np.diff(base_acc,axis=0),axis=1)/np.diff(t)]
        quality &= base_jerk<30.
    if base_gyro is not None:
        base_gyro=np.asarray(base_gyro,dtype=float)
        base_bias=(np.median(base_gyro[initial],axis=0) if base_gyro_bias_xyz is None
                   else np.asarray(base_gyro_bias_xyz,dtype=float))
        rate-=(base_gyro-base_bias)@ba
        low_motion &= np.linalg.norm(base_gyro-base_bias,axis=1)<.25
    quality &= np.isfinite(phase)
    n=len(t);start=np.deg2rad(initial_angle_deg)
    fused=np.full(n,np.nan);integral=np.full(n,np.nan);acc_angle=np.full(n,np.nan)
    used=np.zeros(n,dtype=bool);valid=np.zeros(n,dtype=bool)
    fused[0]=integral[0]=start;valid[0]=True
    previous=start
    for k in range(n):
        if k and t[k]-t[k-1]>max_gap_s:
            previous=np.nan
            continue
        if k:
            dt=t[k]-t[k-1]
            delta=-.5*(rate[k]+rate[k-1])*dt
            if np.isfinite(integral[k-1]): integral[k]=integral[k-1]+delta
            predicted=previous+delta
        else:
            dt=0.;predicted=start
        if np.isfinite(phase[k]):
            raw=start-phase[k]
            anchor=predicted if np.isfinite(predicted) else start
            observation=raw+2*np.pi*np.round((anchor-raw)/(2*np.pi))
            acc_angle[k]=observation
        else:
            observation=np.nan
        if not np.isfinite(predicted):
            if quality[k] and low_motion[k]:
                predicted=observation;used[k]=True
            else:
                previous=np.nan
                continue
        innovation=observation-predicted
        if quality[k] and low_motion[k] and abs(innovation)<np.deg2rad(20):
            gain=1-np.exp(-dt/correction_tau_s)
            predicted+=gain*innovation;used[k]=True
        fused[k]=predicted;previous=predicted;valid[k]=True
    return {'angle_deg':np.rad2deg(fused),'gyro_angle_deg':np.rad2deg(integral),
            'acc_angle_deg':np.rad2deg(acc_angle),'rate_deg_s':-np.rad2deg(rate),
            'acc_used':used,'acc_quality':quality,'low_motion':low_motion,'valid':valid,
            'gyro_bias_rad_s':float(bias@axis-base_bias@ba),
            'gyro_bias_xyz_rad_s':bias.tolist(),'base_gyro_bias_xyz_rad_s':base_bias.tolist(),
            'initial_gravity_norm_m_s2':g_ref}
