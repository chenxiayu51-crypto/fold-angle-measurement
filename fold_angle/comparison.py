"""Physical baselines with explicit gyro masking and common-support metrics."""
import numpy as np
from imu_angle import estimate_angle, _gravity_phase


def run_baseline(data, *, keep_gyro, initial_angle_deg, axis_b, axis_a,
                 bias_b, bias_a, reference_window_s=(0.,1.)):
    t=np.asarray(data['t']); acc_b=np.asarray(data['acc_b']);acc_a=np.asarray(data['acc_a'])
    lo,hi=reference_window_s; initial=(t>=lo)&(t<hi)
    if initial.sum()<3:raise ValueError('Insufficient initial reference')
    if lo>t[0]+.001:raise ValueError('Current baseline requires a reference at initialization')
    if keep_gyro not in ('AB','A','B','B_fixed'):raise ValueError('Unknown gyro mask')
    if keep_gyro=='A':
        # No missing-B gyro is substituted with zero: absolute phase is used
        # only during low dynamics inferred from acceleration itself.
        pb,ob,gb=_gravity_phase(acc_b,np.asarray(axis_b),initial)
        pa,oa,ga=_gravity_phase(acc_a,np.asarray(axis_a),initial)
        phase=np.arctan2(np.sin(pb-pa),np.cos(pb-pa))
        angle=initial_angle_deg-np.rad2deg(phase)
        na=np.linalg.norm(acc_a,axis=1);nb=np.linalg.norm(acc_b,axis=1)
        jerk_b=np.r_[0.,np.linalg.norm(np.diff(acc_b,axis=0),axis=1)/np.diff(t)]
        jerk_a=np.r_[0.,np.linalg.norm(np.diff(acc_a,axis=0),axis=1)/np.diff(t)]
        phase_rate=np.r_[0.,np.abs(np.diff(np.unwrap(phase)))/np.diff(t)]
        good=(np.abs(na-ga)<.6)&(np.abs(nb-gb)<.6)&(oa>.2)&(ob>.2)&(jerk_a<30)&(jerk_b<30)&(phase_rate<.25)
        base_g=np.asarray(data['gyro_a'])-np.asarray(bias_a)
        good &= np.linalg.norm(base_g,axis=1)<.25
        return {'angle_deg':np.where(good,angle,np.nan),'gyro_angle_deg':np.full(len(t),np.nan),
                'acc_angle_deg':angle,'acc_quality':good,'acc_used':good,'valid':good,
                'low_motion':good,'rate_deg_s':np.full(len(t),np.nan),
                'note':'Acc phase baseline with only A gyro; no high-dynamic tracking claimed; invalid samples are retained.'}
    gyro_b=np.asarray(data['gyro_b'])-np.asarray(bias_b)
    kwargs={}
    if keep_gyro in ('AB','B'):kwargs['base_acc']=acc_a
    if keep_gyro=='AB':kwargs['base_gyro']=np.asarray(data['gyro_a'])-np.asarray(bias_a)
    result=estimate_angle(t,acc_b,gyro_b,axis=tuple(axis_b),base_axis=tuple(axis_a),
                          calibration_s=hi-t[0],initial_angle_deg=initial_angle_deg,
                          gyro_bias_xyz=[0,0,0],base_gyro_bias_xyz=[0,0,0],**kwargs)
    result['note']='Existing one-DOF estimate_angle; offline gap-safe regridding, actual gyro times.'
    return result


def residual_metrics(reference, estimate, mask=None):
    ref=np.asarray(reference);est=np.asarray(estimate)
    m=np.isfinite(ref)&np.isfinite(est)
    if mask is not None:m &= mask
    if not m.any():return {'samples':0,'mae_deg':None,'rmse_deg':None,'p95_abs_deg':None,'max_abs_deg':None,'mean_bias_deg':None}
    e=est[m]-ref[m]
    return {'samples':int(m.sum()),'mae_deg':float(np.abs(e).mean()),'rmse_deg':float(np.sqrt(np.mean(e*e))),
            'p95_abs_deg':float(np.percentile(np.abs(e),95)),'max_abs_deg':float(np.max(np.abs(e))),
            'mean_bias_deg':float(e.mean()),'median_bias_deg':float(np.median(e))}
