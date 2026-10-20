import numpy as np
import pytest


def test_integer_timestamps_duplicates_and_conflicts():
    from imu_angle import read_log
    from tempfile import TemporaryDirectory
    from pathlib import Path
    with TemporaryDirectory() as folder:
        p=Path(folder)/'sensor.csv'
        p.write_text('timestamp_ns,delta_ms,sensor,x,y,z,w\n1792113199017999900,,acc1,1,2,3,0\n1792113199017999900,,acc1,1,2,3,0\n1792113199017999900,,acc1,3,2,3,0\n1792113199018999900,,acc1,4,5,6,0\n',encoding='utf-8')
        stream=read_log(p)['acc1']
        assert stream.timestamps_ns.dtype==np.int64
        assert stream.timestamps_ns[1]-stream.timestamps_ns[0]==1000000
        assert stream.duplicate_records==2 and stream.conflicting_timestamps==1
        assert np.array_equal(stream.values[0],[3,2,3])


def test_causal_filter_crosses_acc_branch_without_endpoint_snap():
    from imu_angle import estimate_angle
    t=np.arange(0,12,.01)
    phase=np.piecewise(t,[t<2,(t>=2)&(t<4),(t>=4)&(t<7),(t>=7)&(t<9),t>=9],
                       [0,lambda x:(x-2)*np.pi/2,np.pi,lambda x:np.pi-(x-7)*np.pi/2,0])
    gyro=np.zeros((len(t),3));gyro[:,1]=np.gradient(phase,t)+.003
    acc=np.column_stack((-9.81*np.sin(phase),np.zeros(len(t)),9.81*np.cos(phase)))
    # Dynamic acceleration anomaly should be rejected; no endpoint clipping.
    acc[(t>3.0)&(t<3.1)]*=2
    result=estimate_angle(t,acc,gyro,calibration_s=1,initial_angle_deg=180)
    truth=180-np.rad2deg(phase)
    assert np.max(np.abs(result['angle_deg']-truth))<1.0
    assert not result['acc_used'][(t>3.0)&(t<3.1)].any()
    assert result['gyro_bias_rad_s']==pytest.approx(.003,abs=1e-10)
    assert np.isfinite(result['angle_deg']).all()


def test_gravity_parallel_to_hinge_is_not_angle_measurement():
    from imu_angle import estimate_angle
    t=np.arange(0,5,.01)
    gyro=np.zeros((len(t),3));gyro[:,1]=np.where(t>=1,.1,0)
    acc=np.tile([0,9.81,0.],(len(t),1))
    result=estimate_angle(t,acc,gyro,calibration_s=.8,initial_angle_deg=100)
    assert not result['acc_used'].any()
    assert result['angle_deg'][-1]<80
    assert np.isnan(result['acc_angle_deg']).all()


def test_common_motion_cancels_with_explicit_same_hinge_axis():
    from imu_angle import estimate_angle
    t=np.arange(0,5,.01)
    p=np.where(t<1,0,(t-1)*.2)
    q=np.where(t<1,0,.1*np.sin(t-1))
    acc=np.column_stack((-9.81*np.sin(p+q),np.zeros(len(t)),9.81*np.cos(p+q)))
    base=np.column_stack((-9.81*np.sin(q),np.zeros(len(t)),9.81*np.cos(q)))
    gyro=np.zeros_like(acc);gyro[:,1]=np.gradient(p+q,t)
    bg=np.zeros_like(acc);bg[:,1]=np.gradient(q,t)
    result=estimate_angle(t,acc,gyro,base_acc=base,base_gyro=bg,calibration_s=.8)
    assert np.max(np.abs(result['angle_deg']-(180-np.rad2deg(p))))<.2


def test_long_gap_is_marked_unavailable_and_not_bridged():
    from imu_angle import estimate_angle
    t=np.r_[np.arange(0,2,.01),np.arange(3,5,.01)]
    acc=np.tile([0,0,9.81],(len(t),1));gyro=np.zeros_like(acc)
    result=estimate_angle(t,acc,gyro,calibration_s=.8)
    gap=np.flatnonzero(np.diff(t)>.1)+1
    assert np.isnan(result['angle_deg'][gap]).all()
    assert not result['valid'][gap].any()
    assert result['valid'][-1]
