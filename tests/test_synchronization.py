"""Known timing, gaps, direction and ablation leakage checks."""
import numpy as np
import pytest


def test_offset_recovery_with_missing_visual_data_and_separate_validation():
    from fold_angle.synchronization import align_motion
    tv=np.arange(0,35,1/30)
    def signal(t):
        return (90*np.exp(-((t-4)/.6)**2)-130*np.exp(-((t-9)/.35)**2)
                +65*np.exp(-((t-16)/.9)**2)-100*np.exp(-((t-27)/.5)**2))
    vi=signal(tv); vi[(tv>13)&(tv<14)]=np.nan
    ti=np.arange(0,36,.01); offset=-1.23
    imu=signal(ti-offset)/180*np.pi
    result=align_motion(tv,vi,ti,imu,offset_bounds=(-5,5))
    assert result['a']==1.0
    assert abs(result['b_s']-offset)<.04
    assert result['correlation_after']>.98
    assert len(result['validation_events'])>=1


def test_no_extrapolation_or_interpolation_across_large_gap():
    from fold_angle.synchronization import gap_interp
    t=np.array([0.,.1,.2,2.,2.1]); v=np.array([0.,1.,2.,20.,21.])
    q=np.array([-.1,.05,1.,2.05,2.2])
    out=gap_interp(t,v,q,max_gap_s=.3)
    assert np.isnan(out[[0,2,4]]).all()
    np.testing.assert_allclose(out[[1,3]],[.5,20.5])


def test_flat_signals_cannot_claim_synchronization():
    from fold_angle.synchronization import align_motion
    t=np.arange(0,10,.02)
    with pytest.raises(ValueError,match='motion'):
        align_motion(t,t*0,t,t*0)


def test_repeated_motion_is_reported_as_ambiguous():
    from fold_angle.synchronization import align_motion
    t=np.arange(0,40,.02);v=100*np.sin(2*np.pi*t/5)
    result=align_motion(t,v,t,np.deg2rad(v),offset_bounds=(-10,10))
    assert result['ambiguous_offset_candidates'] is True


def test_bidirectional_angles_preserve_complementary_configuration():
    from fold_angle.comparison import run_baseline
    t=np.arange(0,5,.01); theta=30+20*t
    a=np.column_stack([-9.81*np.sin(np.deg2rad(theta)),np.zeros(len(t)),-9.81*np.cos(np.deg2rad(theta))])
    g=np.tile([0.,-np.deg2rad(20),0.],(len(t),1))
    data={'t':t,'acc_b':a,'acc_a':np.tile([0,0,9.81],(len(t),1)),'gyro_b':g,'gyro_a':g*0}
    result=run_baseline(data,keep_gyro='B',initial_angle_deg=30,axis_b=[0,1,0],axis_a=[0,1,0],bias_b=[0,0,0],bias_a=[0,0,0],reference_window_s=(0,.15))
    assert abs(result['gyro_angle_deg'][-1]-theta[-1])<1e-8
    altered=dict(data,gyro_a=np.full_like(g,999.))
    again=run_baseline(altered,keep_gyro='B',initial_angle_deg=30,axis_b=[0,1,0],axis_a=[0,1,0],bias_b=[0,0,0],bias_a=[0,0,0],reference_window_s=(0,.15))
    np.testing.assert_allclose(result['angle_deg'],again['angle_deg'],equal_nan=True)


def test_keep_a_never_depends_on_gyro_b():
    from fold_angle.comparison import run_baseline
    t=np.arange(0,5,.01); a=np.tile([0.,0.,9.81],(len(t),1)); z=a*0
    data={'t':t,'acc_b':a,'acc_a':a,'gyro_b':z,'gyro_a':z}
    params=dict(keep_gyro='A',initial_angle_deg=20,axis_b=[0,1,0],axis_a=[0,1,0],bias_b=[0,0,0],bias_a=[0,0,0],reference_window_s=(0,1))
    r1=run_baseline(data,**params)
    data['gyro_b']=np.tile([1.,5.,-1.],(len(t),1))
    r2=run_baseline(data,**params)
    np.testing.assert_allclose(r1['angle_deg'],r2['angle_deg'],equal_nan=True)
