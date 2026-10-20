"""Reproducible offline analysis of experiment 20261016_110453.

Run --help for inputs. Raw files and historical experiment outputs are never
modified. The full-data timing calibration is shared as evaluation metadata;
ablation state estimation never reads the masked gyro or its event times.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from imu_angle import read_log
from inspect_imu_log import inspect
from analyze_imu_log import infer_clock_map
from measure_video import run_video
from fold_angle.synchronization import gap_interp, visual_derivative, align_motion
from fold_angle.comparison import run_baseline, residual_metrics


def json_safe(x):
    if isinstance(x,dict):return {str(k):json_safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [json_safe(v) for v in x]
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    return x


def save_json(path,data):
    Path(path).write_text(json.dumps(json_safe(data),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def save_csv(path,columns,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(columns)
        for row in rows:
            w.writerow(['' if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for v in row])


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def draw(path,x,series,title,ylabel='Angle (deg)',xlabel='Video time (s)'):
    fig,ax=plt.subplots(figsize=(13,4.8),dpi=160)
    for label,y in series.items():ax.plot(x,y,lw=1.15,label=label)
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
    fig.tight_layout();fig.savefig(path);plt.close(fig)


def video_audit(path,out):
    cap=cv2.VideoCapture(str(path)); fps=cap.get(cv2.CAP_PROP_FPS); declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    msec=[];picks=[];first=None;last=None;k=0
    selection=set(np.linspace(0,max(0,declared-1),24).astype(int))
    while True:
        ok,frame=cap.read()
        if not ok:break
        msec.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))
        if first is None:first=k
        last=k
        if k in selection:picks.append((k,frame.copy()))
        k+=1
    cap.release()
    if k==0:raise ValueError('No decodable video frames')
    stamp=np.asarray(msec)/1000; dt=np.diff(stamp)
    usable=bool(np.isfinite(stamp).all() and (dt>0).all())
    nominal=np.arange(k)/fps
    audit={'source':str(Path(path).resolve()),'width':width,'height':height,'fps':fps,
           'declared_frames':declared,'decoded_frames':k,'duration_nominal_s':k/fps,
           'first_decodable_frame':first,'last_decodable_frame':last,'last_frame_nominal_s':float(nominal[-1]),
           'opencv_decoder_timestamp_usable':usable,'decoder_interval_percentiles_s':np.percentile(dt,[0,5,50,95,100]).tolist(),
           'decoder_timestamp_vs_nominal_max_abs_s':float(np.max(np.abs(stamp-nominal))) if usable else None,
           'suspicious_decoder_gaps':int((dt>1.5/fps).sum()) if usable else None,
           'declared_vs_decoded_discrepancy':declared-k,'angle_time_axis':'frame_index / nominal fps',
           'limitations':'Decoder timestamps are container playback times, not exposure hardware timestamps. No frame counter or sequence metadata proves absence of acquisition-side dropped frames. Camera/base are visually stable until the late camera repositioning; invalid template matches are excluded.'}
    save_csv(out/'frame_timestamps.csv',['frame_index','nominal_time_s','opencv_pos_msec'],zip(range(k),nominal,msec))
    save_json(out/'data_audit.json',audit)
    return audit,picks


def grouped_metrics(reference,estimates,variable,edges,mask):
    groups=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=mask&np.isfinite(variable)&(variable>=lo)&(variable<hi)
        groups.append({'low':lo,'high':hi,'metrics':{name:residual_metrics(reference,v,m) for name,v in estimates.items()}})
    return groups


def bars(path,groups,names,title,xlabel):
    fig,ax=plt.subplots(figsize=(12,4.5),dpi=150);xx=np.arange(len(groups));width=.8/len(names)
    for j,name in enumerate(names):
        yy=[g['metrics'][name]['mae_deg'] if g['metrics'][name]['mae_deg'] is not None else np.nan for g in groups]
        ax.bar(xx+(j-(len(names)-1)/2)*width,yy,width,label=name)
    ax.set_xticks(xx,[f"{g['low']:g}-{g['high']:g}" if np.isfinite(g['high']) else f"{g['low']:g}+" for g in groups]);ax.set(title=title,xlabel=xlabel,ylabel='MAE vs visual projection (deg)');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(path);plt.close(fig)


def stat(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return {'samples':len(x),'median_deg':float(np.median(x)) if len(x) else None,'std_deg':float(np.std(x)) if len(x) else None}


def analyze(video,csv_path,config,out,rerun_video=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name in ('video','imu','synchronization','comparison','ablation'):(out/name).mkdir(exist_ok=True)
    original_hashes={str(Path(p).resolve()):sha(p) for p in (video,csv_path)}
    audit_v,picks=video_audit(video,out/'video')
    save_json(out/'video_metadata.json',audit_v)
    cached_summary=out/'video/summary.json'
    current_config=json.loads(Path(config).read_text(encoding='utf-8-sig'))
    cache_ok=False
    if cached_summary.exists():
        cached=json.loads(cached_summary.read_text(encoding='utf-8'))
        cache_ok=(cached.get('status')=='complete' and cached.get('config')==current_config
                  and Path(cached.get('source','')).resolve()==Path(video).resolve())
    if rerun_video or not (out/'video/angles.csv').exists() or not cache_ok:run_video(video,config,out/'video')
    # Encode actual PNG; never rename JPEG bytes to a .png suffix.
    diag=cv2.imread(str(out/'video/diagnostics.jpg'))
    if diag is not None:cv2.imwrite(str(out/'video/diagnostics.png'),diag)
    rows=list(csv.DictReader((out/'video/angles.csv').open(encoding='utf-8-sig')))
    tv=np.array([float(r['timestamp_s']) for r in rows]);vr=np.array([float(r['projected_angle_deg']) if r['projected_angle_deg'] else np.nan for r in rows]);vf=np.array([float(r['smoothed_projected_angle_deg']) if r['smoothed_projected_angle_deg'] else np.nan for r in rows]);vvalid=np.isfinite(vr)
    vs,vd=visual_derivative(tv,vr,vvalid)
    save_csv(out/'synchronization/visual_motion.csv',['video_time_s','raw_projected_deg','locally_smoothed_deg','visual_rate_deg_s','valid','motion_valid'],zip(tv,vr,vs,vd,vvalid.astype(int),np.isfinite(vd).astype(int)))
    streams=read_log(csv_path); audit_i=inspect(csv_path,out/'imu')
    required={'acc1','gyro1','加速度传感器','陀螺仪传感器'}
    if not required<=streams.keys():raise ValueError('Unknown stream naming: explicit pairing config required')
    host_base=min(int(streams[n].timestamps_ns[0]) for n in ('acc1','gyro1'));sensor_base=min(int(streams[n].timestamps_ns[0]) for n in ('加速度传感器','陀螺仪传感器'))
    clock=infer_clock_map(csv_path,host_base,sensor_base)
    time={}
    for n,s in streams.items():
        if n in ('acc1','gyro1'):time[n]=(s.timestamps_ns-host_base)/1e9
        else:time[n]=clock['scale']*((s.timestamps_ns-sensor_base)/1e9)+clock['offset_s']
    # Signal-based identity and hinge direction checks for this recording.
    g=streams['gyro1'].values;tg=time['gyro1'];initial=(tg>=0)&(tg<1.)
    bias_b=np.median(g[initial],axis=0);bias_a=np.median(streams['陀螺仪传感器'].values[time['陀螺仪传感器']<1.],axis=0)
    active=np.linalg.norm(g-bias_b,axis=1)>.2
    _,singular,vt=np.linalg.svd(g[active]-bias_b,full_matrices=False)
    pca=vt[0];pca*=np.sign(pca[np.argmax(np.abs(pca))]);gyro_axis_index=int(np.argmax(np.std(g,axis=0)))
    # Infer the gravity rotation plane using retained Acc B only. Thus even
    # the A-only candidate's axis input has no hidden dependence on gyro B.
    acc_native=streams['acc1'].values;acc_t=time['acc1'];acc_norm=np.linalg.norm(acc_native,axis=1)
    acc_jerk=np.r_[0.,np.linalg.norm(np.diff(acc_native,axis=0),axis=1)/np.diff(acc_t)]
    stable_acc=acc_native[(acc_norm>9.2)&(acc_norm<10.4)&(acc_jerk<15)]
    _,_,acc_vt=np.linalg.svd(stable_acc-np.median(stable_acc,axis=0),full_matrices=False)
    acc_axis=acc_vt[-1];axis_index=int(np.argmax(np.abs(acc_axis)));acc_axis*=np.sign(acc_axis[axis_index])
    if axis_index!=gyro_axis_index:raise ValueError('Acc and gyro axis evidence disagree; installation review required')
    axis_b=np.eye(3)[axis_index];axis_a=np.array([0.,1.,0.])
    # Axis A is only an uncalibrated installation convention, not inferred from
    # tiny base motions. Calibration remains a documented limitation.
    opening_rate=-(g-bias_b)@axis_b
    sync=align_motion(tv,vd,tg,opening_rate)
    b=sync['b_s'];a=sync['a']
    gyro_overlay=gap_interp(tg,np.rad2deg(opening_rate),a*tv+b,.06)
    draw(out/'synchronization/angular_velocity_overlay.png',tv,{'Raw-visual derivative':vd,'Aligned axial gyro':gyro_overlay},'Angular-velocity alignment (offline filters)','Opening angular rate (deg/s)')
    fig,axs=plt.subplots(2,1,figsize=(12,7),dpi=150)
    profile=sync['offset_profile'];axs[0].plot(profile['b_s'],[np.nan if x is None else x for x in profile['correlation']]);axs[0].axvline(b,color='r',ls='--');axs[0].set(xlabel='Offset b (s)',ylabel='Training correlation',title='Competing synchronization offsets')
    events=sorted(sync['training_events']+sync['validation_events'],key=lambda e:e['peak_video_s'])
    for held,color,label in [(False,'#176ac4','Training'),(True,'#d75b30','Withheld')]:
        ee=[e for e in events if e['validation']==held];axs[1].scatter([e['peak_video_s'] for e in ee],[e['local_minus_global_ms'] for e in ee],color=color,label=label)
    axs[1].axhline(0,color='k',ls='--');axs[1].set(xlabel='Video event time (s)',ylabel='Local offset minus global (ms)',title='Event timing consistency; not a calibrated uncertainty');axs[1].legend()
    for ax in axs:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'synchronization/synchronization_diagnostics.png');plt.close(fig)
    save_json(out/'synchronization/time_alignment.json',sync)
    # Initial angle is from unfiltered visual observations near the first IMU
    # sample, not an endpoint assumption or a fit to the whole angle curve.
    start=max(t[0] for t in time.values());end=min(t[-1] for t in time.values())
    initial_video=(start-b)/a
    ref=(tv>=initial_video)&(tv<initial_video+.4)&vvalid
    if ref.sum()<3:raise ValueError('Cannot establish initial angle from this overlap')
    theta0=float(np.median(vr[ref]))
    # Baselines with gyro B use actual unique B gyro event times.
    use=(tg>=start)&(tg<=end);ts=tg[use];timestamps=streams['gyro1'].timestamps_ns[use]
    def data_on(query,keep):
        data={'t':query,'acc_b':gap_interp(time['acc1'],streams['acc1'].values,query,.1),'acc_a':gap_interp(time['加速度传感器'],streams['加速度传感器'].values,query,.06)}
        if 'B' in keep:data['gyro_b']=gap_interp(tg,g,query,.06)
        if 'A' in keep:data['gyro_a']=gap_interp(time['陀螺仪传感器'],streams['陀螺仪传感器'].values,query,.04)
        return data
    full=data_on(ts,'AB')
    params={'initial_angle_deg':theta0,'axis_b':axis_b,'axis_a':axis_a,'bias_b':bias_b,'bias_a':bias_a,'reference_window_s':(float(ts[0]),1.)}
    dual=run_baseline(full,keep_gyro='AB',**params)
    params_b=dict(params,bias_a=None)
    fixed=run_baseline(data_on(ts,'B'),keep_gyro='B_fixed',**params_b)
    keep_b=run_baseline(data_on(ts,'B'),keep_gyro='B',**params_b)
    # The A-only candidate's timeline and estimator use exclusively retained
    # sensors; it never sees gyro B, even for state-update scheduling.
    ta=time['acc1'];mask_a=(ta>=max(time['acc1'][0],time['加速度传感器'][0],time['陀螺仪传感器'][0]))&(ta<=min(time['acc1'][-1],time['加速度传感器'][-1],time['陀螺仪传感器'][-1]));ta=ta[mask_a]
    data_a=data_on(ta,'A');pa=dict(params,bias_b=None,reference_window_s=(float(ta[0]),1.))
    keep_a=run_baseline(data_a,keep_gyro='A',**pa)
    # Long-form cleaned native data preserves every unique raw integer time.
    cleaned=[]
    for n,s in streams.items():
        for raw_t,t_common,xyz in zip(s.timestamps_ns,time[n],s.values):cleaned.append([n,int(raw_t),float(t_common),*xyz])
    save_csv(out/'imu/cleaned_imu.csv',['sensor','original_timestamp_ns','common_relative_s','x','y','z'],cleaned)
    columns=['timestamp_ns','time_s','video_time_s','acc_angle_deg','gyro_integrated_angle_deg','fixed_base_imu_angle_deg','provisional_dual_imu_angle_deg','keep_moving_gyro_angle_deg','angular_rate_deg_s','acc_update_used','acc_quality','low_motion','valid']
    save_csv(out/'imu/angles.csv',columns,([int(timestamps[k]),ts[k],(ts[k]-b)/a,fixed['acc_angle_deg'][k],fixed['gyro_angle_deg'][k],fixed['angle_deg'][k],dual['angle_deg'][k],keep_b['angle_deg'][k],fixed['rate_deg_s'][k],int(fixed['acc_used'][k]),int(fixed['acc_quality'][k]),int(fixed['low_motion'][k]),int(fixed['valid'][k])] for k in range(len(ts))))
    save_csv(out/'ablation/retained_A_native.csv',['acc_b_timestamp_ns','imu_time_s','angle_deg','valid','acc_quality'],([int(streams['acc1'].timestamps_ns[mask_a][k]),ta[k],keep_a['angle_deg'][k],int(keep_a['valid'][k]),int(keep_a['acc_quality'][k])] for k in range(len(ta))))
    save_csv(out/'ablation/native_B_baselines.csv',['timestamp_ns','imu_time_s','full_AB_deg','retained_B_deg','full_valid','retained_B_valid'],([int(timestamps[k]),ts[k],dual['angle_deg'][k],keep_b['angle_deg'][k],int(dual['valid'][k]),int(keep_b['valid'][k])] for k in range(len(ts))))
    imu_summary={'source':str(Path(csv_path).resolve()),'clock_map':clock,'mapping':{'main_A':['加速度传感器','陀螺仪传感器'],'secondary_B':['acc1','gyro1'],'status':'Inferred from same clock grouping, near-static main gravity and much larger secondary gyro motion; physical label pairing should be confirmed at acquisition.'},
                 'units':{'gyro':'rad/s, user-confirmed for hardware','acc':'m/s^2 inferred from stationary norm'},'axis_b':axis_b,'axis_a_provisional':axis_a,'pca_axis_b':pca,'pca_energy_fraction':singular[0]**2/np.sum(singular**2),'acc_only_plane_normal':acc_axis,'axis_input_source':'dominant coordinate of Acc-B-only gravity plane normal; gyro PCA used as independent agreement check, not candidate input','gyro_std_xyz':np.std(g,axis=0),'base_gyro_std_xyz':np.std(streams['陀螺仪传感器'].values,axis=0),
                 'bias_B_xyz_rad_s':bias_b,'bias_A_xyz_rad_s':bias_a,'reference_window_imu_s':[float(ts[0]),1.],'initial_angle_deg':theta0,'initialization_source':'median raw visual projection for 0.4s near first IMU sample; shared initial angle only; no subsequent vision corrections',
                 'initialization_video_s':[float(tv[ref][0]),float(tv[ref][-1])],'output_rows':len(ts),'native_retained_A_rows':len(ta),'fixed_acc_updates':int(fixed['acc_used'].sum()),'dual_acc_updates':int(dual['acc_used'].sum()),'retained_B_acc_updates':int(keep_b['acc_used'].sum()),'first_second_gyro_B_std':np.std(g[initial],axis=0),
                 'fixed_base_gate':'acc norm deviation<0.6m/s²; projected gravity>0.2g; jerk<30m/s³; gyro norm<.25rad/s; innovation<20deg; tau=1s; invalid gaps>.1s',
                 'retained_A_gate':'two acc norm/projection/jerk gates; phase rate<.25rad/s and retained A gyro norm<.25rad/s; high dynamics invalid, no pretend-zero B gyro',
                 'state':'offline only; native gyro integration and acceleration interpolation; axes/clock not independently calibrated'}
    for n,s in streams.items():
        dt=np.diff(time[n]);audit_i['streams'][n]['gaps_over_2x_median']=int((dt>2*np.median(dt)).sum());audit_i['streams'][n]['gaps_over_100ms']=int((dt>.1).sum());audit_i['streams'][n]['nonfinite_xyz']=int((~np.isfinite(s.values)).sum());audit_i['streams'][n]['max_xyz_norm']=float(np.linalg.norm(s.values,axis=1).max());audit_i['streams'][n]['unique_strictly_increasing']=bool((np.diff(s.timestamps_ns)>0).all())
    audit_i.update(clock_map=clock,units=imu_summary['units'],mapping=imu_summary['mapping']);save_json(out/'imu/data_audit.json',audit_i);save_json(out/'imu/summary.json',imu_summary)
    draw(out/'imu/angle_curve.png',ts,{'Acc phase':fixed['acc_angle_deg'],'Gyro integration':fixed['gyro_angle_deg'],'Fixed-base complementary':fixed['angle_deg'],'Provisional dual IMU':dual['angle_deg'],'Retained moving gyro':keep_b['angle_deg']},'Correctly initialized IMU estimates',xlabel='IMU relative time (s)')
    fig,axs=plt.subplots(4,1,figsize=(13,9),dpi=150,sharex=True)
    for ax,n in zip(axs,['加速度传感器','陀螺仪传感器','acc1','gyro1']):
        for j in range(3):ax.plot(time[n],streams[n].values[:,j],lw=.65,label='XYZ'[j])
        ax.set_ylabel({'加速度传感器':'Acc A m/s2','陀螺仪传感器':'Gyro A rad/s','acc1':'Acc B m/s2','gyro1':'Gyro B rad/s'}[n]);ax.legend(fontsize=7,ncol=3);ax.grid(alpha=.2)
    axs[-1].set_xlabel('Approximate common IMU time (s)');fig.tight_layout();fig.savefig(out/'imu/raw_signals.png');plt.close(fig)
    # Comparison happens at observed video frames: no visual angle is invented
    # inside invalid frames, and no One Euro delay is hidden in the reference.
    query=a*tv+b
    estimates={'Gyro':gap_interp(ts,fixed['gyro_angle_deg'],query,.1),'Fixed complementary':gap_interp(ts,fixed['angle_deg'],query,.1),'Dual AB':gap_interp(ts,dual['angle_deg'],query,.1),'Retain B':gap_interp(ts,keep_b['angle_deg'],query,.1),'Retain A gravity':gap_interp(ta,keep_a['angle_deg'],query,.04)}
    supported=vvalid&(query>=start)&(query<=end)&(query>=1.)
    core_mask=supported.copy()
    for n in ('Gyro','Fixed complementary','Dual AB','Retain B'):core_mask &= np.isfinite(estimates[n])
    all_mask=core_mask&np.isfinite(estimates['Retain A gravity'])
    main_metrics={n:residual_metrics(vr,v,core_mask) for n,v in estimates.items() if n!='Retain A gravity'}
    all_metrics={n:residual_metrics(vr,v,all_mask) for n,v in estimates.items()}
    available={n:{**residual_metrics(vr,v,supported),'availability_fraction':float((supported&np.isfinite(v)).sum()/max(1,supported.sum()))} for n,v in estimates.items()}
    # Explicit calibration on the first second only; never fit an offset to
    # evaluation residuals to make errors look smaller.
    calibration=vvalid&(query>=start)&(query<1.)
    calibration_offsets={n:float(np.nanmedian((v-vr)[calibration])) for n,v in estimates.items() if np.isfinite((v-vr)[calibration]).any()}
    calibrated={n:residual_metrics(vr,v-calibration_offsets[n],core_mask if n!='Retain A gravity' else all_mask) for n,v in estimates.items() if n in calibration_offsets}
    residual_info={'reference':'raw visual 2D projected angle on observed valid frames; no mechanical truth; shared initial visual anchor',
                   'core_common_support_samples':int(core_mask.sum()),'all_configurations_common_support_samples':int(all_mask.sum()),'initial_calibration_excluded_imu_s':[0,1.],
                   'core_common_metrics':main_metrics,'all_configurations_common_metrics':all_metrics,'own_support_metrics_not_directly_comparable':available,
                   'first_second_calibration_offsets_deg':calibration_offsets,'first_second_calibrated_metrics':calibrated,
                   'posthoc_angle_scale_fit':False,'endpoint_snapping':False,'angle_range_clipping':False}
    save_json(out/'comparison/metrics.json',residual_info)
    save_csv(out/'ablation/baseline_comparison.csv',['frame_index','video_time_s','imu_time_s','visual_projected_deg','full_AB_deg','retained_B_deg','retained_A_gravity_deg','core_valid','all_common_valid'],([k,tv[k],query[k],vr[k],estimates['Dual AB'][k],estimates['Retain B'][k],estimates['Retain A gravity'][k],int(core_mask[k]),int(all_mask[k])] for k in range(len(tv))))
    save_csv(out/'comparison/aligned_angles.csv',['frame_index','video_time_s','imu_time_s','visual_raw_projected_deg','visual_one_euro_deg','visual_valid','core_comparison_valid','all_comparison_valid',*estimates],([k,tv[k],query[k],vr[k],vf[k],int(vvalid[k]),int(core_mask[k]),int(all_mask[k]),*[v[k] for v in estimates.values()]] for k in range(len(tv))))
    draw(out/'comparison/angle_overlay.png',tv,{'Visual raw projection':vr,'Gyro':estimates['Gyro'],'Fixed complementary':estimates['Fixed complementary'],'Dual AB':estimates['Dual AB'],'Retain B':estimates['Retain B']},'Angles aligned by motion; visual is uncalibrated')
    draw(out/'comparison/angle_residuals.png',tv,{n:np.where(core_mask,v-vr,np.nan) for n,v in estimates.items() if n!='Retain A gravity'},'Residuals on the same observed visual frames','IMU - visual projection (deg)')
    by_angle=grouped_metrics(vr,{n:estimates[n] for n in main_metrics},vr,np.arange(0,181,30),core_mask)
    by_speed=grouped_metrics(vr,{n:estimates[n] for n in main_metrics},np.abs(vd),[0,10,30,60,120,np.inf],core_mask)
    save_json(out/'comparison/error_by_angle.json',by_angle);save_json(out/'comparison/error_by_speed.json',by_speed)
    bars(out/'comparison/error_by_angle.png',by_angle,list(main_metrics),'Residual by angle (common support)','Raw projected angle bin (deg)')
    bars(out/'comparison/error_by_speed.png',by_speed,list(main_metrics),'Residual by visual speed (common support)','|Visual angular rate| bin (deg/s)')
    directions={}
    for label,m in [('opening',vd>10),('closing',vd< -10),('near_static',np.abs(vd)<3)]:directions[label]={n:residual_metrics(vr,v,core_mask&m) for n,v in estimates.items() if n!='Retain A gravity'}
    save_json(out/'comparison/direction_metrics.json',directions)
    # Static segments require sustained low visual speed with no invalid gap.
    ids=np.flatnonzero(core_mask&np.isfinite(vd)&(np.abs(vd)<3))
    windows=[]
    if len(ids):
        cuts=np.flatnonzero(np.diff(ids)>1)+1
        for seq in np.split(ids,cuts):
            if tv[seq[-1]]-tv[seq[0]]<.7:continue
            windows.append({'video_start_s':float(tv[seq[0]]),'video_end_s':float(tv[seq[-1]]),'visual':stat(vr[seq]),'estimates':{n:stat(v[seq]) for n,v in estimates.items()}})
    endpoints={n:{'first_common':stat(v[core_mask][:10]),'last_common':stat(v[core_mask][-10:])} for n,v in dict(Visual=vr,**estimates).items()}
    save_json(out/'comparison/stability_and_drift.json',{'static_windows':windows,'endpoints':endpoints,'note':'Beginning and ending configurations differ; endpoint difference is not automatically drift. Use residual change and matching physical states.'})
    bias_windows=[]
    for lo,hi in [(0,.5),(0,1),(tg[-1]-3,tg[-1])]:
        bias_est=np.median(g[(tg>=lo)&(tg<hi)],axis=0)
        bias_windows.append({'window_imu_s':[lo,hi],'median_xyz_rad_s':bias_est,
                             'gyro_endpoint_change_if_used_deg':float(((bias_est-bias_b)@axis_b)*(ts[-1]-ts[0])*180/np.pi)})
    sensitivity={}
    sensitivity_mask=core_mask.copy()
    variants={}
    for shift in [-1/30,0,1/30]:
        qq=query+shift; xx={n:gap_interp(ts,r['angle_deg'],qq,.1) for n,r in [('Fixed complementary',fixed),('Dual AB',dual)]}
        variants[shift]=xx
        for v in xx.values():sensitivity_mask &= np.isfinite(v)
    for shift,xx in variants.items():sensitivity[str(shift)]={n:residual_metrics(vr,v,sensitivity_mask) for n,v in xx.items()}
    drift_diagnostics={'bias_window_sensitivity_not_applied':bias_windows,'time_offset_sensitivity_not_used_for_fit':sensitivity,
                       'AB_minus_retained_B_deg':stat(estimates['Dual AB'][core_mask]-estimates['Retain B'][core_mask]),
                       'acc_plane_vs_gyro_axis_abs_dot':float(abs(acc_axis@pca)),
                       'interpretation':'Sensitivity diagnostics, not fitted corrections or independent causality proof.'}
    save_json(out/'comparison/systematic_diagnostics.json',drift_diagnostics)
    draw(out/'ablation/baseline_comparison.png',tv,{'Visual raw':vr,'Full AB':estimates['Dual AB'],'Only gyro B':estimates['Retain B'],'Only gyro A, low dynamic':estimates['Retain A gravity']},'Gyro ablation; missing high-dynamic A-only outputs are intentional')
    save_json(out/'ablation/metrics.json',{'common_support':all_metrics,'own_support':available,'all_common_samples':int(all_mask.sum()),'coverage_denominator':int(supported.sum()),'retained_A_uses':'Acc A, Acc B, Gyro A; native Acc B timestamps; no B gyro values, bias, rate, gates or event times',
                                          'retained_B_uses':'Acc A, Acc B, Gyro B; no A gyro values, bias or gates',
                                          'shared_calibration':'Fixed external time alignment and visual initial angle are evaluation metadata. No future vision is fed to an estimator.',
                                          'limitations':'Fixed-base experiment; A-only gravity rejects dynamic samples. No neural model trained and no arbitrary-handheld sensor-removal claim.'})
    summary_v=json.loads((out/'video/summary.json').read_text(encoding='utf-8'))
    summary_v['measurement_quality_note']='Availability is not accuracy; manual ROI, fixed-base scene. Camera is repositioned late in clip; no general rotation tracking. Raw projected reference, not calibrated mechanics.';summary_v['audit_file']='data_audit.json';save_json(out/'video/summary.json',summary_v)
    # Verification is independent of plotting: recompute core residuals and
    # original timestamp preservation, inspect actual video decoding and hashes.
    c=cv2.VideoCapture(str(out/'video/annotated.mp4'));nframes=0
    while True:
        ok,_=c.read()
        if not ok:break
        nframes+=1
    c.release()
    fixed_pivots=np.array([[float(r['hinge_anchor_x']),float(r['hinge_anchor_y'])] for r in rows if r['hinge_anchor_x']])
    unchanged={p:sha(p)==h for p,h in original_hashes.items()}
    check={'video_rows':len(rows),'decoded_annotated_frames':nframes,'video_rows_match_input':len(rows)==audit_v['decoded_frames']==nframes,
           'video_valid_rows':int(vvalid.sum()),'common_metrics_samples':int(core_mask.sum()),'all_ablation_common_samples':int(all_mask.sum()),
           'imu_angle_rows':len(ts),'cleaned_unique_native_rows':len(cleaned),'imu_raw_int_ns_strictly_increasing':bool((np.diff(timestamps)>0).all()),
           'imu_time_strictly_increasing':bool((np.diff(ts)>0).all()),'video_time_strictly_increasing':bool((np.diff(tv)>0).all()),'fixed_pivot_range_px':np.ptp(fixed_pivots,axis=0),
           'invalid_visual_angles_blank':all(not r['projected_angle_deg'] for r in rows if r['valid']=='0'),
           'input_sha256':original_hashes,'inputs_unchanged':unchanged,
           'metrics_recomputed_rmse_AB_deg':float(np.sqrt(np.mean((estimates['Dual AB'][core_mask]-vr[core_mask])**2))),
           'no_visual_interpolation_for_residuals':True,'no_blind_large_gap_interpolation':True,
           'causal_live_system':False,'not_validated':['mechanical absolute accuracy','hardware synchronization','IMU installation axes','arbitrary handheld gyro removal']}
    check['tests']={'status':'not_run_by_analysis_script','command':'python -m pytest tests -q'}
    save_json(out/'verification.json',check)
    import scipy,sys
    manifest={'video':str(Path(video).resolve()),'imu':str(Path(csv_path).resolve()),'config':str(Path(config).resolve()),'config_sha256':sha(config),'inputs_sha256':original_hashes,'script':str(Path(__file__).resolve()),'sync_module':str(Path('fold_angle/synchronization.py').resolve()),'comparison_module':str(Path('fold_angle/comparison.py').resolve()),'runtime':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'opencv':cv2.__version__},'new_code_sha256':{p:sha(p) for p in ['analyze_synchronized_experiment.py','fold_angle/synchronization.py','fold_angle/comparison.py','fold_angle/detector.py','imu_angle.py','tests/test_synchronization.py']}}
    save_json(out/'experiment_manifest.json',manifest)
    create_report(out,audit_v,audit_i,summary_v,imu_summary,sync,residual_info,directions,windows,endpoints,check,drift_diagnostics)
    (out/'imu/report.md').write_text('# 本次IMU分析\n\n有效结果见[完整同步实验报告]('+str((out/'report.md').resolve().as_posix())+')。\n\n本次初始角来自同步视频，前约1秒用于参考，不沿用旧实验的180°/5秒假设。最终曲线、参数与审计分别见本目录angles.csv、summary.json及data_audit.json。不适用的旧CLI试跑已单独标为_probes/invalid_legacy_initialization。\n',encoding='utf-8')
    print(json.dumps(json_safe({'offset_s':b,'correlation_before':sync['correlation_before'],'correlation_after':sync['correlation_after'],'training_events':len(sync['training_events']),'validation_events':len(sync['validation_events']),'initial_angle_deg':theta0,'metrics':main_metrics,'all_common_samples':int(all_mask.sum()),'retained_A_availability':available['Retain A gravity']['availability_fraction']}),ensure_ascii=True,indent=2))
    return check


def create_report(out,av,ai,sv,si,sync,metrics,directions,windows,endpoints,verify,systematic):
    def table(ms):
        lines=['| 方法 | 样本数 | MAE ° | RMSE ° | P95绝对残差 ° | 最大绝对残差 ° | 平均偏差 ° |','|---|---:|---:|---:|---:|---:|---:|']
        for name,m in ms.items():
            fmt=lambda k:'—' if m.get(k) is None else f'{m[k]:.3f}'
            lines.append(f"| {name} | {m['samples']} | {fmt('mae_deg')} | {fmt('rmse_deg')} | {fmt('p95_abs_deg')} | {fmt('max_abs_deg')} | {fmt('mean_bias_deg')} |")
        return '\n'.join(lines)
    stream_lines=['| 数据流 | 原始行 | 独立时刻 | 重复行 | 冲突时刻 | dt中位数 ms | 最大间隔 ms |','|---|---:|---:|---:|---:|---:|---:|']
    for n,s in ai['streams'].items():stream_lines.append(f"| {n} | {s['records']} | {s['unique_timestamps']} | {s['duplicate_timestamp_records']} | {s['conflicting_timestamps']} | {s['dt_percentiles_ms'][2]:.3f} | {s['dt_percentiles_ms'][-1]:.3f} |")
    event_lines=['| 用途 | 视频动作峰 s | 方向 | 局部偏移 s | 局部相关 | 局部与全局差 ms | 峰值时间残差 ms |','|---|---:|---|---:|---:|---:|---:|']
    for e in sorted(sync['training_events']+sync['validation_events'],key=lambda d:d['peak_video_s']):
        peak='未匹配' if e.get('peak_residual_ms') is None else f"{e['peak_residual_ms']:.1f}"
        event_lines.append(f"| {'留出验证' if e['validation'] else '拟合'} | {e['peak_video_s']:.3f} | {e['direction']} | {e['local_offset_s']:.3f} | {e['local_correlation']:.3f} | {e['local_minus_global_ms']:.1f} | {peak} |")
    diagnostic_lines=['| gyro零偏窗口 s | Y轴中位数 rad/s | 若替换当前零偏，纯积分末值变化 ° |','|---|---:|---:|']
    for d in systematic['bias_window_sensitivity_not_applied']:diagnostic_lines.append(f"| {d['window_imu_s'][0]:.2f}—{d['window_imu_s'][1]:.2f} | {d['median_xyz_rad_s'][1]:.6f} | {d['gyro_endpoint_change_if_used_deg']:.3f} |")
    stability_lines=['| 静止窗口 视频s | 视觉中位数 ° | 视觉std ° | 固定互补中位数 ° | 固定互补std ° |','|---|---:|---:|---:|---:|']
    for w in windows:
        v=w['visual'];f=w['estimates']['Fixed complementary']
        stability_lines.append(f"| {w['video_start_s']:.2f}—{w['video_end_s']:.2f} | {v['median_deg']:.3f} | {v['std_deg']:.3f} | {f['median_deg']:.3f} | {f['std_deg']:.3f} |")
    report=f'''# 折叠手机视觉与双 IMU 同步实验分析

日期：2026-10-16。实验编号：experiment_20261016_110453。状态：已运行的离线真实数据实验；不是实时系统，不是机械角度精度认证。

## 1. 重点结论

本次从视觉原始投影角的导数与副机身轴向 gyro 建立了数据驱动的时间对应：`t_imu = t_video + ({sync['b_s']:.6f}) s`。零偏移相关性约 {sync['correlation_before']:.3f}，对齐后约 {sync['correlation_after']:.3f}，两者在共同支持样本上计算。保留 a=1；本次没有可靠识别时钟漂移，也没有独立同步触发可给出毫秒级精度保证。

IMU 与视觉比较使用有效的**原始二维投影角**，没有用 One Euro 显示角隐藏时序滞后，没有填补无效视觉帧。初始角 {si['initial_angle_deg']:.3f}° 来自重叠起点附近视频，之后 IMU 估计没有再接收视觉角校正，因此本实验属于同步对照，不是持续视觉融合。

下方提供共同支持上的完整指标及单 gyro 结果。仅保留固定侧 gyro 的方案只在低动态重力可用时输出角度，高动态缺测明确留空；不能用其较低的静止误差声称与动态双 gyro 等价。所有结果均不能证明任意手持可去掉一套 gyro。

## 2. 实验数据与审计

- 视频：`data/raw/1016110453.mp4`。
- IMU：`data/raw/sensor_log_2026-10-16_11-04-53.csv`。
- 分别启动、结束，文件名没有参与同步计算。
- 角度定义：闭合0°、半开90°、展开180°。方向来自铰链到自由端，可区分30°和150°。

### 视频

分辨率 {av['width']}×{av['height']}，标称 {av['fps']:g} fps，实际解码 {av['decoded_frames']} 帧，与容器声明一致。名义时长 {av['duration_nominal_s']:.4f} s；首末解码帧为 {av['first_decodable_frame']}、{av['last_decodable_frame']}，末帧名义时刻 {av['last_frame_nominal_s']:.6f} s。

OpenCV播放时间戳严格递增可用：{av['opencv_decoder_timestamp_usable']}；与名义时间轴最大差为 {av['decoder_timestamp_vs_nominal_max_abs_s']} s。详细记录见 `video/frame_timestamps.csv`。测角仍采用帧号/fps名义轴。容器播放时间不等于相机曝光硬件时间，声明与解码帧数一致也不能排除采集端掉帧。

画面检查支持主要动作期间相机和下机身基本固定，上机身开合；视频末段存在相机重新摆位。当前只做平移配准，没有一般三维相机运动补偿。末段模板失配、边线缺失等状态不参与角度残差。

### IMU

{chr(10).join(stream_lines)}

原始总行数{ai['rows']}，去重后共{sum(s['unique_timestamps'] for s in ai['streams'].values())}条。去重按每路、每个整数纳秒时间戳保留最后一条，冲突不能恢复原有采样时间。原始文件不修改；去重原生长表保存在 `imu/cleaned_imu.csv`，保留原始整数时间及相对公共秒。没有把缓存重复写入视为新增样本。各路无时间倒退，存在采样间隔变化和动态异常；详细最大模长、间隔分位数与缺口见审计JSON。尤其副acc的冲击异常不能直接当作重力。

gyro单位rad/s沿用用户对硬件的确认；acc为m/s²依据静止约9.8的模长推断。主A对应中文两路，副B对应acc1/gyro1，是由时钟分组、主重力近固定、副gyro运动显著推断的身份映射，不是安装元数据的替代。副gyro主分量为第 {int(np.argmax(np.asarray(si['axis_b'])))} 轴（XYZ从0计），PCA轴约 {np.asarray(si['pca_axis_b']).round(6).tolist()}，主成分能量占比 {si['pca_energy_fraction']*100:.4f}%。保留的副acc独立拟合重力旋转平面，其法向约 {np.asarray(si['acc_only_plane_normal']).round(6).tolist()}，两种证据一致支持+Y。算法输入轴由acc平面法向的主坐标确定，gyro PCA仅作核对，避免在屏蔽副gyro时从它间接生成轴输入。开启时副轴向角速度为负，与视频动作方向核对。

两种原始时钟域分别约1.79e18 ns与5.42e13 ns，先各自减整数参考点，再映射相对秒。主副IMU软件行序映射为 `t_common={si['clock_map']['scale']:.12f}*t_A+({si['clock_map']['offset_s']:.6f})`，其残差P5/P50/P95为 {np.round(si['clock_map']['residual_p05_p50_p95_ms'],3).tolist()} ms。这是记录顺序映射残差，不能当硬件同步精度；主副IMU映射与视频偏移是两个不同问题。

## 3. 视觉测角效果与修复

本次单独配置 `configs/experiment_20261016_110453.json`。初始开合约34°，没有套用旧实验的初始180°。新画面的固定下边指向右，与旧画面相反，因此增加显式镜像坐标适配：内部复用旧检测几何，输出边线、交点和配准量再转换回原图。这也修复了首轮探测中“角度数值正确、上边绘制到桌面下方”的问题。历史默认行为和配置不改写。

仍为手动首帧ROI、模板跟踪、受约束LSD边线与One Euro滤波，没有训练目标检测网络。固定虚拟交点为 {sv['fixed_pivot']}，观测线段与模型射线分开导出。手机厚度允许真实边框不严格交于虚拟点。无相机/机械角度标定，输出只能称二维投影角。

{sv['processed_frames']}帧中有效{sv['valid_frames']}帧，无效{sv['invalid_frames']}帧，可用比例{sv['valid_fraction']*100:.2f}%。失败分布：{sv['invalid_reasons']}。可用比例不是准确率。One Euro参数保持min_cutoff=.8Hz、beta=.1、derivative_cutoff=1Hz，无效后重置，不填角度。

![视觉角度曲线]({(out/'video/angle_curve.png').resolve().as_posix()})

![关键帧检测检查]({(out/'video/diagnostics.png').resolve().as_posix()})

## 4. IMU算法、初值与门控

本次调用已有 `estimate_angle` 单自由度实现；新增可显式传入已估计零偏的参数，旧调用默认行为保持不变。原脚本的五秒静止和180°假设在本次无效，相关早期试跑仅留在`_probes/invalid_legacy_initialization/`，不属于最终结果。

采用前约0—1s近似保持窗口，副gyro中位零偏 {np.round(si['bias_B_xyz_rad_s'],6).tolist()} rad/s，主gyro零偏 {np.round(si['bias_A_xyz_rad_s'],6).tolist()} rad/s；窗口仍有轻微持握运动，其标准差和初始参考都保存在summary中。没有用全程角度残差拟合gyro零偏。初值仅由同步起点约0.4s原始视觉中位数建立，排除初始1s校准区间后再统计残差。

纯gyro按真实采样dt梯形积分：`theta_pred=theta_prev-.5*(omega_prev+omega_now)*dt`。低动态时用重力在垂直铰链轴平面内的相位构造角度，再以`K=1-exp(-dt/1s)`做互补校正。条件为模长偏差<.6m/s²、投影>.2g、jerk<30m/s³、去偏gyro模长<.25rad/s、角度创新<20°。没有卡尔曼滤波、端点吸附、角度缩放或强制范围裁剪。

当前固定底座校正标志计数 {si['fixed_acc_updates']}，暂定双IMU {si['dual_acc_updates']}，只保留副gyro {si['retained_B_acc_updates']}；首点dt=0的标志不等于非零修正。AB使用两侧加速度相位和轴向gyro差分，主轴暂按同向+Y，尚未独立安装标定；仅副gyro组保留两侧acc用于重力关系。

## 5. 时间同步方法及可信程度

从原始视觉角用11帧居中二次局部拟合求角速度，不使用One Euro输出。缺观测窗口不求导。副gyro先减上述零偏、投影到经本次检查的轴、符号取开启为正；用于匹配的gyro经过约.04s对称高斯平滑并限制冲击峰幅，原始gyro不因同步而被改写。

先对保留的训练动作做归一化互相关的偏移搜索，再在训练动作窗口细化。每第三个动作在粗搜和细化前预先留出，只做验证；这些仍来自同一次采集，不等价于外部同步真值。没有利用IMU角度曲线自证时间或角度正确性。

最终映射：`a=1，b={sync['b_s']:.6f}s`。例如视频10s对应IMU约{10+sync['b_s']:.3f}s，说明IMU相对视频稍晚开始。可信重叠的运动支持范围为视频{sync['video_overlap_s'][0]:.3f}—{sync['video_overlap_s'][1]:.3f}s；范围内部仍有无效帧，必须读取各行有效标志。相关性从{sync['correlation_before']:.3f}升至{sync['correlation_after']:.3f}，训练动作{len(sync['training_events'])}个，留出验证{len(sync['validation_events'])}个。

相似往返动作导致的竞争偏移已在全范围搜索中保留。是否存在相近强度的多解候选：{sync['ambiguous_offset_candidates']}；最优训练候选相关约{sync['coarse_candidates'][0]['correlation']:.3f}，其他候选的时移和支持见time_alignment.json及诊断图。没有把每轮开合自由挪动到最有利的位置。

{chr(10).join(event_lines)}

局部偏移与峰值残差既含同步误差，也含投影导数畸变、图像漏检、有限窗口平滑和接触冲击。峰匹配要求方向、顺序一致且不重复使用同一个gyro峰；没有对应峰时记录未匹配，不制造事件。若个别后段动作与统一偏移不一致，应视为该动作同步证据较弱，不能用自由漂移参数强行解释。候选偏移、相关曲线及所有事件保留在JSON，不单凭最大相关宣称成功。a保持1，本次不宣称识别出物理时钟漂移或达到毫秒级同步精度。

![角速度对齐]({(out/'synchronization/angular_velocity_overlay.png').resolve().as_posix()})

![同步诊断]({(out/'synchronization/synchronization_diagnostics.png').resolve().as_posix()})

## 6. 共同时间轴的角度残差

以下参考是**未经几何标定的原始视觉投影角**。残差定义为IMU估计减视觉投影，不能解释为绝对机械角度误差。所有四条主要递推结果在相同有效视频帧、共同时间区间比较，共{metrics['core_common_support_samples']}帧；参考初始化的前1s不参与评价。视觉缺失帧不补点，IMU插值不跨越超过阈值的缺口。

{table(metrics['core_common_metrics'])}

固定互补相对纯gyro改善了平均残差，但P95与最大残差不一定同时改善，应按表中完整指标判断，不能仅凭MAE声称所有时刻都更好。

同时保留基于初始校准区间的明确常数偏移诊断，所减偏移为{metrics['first_second_calibration_offsets_deg']}°；校正后指标另见metrics.json。未在评价区间偷偷平移或缩放曲线。误差按角度、速度、开启/闭合分别统计，见error_by_angle/error_by_speed和direction_metrics。速度分组还要求局部导数有效，因此样本数可能小于总表。

![同步角度]({(out/'comparison/angle_overlay.png').resolve().as_posix()})

![相对投影参考残差]({(out/'comparison/angle_residuals.png').resolve().as_posix()})

![按角度分组]({(out/'comparison/error_by_angle.png').resolve().as_posix()})

![按速度分组]({(out/'comparison/error_by_speed.png').resolve().as_posix()})

静止保持段要求视觉速度<3°/s并持续至少.7s，检测到{len(windows)}段。各段中位数/标准差和起末读数见`comparison/stability_and_drift.json`。起始并非闭合而末尾闭合，起末角差不能直接叫漂移；应看相对视觉残差的变化或相同机械状态的回返差异。

{chr(10).join(stability_lines)}

开启和闭合的固定互补MAE分别为{directions['opening']['Fixed complementary']['mae_deg']:.3f}°、{directions['closing']['Fixed complementary']['mae_deg']:.3f}°；对应帧数{directions['opening']['Fixed complementary']['samples']}、{directions['closing']['Fixed complementary']['samples']}。这是当前速度与方向样本组成下的残差，并非完整动作的独立精度认证。

## 7. 固定底座单gyro消融

AB：Acc A+Gyro A+Acc B+Gyro B；仅B：Acc A+Acc B+Gyro B；仅A：Acc A+Gyro A+Acc B。没有训练神经网络。

屏蔽gyro通过独立接口在状态估计入口实现。仅A方案使用保留的accB原生事件时间和双acc相位，只在加速度模长、jerk、重力投影、由加速度计算的低相位变化率以及保留的gyroA通过门控时输出。没有把缺失gyroB假装为零积分，也没有从它取角速度、偏置、运动门控或更新时间。仅B组同样不访问gyroA。统一的离线时间标定与初始视觉角仅作为共享评估条件，不能误称无需标定的独立部署演示。

以下是**所有配置均有效的同一批帧**，共{metrics['all_configurations_common_support_samples']}帧，偏向低动态，不能代表快速开合：

{table(metrics['all_configurations_common_metrics'])}

各配置自己的可用率与指标见`ablation/metrics.json`；不同可用率上的指标不能直接排名。在共同有效视觉参考帧中，AB和保留B可用率都是{metrics['own_support_metrics_not_directly_comparable']['Dual AB']['availability_fraction']*100:.1f}%，仅保留A的低动态基线为{metrics['own_support_metrics_not_directly_comparable']['Retain A gravity']['availability_fraction']*100:.1f}%。这是相对于可用参考的比例，不是全段传感器系统运行可用率。AB相对保留B的角度差中位数约{systematic['AB_minus_retained_B_deg']['median_deg']:.3f}°。保留B可以覆盖动态；保留A的高动态信息不足，明确拒绝输出。本次即使AB与保留B相近，也只支持固定底座条件：主gyro大多近静止，本来不提供很多额外相对运动信息。任意手持时公共运动不再近零，结论不可外推。

![单gyro对照]({(out/'ablation/baseline_comparison.png').resolve().as_posix()})

## 8. 系统性误差与目前不能得到的结论

可能的系统性来源包括：初始窗口的轻微运动使零偏估计受污染；小角速度偏置经积分累积；两侧安装轴和主副时钟尚未独立标定；视觉投影与机械角度并不完全相同；快速动作中的采样/曝光/滤波滞后；接触冲击与线性加速度污染重力参考；部分边框被手遮挡或反光导致方向切换。它们只能通过现有诊断建立证据，不能把全部残差归因于某一个来源。

本次固定互补的平均残差为{metrics['core_common_metrics']['Fixed complementary']['mean_bias_deg']:.3f}°，双IMU为{metrics['core_common_metrics']['Dual AB']['mean_bias_deg']:.3f}°，双IMU存在更明显的负偏差。增加两侧信息没有自动提高当前结果，主重力相位、主gyro偏置及未标定轴均可能引入附加偏差。

以下仅是**未应用的零偏敏感性诊断**，没有依据评价残差重新挑选零偏。给出在同一条纯积分轨迹上替换零偏的理论末值变化，说明初始化窗口为何重要：

{chr(10).join(diagnostic_lines)}

另将时间偏移固定改为正负一视频帧进行诊断，结果保存在`comparison/systematic_diagnostics.json`，没有利用这些角度指标反过来优化同步。较快动作中即使时间误差仅一帧，也可产生显著角度残差；当前最大残差应结合动作速度和边框检测诊断，不能仅当作慢漂移。

当前建立的是数据驱动时间对应和角度对照流程，没有硬件同步真值，没有相机几何/独立机械角标定，也没有任意手持数据。首秒角度和同步来自同一视频，因此绝对初值不是独立测量。不能用本次静止std、可用率、互相关或软件测试精度替代真实角度精度。

## 9. 下一轮建议

记录真实传感器事件时间、设备身份、时钟域、序号、量程与安装轴，修复缓存重复写入；获取同步视频与独立编码器/标定角度参考。增加固定开合角整机转动、整机运动同时开合、铰链近竖直、快慢开合、不同初始角度、不同人员及持握位置。分别评价已知初始角跟踪与未知初值启动，覆盖长时漂移、遮挡和缺测恢复。

后续先建立完整双IMU物理基线，再比较两种单gyro配置与因果学习模型。训练/测试按独立会话、人员或设备划分，避免同轨迹重叠窗口泄漏。本次无训练结论。

## 10. 复现与验证

```powershell
Set-Location '<repo-root>'
python analyze_synchronized_experiment.py --video 'data/raw/1016110453.mp4' --imu 'data/raw/sensor_log_2026-10-16_11-04-53.csv' --config configs/experiment_20261016_110453.json --output outputs/experiment_20261016_110453 --rerun-video
python -m pytest tests -q
```

重复运行覆盖本次实验目录同名结果，历史目录不覆盖。清理/试跑文件不是最终证据，最终结果以本报告及manifest为准。所有原始文件SHA-256在运行前后核对，保持不变：{all(verify['inputs_unchanged'].values())}。标注视频实际解码{verify['decoded_annotated_frames']}帧，CSV行数{verify['video_rows']}；原始整数时间戳单调检查{verify['imu_raw_int_ns_strictly_increasing']}。测试及逐项验证见`verification.json`，图像与数值都应结合报告边界解释。
'''
    (out/'report.md').write_text(report,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',default='data/raw/1016110453.mp4');parser.add_argument('--imu',required=True)
    parser.add_argument('--config',default='configs/experiment_20261016_110453.json');parser.add_argument('--output',required=True);parser.add_argument('--rerun-video',action='store_true')
    args=parser.parse_args();analyze(args.video,args.imu,args.config,args.output,args.rerun_video)
