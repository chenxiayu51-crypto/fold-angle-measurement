"""Offline audit and angle reconstruction for the supplied four-stream log."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from imu_angle import read_log, estimate_angle
from inspect_imu_log import inspect


def interpolate(t, values, target):
    return np.column_stack([np.interp(target,t,values[:,j],left=np.nan,right=np.nan) for j in range(3)])


def infer_clock_map(source,host_base,sensor_base):
    # CSV row order only supplies a software logging approximation, not clock calibration.
    with Path(source).open(encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.DictReader(handle))
    host=[(i,int(r['timestamp_ns'])) for i,r in enumerate(rows) if r['sensor'] in ('acc1','gyro1')]
    index=np.array([a for a,b in host]);timestamps=np.array([b for a,b in host],dtype=np.int64)
    xx,yy,width=[],[],[]
    for i,r in enumerate(rows):
        if r['sensor'] in ('acc1','gyro1'):continue
        j=np.searchsorted(index,i)
        before=int(timestamps[max(0,j-1)]);after=int(timestamps[min(j,len(timestamps)-1)])
        xx.append((int(r['timestamp_ns'])-sensor_base)/1e9)
        yy.append(((before-host_base)+(after-host_base))/2e9)
        width.append((after-before)/1e9)
    xx=np.array(xx);yy=np.array(yy);width=np.array(width);selected=width<.025
    fit=least_squares(lambda z:z[0]*xx[selected]+z[1]-yy[selected],[1.,0.],loss='soft_l1',f_scale=.005)
    residual=yy-(fit.x[0]*xx+fit.x[1])
    return {'scale':float(fit.x[0]),'offset_s':float(fit.x[1]),
            'method':'robust fit to neighboring host-clock CSV rows; uncalibrated software approximation',
            'residual_p05_p50_p95_ms':np.percentile(residual*1000,[5,50,95]).tolist(),
            'host_base_ns':host_base,'sensor_base_ns':sensor_base}


def run(source,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    audit=inspect(source,output);streams=read_log(source)
    for name in ('acc1','gyro1','加速度传感器','陀螺仪传感器'):
        if name not in streams:raise ValueError(f'Missing required stream {name!r}')
    host_base=min(int(streams[n].timestamps_ns[0]) for n in ('acc1','gyro1'))
    sensor_base=min(int(streams[n].timestamps_ns[0]) for n in ('加速度传感器','陀螺仪传感器'))
    clock=infer_clock_map(source,host_base,sensor_base)
    times={}
    for name,stream in streams.items():
        if name in ('acc1','gyro1'):times[name]=(stream.timestamps_ns-host_base)/1e9
        else:times[name]=clock['scale']*((stream.timestamps_ns-sensor_base)/1e9)+clock['offset_s']
    # Output at actual unique gyro1 timestamps, within all-stream common support.
    start=max(t[0] for t in times.values());end=min(t[-1] for t in times.values())
    keep=(times['gyro1']>=start)&(times['gyro1']<=end)
    t=times['gyro1'][keep];gyro=streams['gyro1'].values[keep]
    acc=interpolate(times['acc1'],streams['acc1'].values,t)
    base_acc=interpolate(times['加速度传感器'],streams['加速度传感器'].values,t)
    base_gyro=interpolate(times['陀螺仪传感器'],streams['陀螺仪传感器'].values,t)
    fixed=estimate_angle(t,acc,gyro,calibration_s=5.,initial_angle_deg=180.)
    dual=estimate_angle(t,acc,gyro,base_acc=base_acc,base_gyro=base_gyro,
                        calibration_s=5.,initial_angle_deg=180.)
    # Offline data diagnostic, not a calibrated mechanical axis.
    bias=np.array(fixed['gyro_bias_xyz_rad_s']);v=gyro-bias
    active=np.linalg.norm(v,axis=1)>.15
    _,singular,vt=np.linalg.svd(v[active],full_matrices=False)
    axis=vt[0];axis*=np.sign(axis[1])
    columns=['timestamp_ns','time_s','acc_angle_deg','gyro_integrated_angle_deg',
             'fixed_base_imu_angle_deg','provisional_dual_imu_angle_deg','angular_rate_deg_s',
             'acc_update_used','acc_quality','low_motion','valid','acc_norm_m_s2','base_acc_norm_m_s2']
    with (output/'angles.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.writer(handle);writer.writerow(columns)
        for k in range(len(t)):
            writer.writerow([int(streams['gyro1'].timestamps_ns[keep][k]),f'{t[k]:.9f}',
                fixed['acc_angle_deg'][k],fixed['gyro_angle_deg'][k],fixed['angle_deg'][k],dual['angle_deg'][k],
                fixed['rate_deg_s'][k],int(fixed['acc_used'][k]),int(fixed['acc_quality'][k]),
                int(fixed['low_motion'][k]),int(fixed['valid'][k]),np.linalg.norm(acc[k]),np.linalg.norm(base_acc[k])])
    windows=[('initial_flat',0,10),('first_closed',18.5,24.5),('middle_flat',33,34.5),
             ('second_closed',39.5,43.5),('final_flat',51,52.5)]
    statistics=[]
    for label,left,right in windows:
        mask=(t>=left)&(t<right)
        item={'label':label,'start_s':left,'end_s':right,'samples':int(np.sum(mask))}
        for key,array in [('acc',fixed['acc_angle_deg']),('gyro',fixed['gyro_angle_deg']),
                          ('fixed_base_fused',fixed['angle_deg']),('provisional_dual',dual['angle_deg'])]:
            item[key]={'median_deg':float(np.nanmedian(array[mask])),
                       'std_deg':float(np.nanstd(array[mask]))}
        statistics.append(item)
    summary={'source':str(Path(source).resolve()),'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        'duration_s':audit['streams']['acc1']['duration_s'],'output_rows':len(t),
        'output_sampling':'unique gyro1 events, common four-stream support; accelerations interpolated offline',
        'assumptions':{'gyro_unit':'rad/s, confirmed by user',
                       'acc_unit':'m/s^2, inferred from stationary norm',
                       'moving_imu':'secondary IMU, user-confirmed role; acc1/gyro1 mapping inferred from motion',
                       'fixed_imu':'main IMU, user-confirmed role; Chinese-named mapping inferred; small base motion exists',
                       'hinge_axes':'both +Y provisionally; installation signs not independently calibrated',
                       'initial_angle_deg':180,'initial_angle_source':'user-described initially fully unfolded',
                       'calibration_s':5,'correction_tau_s':1,
                       'acc_gate':'norm within 0.6 m/s^2 of initial gravity; projected gravity >0.2g; jerk<30 m/s^3; gyro norm<0.25 rad/s; innovation<20 degrees',
                       'endpoint_clipping':False,'endpoint_scale_fit':False,'zero_phase_filter':False},
        'clock_map':clock,'gyro_principal_axis':axis.tolist(),
        'gyro_principal_component_energy_fraction':float(singular[0]**2/np.sum(singular**2)),
        'initial_bias_xyz_rad_s':fixed['gyro_bias_xyz_rad_s'],
        'fixed_base_fused_min_max_deg':[float(np.nanmin(fixed['angle_deg'])),float(np.nanmax(fixed['angle_deg']))],
        'gyro_only_min_max_deg':[float(np.nanmin(fixed['gyro_angle_deg'])),float(np.nanmax(fixed['gyro_angle_deg']))],
        'valid_rows':int(np.sum(fixed['valid'])),'acc_corrections':int(np.sum(fixed['acc_used'])),
        'provisional_dual_minus_fixed_max_abs_deg':float(np.nanmax(np.abs(dual['angle_deg']-fixed['angle_deg']))),
        'windows':statistics,
        'limitations':['No independent intermediate-angle ground truth; no physical accuracy validated.',
                       'Initialization uses user-provided 180 degrees; it is not an independent measurement.',
                       'Current +Y axis convention is data-supported but not a full IMU installation calibration.',
                       'Clock mapping is approximate; timestamp semantics require acquisition metadata.',
                       '67 acc1 and 23 gyro1 timestamps have differing values; keep-last is a documented lossy resolution.',
                       'This is offline interpolation; a live implementation needs causal asynchronous sampling.',
                       'This sensor capture is not synchronized to the earlier video.']}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.22})
    fig,axes=plt.subplots(3,1,figsize=(14,10),sharex=True,gridspec_kw={'height_ratios':[2,1,1]})
    a=axes[0]
    a.plot(t,fixed['acc_angle_deg'],color='#aaaaaa',lw=.65,alpha=.7,label='Accelerometer angle (dynamic disturbances visible)')
    a.plot(t,fixed['gyro_angle_deg'],color='#d58b22',lw=1.2,label='Gyro integration, initial 180 deg')
    a.plot(t,fixed['angle_deg'],color='#1666b1',lw=1.7,label='Fixed-base IMU fusion (primary)')
    a.plot(t,dual['angle_deg'],color='#7d3bac',lw=.8,ls='--',label='Provisional dual IMU (uncalibrated clock / axes)')
    a.axhline(0,color='k',lw=.6);a.axhline(180,color='k',lw=.6)
    for _,left,right in windows:a.axvspan(left,right,color='#48a879',alpha=.07)
    a.set_ylabel('Opening angle (deg)');a.set_ylim(-22,215);a.legend(loc='lower left',fontsize=8)
    a.set_title('Pure IMU reconstruction | 0 deg closed, 180 deg flat | initial angle supplied by user')
    axes[1].plot(t,fixed['rate_deg_s'],color='#d58b22',lw=1,label='Opening angular rate')
    axes[1].set_ylabel('Rate (deg/s)');axes[1].legend(loc='upper left',fontsize=8)
    axes[2].plot(t,np.linalg.norm(acc,axis=1),color='#1666b1',lw=.8,label='Moving IMU |acc|')
    axes[2].plot(t,np.linalg.norm(base_acc,axis=1),color='#7d3bac',lw=.7,label='Base IMU |acc|')
    axes[2].scatter(t[fixed['acc_used']],np.full(np.sum(fixed['acc_used']),7),s=1,color='#249054',label='Gravity correction accepted')
    axes[2].set_ylabel('Acceleration (m/s²)');axes[2].set_xlabel('Time from first acc1 record (s)')
    axes[2].legend(loc='upper left',fontsize=8);axes[2].set_xlim(t[0],t[-1])
    fig.text(.5,.01,'Gyro rad/s confirmed; +Y hinge axes assumed. Software clock alignment. No endpoint snapping or mechanical accuracy claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.025,1,1]);fig.savefig(output/'angle_curve.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(14,10),sharex=True)
    for ax,(name,stream) in zip(axes,streams.items()):
        for j,label in enumerate('XYZ'):ax.plot(times[name],stream.values[:,j],lw=.65,label=label)
        label=name if name in ('acc1','gyro1') else ('base_acc' if '加速度' in name else 'base_gyro')
        ax.set_ylabel(label+'\n'+('m/s²' if 'acc' in label else 'rad/s'));ax.legend(loc='upper right',ncol=3,fontsize=8)
    axes[0].set_title('Deduplicated raw streams | distinct clocks approximately aligned using log row order')
    axes[-1].set_xlabel('Approximate common time (s)');fig.tight_layout();fig.savefig(output/'raw_signals.png',dpi=140);plt.close(fig)
    lines=['# 纯 IMU 开合角分析','',
        '输入：`'+str(Path(source).resolve())+'`。本次约 52.71 秒。用户说明初始展开 180°，之后闭合、保持并往返。','',
        '## 已完成','',
        '- 四路信号去重、时间检查与三轴诊断；原始 CSV 保持不变。',
        '- 以 acc1/gyro1 为运动半机身，Y 为暂定铰链轴；按固定底座重建角度，并输出暂定双 IMU 对比。',
        '- 同时导出加速度角、纯陀螺积分与低动态重力校正结果；不强制裁剪到 0–180°，不按端点缩放。','',
        '## 静止窗口读数','',
        '下表为选择的保持窗口统计，并非独立真值误差。角度初值 180° 来自用户动作说明。','',
        '| 窗口 | 时间 s | 固定底座融合中位数 ° | 标准差 ° | 纯陀螺中位数 ° |',
        '|---|---:|---:|---:|---:|']
    for s in statistics:
        lines.append(f"| {s['label']} | {s['start_s']}–{s['end_s']} | {s['fixed_base_fused']['median_deg']:.3f} | {s['fixed_base_fused']['std_deg']:.3f} | {s['gyro']['median_deg']:.3f} |")
    lines+=['','## 采集问题与假设','',
        '- 52,947 行中，acc1 有 26,471 条重复时间戳记录，gyro1 有 6,180 条；独立时刻分别为 5,046 和 4,356。重复不等于独立测量。',
        '- acc1 的 67 个、gyro1 的 23 个时间戳对应多个不同数值；暂取最后记录，并标记为不可恢复的时间歧义。',
        '- 一组是约 1.79e18 ns 的时间戳，另一组是约 4.75e13 ns。日志行顺序仅能近似关联时钟，不能证明硬件同步。',
        '- gyro=rad/s 已获用户确认；acc 单位按静止模长推断为 m/s²。用户确认主 IMU 固定、副 IMU 转动，acc1/gyro1 对应副 IMU 的标签映射由信号推断。陀螺主要分量为 Y；安装方向仍待标定。',
        '- 主结果使用同一时钟域的 acc1 与 gyro1，适用于用户指定的固定底座。双 IMU 曲线采用两边同向 +Y 以及近似时钟映射，仅作对比。',
        '- 输出以实际 gyro1 独立时间戳为准；acc 离线插值。曲线连续不代表提高了硬件采样率。',
        '- 前 5 秒用于初始重力参考及陀螺偏置。该离线方案初始化后顺序更新，不等于已经实现实时系统。',
        '- 无同步新视频与独立中间角度真值，不能据端点读数宣称全程精度；未与上一段视频融合。','',
        '## 运行','',
        '```powershell',
        r"python analyze_imu_log.py --input 'data/raw/sensor_log_2026-10-16_09-12-25.csv' --output outputs/imu_20261016_091225",
        '```','',
        '结果：`angle_curve.png`、`raw_signals.png`、`angles.csv`、`summary.json` 和 `data_audit.json`。',
        '下一步采集端应直接保存每次真实传感器事件的硬件时间戳、时钟域、序号和单位，避免循环刷新时重复写缓存值。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'rows':len(t),'windows':statistics,'clock':clock,'range':summary['fixed_base_fused_min_max_deg']},ensure_ascii=True))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.input,args.output)
