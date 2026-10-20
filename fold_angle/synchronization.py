"""Offline motion synchronization with explicit missing-data support.

All times are relative seconds. No angle estimate is used to align clocks.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar


def gap_interp(t, values, query, max_gap_s=.1):
    t=np.asarray(t,float); values=np.asarray(values,float); query=np.asarray(query,float)
    if np.any(np.diff(t)<=0):
        raise ValueError('Sample times must increase')
    if values.ndim==2:
        return np.column_stack([gap_interp(t,values[:,j],query,max_gap_s) for j in range(values.shape[1])])
    out=np.full(query.shape,np.nan)
    good=np.isfinite(t)&np.isfinite(values)
    tt=t[good]; vv=values[good]
    if len(tt)<2:return out
    j=np.searchsorted(tt,query,side='left'); k=np.clip(j,1,len(tt)-1)
    inside=(query>=tt[0])&(query<=tt[-1])
    supported=(tt[k]-tt[k-1])<=max_gap_s
    exact=(j<len(tt))&(tt[np.clip(j,0,len(tt)-1)]==query)
    m=inside&(supported|exact)
    out[m]=np.interp(query[m],tt,vv)
    return out


def visual_derivative(t, angle, valid, window_frames=11):
    """Centered quadratic fit; offline and deliberately not the One Euro output.

    Requires all observed neighbors; fills no invalid frames, and never crosses
    a long outage. Returned slope is in degrees/second.
    """
    t=np.asarray(t); angle=np.asarray(angle); valid=np.asarray(valid)&np.isfinite(angle)
    velocity=np.full_like(angle,np.nan,dtype=float)
    smoothed=velocity.copy(); half=window_frames//2
    for k in range(half,len(t)-half):
        sl=slice(k-half,k+half+1)
        if not valid[sl].all():continue
        dt=t[sl]-t[k]
        if np.max(np.diff(t[sl]))>2.2*np.median(np.diff(t)):continue
        coef=np.polynomial.polynomial.polyfit(dt,angle[sl],2)
        smoothed[k]=coef[0]; velocity[k]=coef[1]
    return smoothed,velocity


def motion_events(t, rate, threshold_deg_s=18., merge_gap_s=.65):
    active=np.flatnonzero(np.isfinite(rate)&(np.abs(rate)>threshold_deg_s))
    if len(active)==0:return []
    cuts=np.flatnonzero(np.diff(t[active])>merge_gap_s)+1
    events=[]
    for ids in np.split(active,cuts):
        if len(ids)<4 or t[ids[-1]]-t[ids[0]]<.1:continue
        peak=ids[np.argmax(np.abs(rate[ids]))]
        events.append({'start_s':float(t[ids[0]]),'end_s':float(t[ids[-1]]),
                       'peak_video_s':float(t[peak]),'peak_video_deg_s':float(rate[peak]),
                       'direction':'opening' if rate[peak]>0 else 'closing'})
    return events


def _corr(x,y):
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<12 or min(np.std(x[m]),np.std(y[m]))<1e-5:return None
    return float(np.corrcoef(x[m],y[m])[0,1])


def align_motion(t_video, rate_video_deg_s, t_imu, rate_imu_rad_s, *, offset_bounds=None):
    """Fit offset on training motion events and reserve other events for checks.

    Input gyro sign must already mean opening-positive. Scale a=1 remains
    fixed: local signal distortion is not interpreted as clock drift.
    """
    tv=np.asarray(t_video,float); vv=np.asarray(rate_video_deg_s,float)
    ti=np.asarray(t_imu,float); gg=np.rad2deg(np.asarray(rate_imu_rad_s,float))
    if np.nanstd(vv)<1 or np.nanstd(gg)<1:
        raise ValueError('Insufficient motion to determine synchronization')
    dt=.02; grid=np.arange(ti[0],ti[-1]+dt/2,dt)
    gyro=gap_interp(ti,gg,grid,.1)
    # Offline symmetric Gaussian suppresses contact impulses for motion matching.
    filled=np.interp(grid,grid[np.isfinite(gyro)],gyro[np.isfinite(gyro)])
    gyro_s=gaussian_filter1d(filled,2.)
    gyro_s[~np.isfinite(gyro)]=np.nan
    clip=250.
    v=np.clip(vv,-clip,clip); g=np.clip(gyro_s,-clip,clip)
    events=motion_events(tv,v)
    initial_fit_mask=np.zeros(len(tv),bool)
    for j,event in enumerate(events):
        event['event_id']=j
        event['validation']=j%3==2
        if not event['validation']:
            initial_fit_mask|=(tv>=event['start_s']-.6)&(tv<=event['end_s']+.6)
    withheld_mask=np.zeros(len(tv),bool)
    for event in events:
        if event['validation']:
            withheld_mask|=(tv>=event['start_s']-.6)&(tv<=event['end_s']+.6)
    initial_fit_mask &= ~withheld_mask
    if initial_fit_mask.sum()<25:initial_fit_mask[:]=True
    def at(b,mask=None):
        y=gap_interp(grid,g,tv+b,.1)
        if mask is not None:y=np.where(mask,y,np.nan)
        return _corr(v,y),int((np.isfinite(v)&np.isfinite(y)).sum())
    if offset_bounds is None:offset_bounds=(-tv[-1]+.4*min(tv[-1],ti[-1]),ti[-1]-.4*min(tv[-1],ti[-1]))
    offsets=np.arange(offset_bounds[0],offset_bounds[1]+.01,.02)
    min_count=max(25,int((np.isfinite(v)&initial_fit_mask).sum()*.45))
    scores=[]
    for b in offsets:
        c,n=at(b,initial_fit_mask); scores.append(c if c is not None and n>=min_count else np.nan)
    scores=np.asarray(scores)
    if not np.isfinite(scores).any():raise ValueError('No supported motion overlap')
    best=int(np.nanargmax(scores)); coarse=float(offsets[best])
    peaks,_=find_peaks(np.nan_to_num(scores,nan=-2),distance=25)
    peaks=np.unique(np.r_[peaks,best]); peaks=peaks[np.argsort(scores[peaks])[::-1]]
    alternatives=[{'b_s':float(offsets[p]),'correlation':float(scores[p])} for p in peaks[:8]]
    distinct=[c for c in alternatives if abs(c['b_s']-coarse)>.75]
    ambiguous=bool(distinct and distinct[0]['correlation']>float(scores[best])-.08)
    supported=[]
    for event in events:
        center=event['peak_video_s']
        if not ti[0]+.2<center+coarse<ti[-1]-.2:continue
        start=max(tv[0],event['start_s']-.6); end=min(tv[-1],event['end_s']+.6)
        mask=(tv>=start)&(tv<=end)&np.isfinite(v)
        options=[]
        for b in np.arange(coarse-.55,coarse+.551,.01):
            c,n=at(b,mask)
            if c is not None and n>=15:options.append((c,b))
        if not options:continue
        c,local=max(options)
        supported.append(dict(event,local_offset_s=float(local),local_correlation=float(c),window_s=[float(start),float(end)]))
    # A subset of events is withheld before refinement, never used to choose b.
    train=[e for e in supported if not e['validation']]
    validation=[e for e in supported if e['validation']]
    train_mask=np.zeros(len(tv),bool)
    for e in train:
        if e['local_correlation']>.5:
            train_mask|=(tv>=e['window_s'][0])&(tv<=e['window_s'][1])
    train_mask &= ~withheld_mask
    if train_mask.sum()>=30:
        fit=minimize_scalar(lambda b: -(at(b,train_mask)[0] or -1),bounds=(coarse-.3,coarse+.3),method='bounded',options={'xatol':.0001})
        b=float(fit.x)
    else:b=coarse
    used_peaks=set();last_peak=-np.inf
    pos,_=find_peaks(np.nan_to_num(g),prominence=12,distance=12)
    neg,_=find_peaks(-np.nan_to_num(g),prominence=12,distance=12)
    for e in supported:
        m=(tv>=e['window_s'][0])&(tv<=e['window_s'][1]); e['global_offset_correlation']=at(b,m)[0]
        e['local_minus_global_ms']=(e['local_offset_s']-b)*1000
        e['validation']=e in validation
        # Peak location is diagnostic only: impulsive gyro and visual gaps can
        # make the peaks represent different instants of the same motion.
        target=e['peak_video_s']+b
        ids=pos if e['direction']=='opening' else neg
        ids=[p for p in ids if p not in used_peaks and grid[p]>last_peak and
             e['start_s']+b-.15<=grid[p]<=e['end_s']+b+.15 and abs(grid[p]-target)<.65]
        if ids:
            p=min(ids,key=lambda p:abs(grid[p]-target));used_peaks.add(p);last_peak=grid[p]
            e['peak_imu_s']=float(grid[p]);e['peak_residual_ms']=float((grid[p]-target)*1000)
        else:
            e['peak_imu_s']=None;e['peak_residual_ms']=None
    y0=gap_interp(grid,g,tv,.1); y1=gap_interp(grid,g,tv+b,.1)
    common=np.isfinite(v)&np.isfinite(y0)&np.isfinite(y1)
    usable=(np.isfinite(v)&np.isfinite(y1))
    return {'model':'t_imu = a*t_video + b','a':1.,'b_s':b,'scale_estimated':False,
            'scale_reason':'Offset-only fit; local motion distortions and short duration do not establish oscillator drift.',
            'coarse_offset_s':coarse,'correlation_before':_corr(v[common],y0[common]),
            'correlation_after':_corr(v[common],y1[common]),'correlation_common_sample_count':int(common.sum()),
            'coarse_candidates':alternatives,'offset_profile':{'b_s':offsets.tolist(),'correlation':[None if not np.isfinite(x) else float(x) for x in scores]},
            'ambiguous_offset_candidates':ambiguous,
            'training_events':[e for e in supported if not e['validation']],
            'validation_events':[e for e in supported if e['validation']],
            'video_overlap_s':[float(tv[usable][0]),float(tv[usable][-1])],
            'ambiguity_note':'Event ordering and withheld checks required; correlation is not synchronization uncertainty.',
            'gyro_matching_filter':'symmetric Gaussian sigma=.04 s; clipped at 250 deg/s, offline',
            'fit_uses':'gyro angular velocity versus derivative of raw visual angle; no IMU angles'}
