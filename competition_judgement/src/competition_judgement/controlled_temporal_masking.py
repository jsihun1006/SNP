"""Paired exact-count temporal deletions. No football labels or policy fitting."""
import numpy as np

from .temporal_features import masked_mean,contiguous_duration

FEATURES=('observed_approach','observed_future_change','observed_nearest_approach_m',
          'observed_nearest_future_change_m','approach_focus_separation_score','stable_close_score',
          'transient_excess_over_mean','close_duration_lower_bound_s','observed_occupancy_3m','observed_occupancy_5m')
FULL_ONLY=('approach_focus_separation_score','stable_close_score','transient_excess_over_mean')
SIGNED=FEATURES[:4]


def diagnostic_estimates(series,valid,radius_codes):
    """Available-sample plug-in, plus duration BOUNDS, not gap interpolation.

    The lower bound uses the unchanged v2-2 contiguous-duration routine. Upper
    assumes unknown positions could sustain contact until first KNOWN off.
    It is a set bound, not a reconstructed observation or a point estimate.
    """
    valid=np.asarray(valid,bool);n,t=valid.shape;c=t//2
    soft=series['soft'];nearest=series['nearest_m'];current=soft[:,c]
    past=masked_mean(soft[:,:c],valid[:,:c]);future=masked_mean(soft[:,c+1:],valid[:,c+1:])
    mean=masked_mean(soft,valid)
    low=np.min(np.where(valid,soft,np.inf),1);high=np.max(np.where(valid,soft,-np.inf),1)
    out=dict(observed_approach=current-past,observed_future_change=future-current,
             observed_nearest_approach_m=masked_mean(nearest[:,:c],valid[:,:c])-nearest[:,c],
             observed_nearest_future_change_m=masked_mean(nearest[:,c+1:],valid[:,c+1:])-nearest[:,c],
             approach_focus_separation_score=np.clip(current-past,0,1)*np.clip(current-future,0,1),
             transient_excess_over_mean=np.clip(current-mean,0,1))
    occupancy={r:masked_mean((series[f'edges_{r}m']>0).astype(float),valid) for r in (3,5)}
    out['observed_occupancy_3m']=occupancy[3];out['observed_occupancy_5m']=occupancy[5]
    active=np.where(radius_codes[:,None]==3,series['edges_3m'],series['edges_5m'])>0
    occ=np.where(radius_codes==3,occupancy[3],occupancy[5])
    out['stable_close_score']=occ*mean*(1-(high-low))
    duration=contiguous_duration(active,valid,np.ones_like(valid),3,25)
    out['close_duration_lower_bound_s']=duration['close_duration_lower_bound_s']
    potential=active|~valid
    lp=np.cumprod(potential[:,:c][:,::-1],axis=1).sum(1)
    lf=np.cumprod(potential[:,c+1:],axis=1).sum(1)
    upper=np.where(active[:,c],(lp+1+lf)/25,0.)
    # Center loss excludes ALL feature comparison, even if side averages exist.
    center=valid[:,c]
    for name in FEATURES:out[name]=np.where(center,out[name],np.nan)
    out['duration_upper_bound_s']=np.where(center,upper,np.nan)
    out['duration_interval_width_s']=out['duration_upper_bound_s']-out['close_duration_lower_bound_s']
    return out


def official_estimates(masked,radius_codes,series):
    """Target-feature adapter, tested against frozen summarize_anchor_series.

    Raw observed quantities match plug-in estimates. The three full-only
    quantities are unavailable after any deletion. No production code changes.
    """
    result=diagnostic_estimates(series,masked,radius_codes)
    for name in FULL_ONLY:result[name]=np.where(masked.all(1),result[name],np.nan)
    return result


def centered_subset(domain,count):
    if count>len(domain):return None
    start=(len(domain)-count)//2
    return np.asarray(domain,dtype=int)[start:start+count]


def position_mask(n,count,location,center_deleted,seed,empirical=None):
    """A-E fixed bands, F uniform, G random, H Train exact-count donors.

    Center-deleted mode spends one of the SAME total count at offset 0; all
    scores for that failure mode are excluded. No count clipping is allowed.
    """
    if not 1<=count<=30:raise ValueError('Unsupported fixed deletion count')
    remaining=count-int(center_deleted)
    mask=np.zeros((n,101),bool)
    if center_deleted:mask[:,50]=True
    domains=dict(far_past=np.arange(0,25),near_past=np.arange(25,50),
                 center_band=np.r_[np.arange(35,50),np.arange(51,66)],
                 near_future=np.arange(51,76),far_future=np.arange(76,101),
                 center_past_one_run=np.arange(25,50),center_future_one_run=np.arange(51,76))
    if location in domains:
        domain=domains[location]
        if remaining>len(domain):return None
        if location=='center_past_one_run':chosen=domain[-remaining:] if remaining else np.array([],int)
        elif location=='center_future_one_run':chosen=domain[:remaining]
        else:chosen=centered_subset(domain,remaining)
        mask[:,chosen]=True
    elif location=='uniform':
        domain=np.r_[np.arange(50),np.arange(51,101)]
        chosen=domain[np.floor((np.arange(remaining)+.5)*100/max(1,remaining)).astype(int)]
        mask[:,chosen]=True
    elif location=='random':
        rng=np.random.default_rng(seed);domain=np.r_[np.arange(50),np.arange(51,101)]
        ranking=np.argsort(rng.random((n,100)),axis=1)[:,:remaining]
        mask[np.arange(n)[:,None],domain[ranking]]=True
    elif location=='empirical':
        if empirical is None:return None
        if empirical.shape!=(n,101) or (empirical.sum(1)!=remaining).any() or empirical[:,50].any():
            raise ValueError('Empirical count/center mismatch')
        mask|=empirical
    else:raise ValueError('Unknown location')
    assert (mask.sum(1)==count).all() and (mask[:,50]==center_deleted).all()
    return mask


def continuity_mask(n,run_lengths,side):
    """Same count=10 and SAME mean offset +/-25.5; run continuity differs.

    [10], [5,5], [2]*5, [1]*10, all within one 50-frame side. Distances and
    spread cannot also be identical; only count, side and mean offset are held.
    """
    if sum(run_lengths)!=10:raise ValueError('Continuity study uses exactly ten')
    k=len(run_lengths)
    if k==1:starts=[20]
    elif k==2:starts=[13,32]
    elif k==5:starts=[8,16,24,32,40]
    elif k==10:starts=[7,11,15,19,23,26,30,34,38,42]
    else:raise ValueError('Unsupported continuity plan')
    domain=np.zeros(50,bool)
    for start,length in zip(starts,run_lengths):domain[start:start+length]=True
    assert domain.sum()==10 and np.flatnonzero(domain).mean()==24.5
    mask=np.zeros((n,101),bool)
    if side=='past':mask[:,:50]=domain
    elif side=='future':mask[:,51:]=domain
    else:raise ValueError('Unknown side')
    return mask


def feature_tolerances(name,config):
    if name.endswith('_m'):return config['deadband_distance_m'],config['small_error_distance_m']
    if name=='close_duration_lower_bound_s':return None,config['small_error_duration_s']
    return config['deadband_dimensionless'],config['small_error_dimensionless']


def error_summary(reference,estimate,name,config):
    valid=np.isfinite(reference)&np.isfinite(estimate)
    delta=estimate[valid]-reference[valid];absolute=np.abs(delta)
    deadband,small=feature_tolerances(name,config)
    result=dict(n=len(reference),finite_n=int(valid.sum()),nan_fraction=float((~np.isfinite(estimate)).mean()) if len(estimate) else None,
                mae=float(absolute.mean()) if len(delta) else None,median_ae=float(np.median(absolute)) if len(delta) else None,
                p95_ae=float(np.quantile(absolute,.95)) if len(delta) else None,bias=float(delta.mean()) if len(delta) else None,
                rmse=float(np.sqrt(np.mean(delta*delta))) if len(delta) else None,
                small_error_fraction=float((absolute<=small).mean()) if len(delta) else None,
                large_error_fraction=float((absolute>=small*config['large_error_multiplier']).mean()) if len(delta) else None)
    if name in SIGNED:
        a,b=reference[valid],estimate[valid];strong=np.abs(a)>deadband
        flip=((a>deadband)&(b < -deadband))|((a < -deadband)&(b>deadband))
        result.update(sign_strong_n=int(strong.sum()),sign_flip_rate=float(flip.sum()/strong.sum()) if strong.any() else None,
                      direction_loss_rate=float((strong&(np.abs(b)<=deadband)).sum()/strong.sum()) if strong.any() else None)
    return result
