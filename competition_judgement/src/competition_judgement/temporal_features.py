"""Bidirectional descriptors around fixed present geometry, never contest classes."""
import numpy as np
import pandas as pd


def masked_mean(values,valid):
    n=valid.sum(axis=1)
    return np.divide(np.where(valid,values,0).sum(axis=1),n,out=np.full(len(n),np.nan),where=n>0)


def contiguous_duration(series,valid,record_support,radius_index,fps):
    """Lower bounds; missing/changed observations never count as known separation."""
    batch,t=series.shape;c=t//2
    active=series>0
    current=valid[:,c]&active[:,c]
    counts=[];censored=[];reason=[]
    for order in (np.arange(c-1,-1,-1),np.arange(c+1,t)):
        continuation=valid[:,order]&active[:,order]
        prefix=np.cumprod(continuation,axis=1,dtype=int)
        count=prefix.sum(axis=1);counts.append(count)
        at_end=count==len(order)
        stop_index=order[np.minimum(count,len(order)-1)]
        row=np.arange(batch)
        unknown=~valid[row,stop_index]
        censored.append(at_end|unknown|~current)
        why=np.where(at_end,"window_censored",np.where(~record_support[row,stop_index],"boundary_censored",
                     np.where(unknown,"observation_or_region_ambiguous","observed_local_relation_end")))
        why[~current & valid[:,c]]="current_relation_not_active"
        why[~valid[:,c]]="current_observation_unknown"
        reason.append(why)
    duration=np.where(valid[:,c],np.where(current,(counts[0]+1+counts[1])/fps,0.),np.nan)
    onset=np.where(current & ~censored[0],-counts[0],np.nan)
    # Reapproach only when EVERY future observation is interpretable.
    future=active[:,c+1:];known=valid[:,c+1:].all(axis=1)&current
    inactive_seen=np.maximum.accumulate(~future,axis=1)
    reapproach=np.where(known,(inactive_seen&future).any(axis=1).astype(float),np.nan)
    return dict(close_duration_lower_bound_s=duration,
                past_close_lower_bound_s=np.where(valid[:,c],np.where(current,counts[0]/fps,0.),np.nan),
                future_close_lower_bound_s=np.where(valid[:,c],np.where(current,counts[1]/fps,0.),np.nan),
                onset_offset_frames=onset,onset_censored=censored[0],end_censored=censored[1],
                past_end_reason=reason[0],future_end_reason=reason[1],reapproach_observed=reapproach)


def summarize_anchor_series(s,radius,fps):
    t=s['soft'].shape[1];c=t//2;batch=len(s['soft'])
    past=np.arange(t)<c;future=np.arange(t)>c
    observed=s['observed'];reliable=s['reliable'];records=s['record_support']
    soft=s['soft'];current=np.where(observed[:,c],soft[:,c],np.nan)
    pm=masked_mean(soft,observed & past);fm=masked_mean(soft,observed & future)
    whole=masked_mean(soft,observed)
    std=np.sqrt(masked_mean((soft-whole[:,None])**2,observed))
    maximum=np.max(np.where(observed,soft,-np.inf),axis=1)
    minimum=np.min(np.where(observed,soft,np.inf),axis=1)
    maximum[~observed.any(1)]=np.nan;minimum[~observed.any(1)]=np.nan
    full=reliable.all(1)
    rp=masked_mean(soft,reliable & past);rf=masked_mean(soft,reliable & future)
    out=dict(record_past_frames=records[:,past].sum(1),record_future_frames=records[:,future].sum(1),
        observed_past_frames=observed[:,past].sum(1),observed_future_frames=observed[:,future].sum(1),
        reliable_past_frames=reliable[:,past].sum(1),reliable_future_frames=reliable[:,future].sum(1),
        full_record_window=records.all(1),pattern_score_valid=full,
        window_observed_fraction=observed.mean(1),window_strict_fraction=(records&s['strict']).sum(1)/t,
        global_count_change=~(s['global_count_same']|~records).all(1),
        local_count_change=((np.max(np.where(observed,s['anchor_n_home'],-1),axis=1)!=np.min(np.where(observed,s['anchor_n_home'],999),axis=1))|
                            (np.max(np.where(observed,s['anchor_n_away'],-1),axis=1)!=np.min(np.where(observed,s['anchor_n_away'],999),axis=1))),
        present_anchor_soft=current,present_anchor_soft_pair_mean=np.where(observed[:,c],s['soft_pair_mean'][:,c],np.nan),
        observed_past_soft_mean=pm,observed_future_soft_mean=fm,observed_window_soft_mean=whole,
        observed_window_soft_std=std,observed_window_soft_max=maximum,observed_window_soft_range=maximum-minimum,
        observed_approach=current-pm,observed_future_change=fm-current,
        observed_nearest_approach_m=masked_mean(s['nearest_m'],observed&past)-s['nearest_m'][:,c],
        observed_nearest_future_change_m=masked_mean(s['nearest_m'],observed&future)-s['nearest_m'][:,c],
        reliable_approach=np.where(full,current-rp,np.nan),reliable_future_change=np.where(full,rf-current,np.nan),
        approach_focus_separation_score=np.where(full,np.clip(current-rp,0,1)*np.clip(current-rf,0,1),np.nan),
        transient_excess_over_mean=np.where(full,np.clip(current-whole,0,1),np.nan))
    near_past=past & (np.arange(t)>=c-max(1,c//2))
    far_past=past & ~near_past
    out['short_approach']=current-masked_mean(soft,observed&near_past)
    out['long_approach']=current-masked_mean(soft,observed&far_past)
    for r in (3,5):
        edges=s[f'edges_{r}m'];size=s[f'contact_size_{r}m']
        occupancy=masked_mean((edges>0).astype(float),observed)
        out[f'observed_occupancy_{r}m']=occupancy
        out[f'observed_edge_growth_{r}m']=np.where(observed[:,c],edges[:,c],np.nan)-masked_mean(edges,observed&past)
        out[f'observed_edge_future_change_{r}m']=masked_mean(edges,observed&future)-np.where(observed[:,c],edges[:,c],np.nan)
        out[f'observed_contact_size_growth_{r}m']=np.where(observed[:,c],size[:,c],np.nan)-masked_mean(size,observed&past)
        out[f'observed_contact_size_future_change_{r}m']=masked_mean(size,observed&future)-np.where(observed[:,c],size[:,c],np.nan)
    out['stable_close_score']=np.where(full,out[f'observed_occupancy_{radius}m']*whole*(1-(maximum-minimum)),np.nan)
    out.update(contiguous_duration(s[f'edges_{radius}m'],reliable,records,radius,fps))
    return pd.DataFrame(out)


def analyze_anchor_batch(source,clusters):
    config=source.config
    rows=np.searchsorted(source.ids,clusters.frame.to_numpy())
    if (rows>=len(source.ids)).any() or not np.array_equal(source.ids[rows],clusters.frame):
        raise ValueError("Cluster frame not in source")
    centers=clusters[['cluster_centroid_x','cluster_centroid_y']].to_numpy()
    radii=np.maximum(config['anchor_min_radius_m'],clusters.cluster_max_pairwise_distance_m.to_numpy()+config['anchor_diameter_margin_m'])
    series=source.anchor_series(rows,centers,radii)
    nearby=source.nearby_components(series['index'],centers,radii,(clusters.radius_m.to_numpy()==5).astype(int))
    pieces=[]
    for radius in (3,5):
        select=clusters.radius_m.to_numpy()==radius
        if not select.any():continue
        subset={k:(v[select] if isinstance(v,np.ndarray) and v.ndim==2 else v) for k,v in series.items()}
        values=summarize_anchor_series(subset,radius,config['fps'])
        values.index=np.flatnonzero(select);pieces.append(values)
    result=pd.concat(pieces).sort_index().reset_index(drop=True)
    result['anchor_radius_m']=radii
    for name,mask in (('record',series['record_support']),('observed',series['observed']),('reliable',series['reliable'])):
        result[f'{name}_mask_hex']=[v.tobytes().hex() for v in np.packbits(mask,axis=1,bitorder='little')]
    if nearby is not None:
        c=config['window_radius_frames'];past=np.arange(2*c+1)<c;future=np.arange(2*c+1)>c
        for name,values in nearby.items():
            result[f'present_{name}']=np.where(series['observed'][:,c],values[:,c],np.nan)
            result[f'observed_{name}_past_growth']=result[f'present_{name}']-masked_mean(values,series['observed']&past)
            result[f'observed_{name}_future_change']=masked_mean(values,series['observed']&future)-result[f'present_{name}']
    result['local_count_change']=np.where(series['observed'].any(1),result.local_count_change.to_numpy(dtype=float),np.nan)
    # Static, non-spatial moving-average baseline, no missing extrema imputed.
    for r in (3,5):
        baseline=source.v2_strength[series['index'],0 if r==3 else 1]
        mask=series['record_support'] & np.isfinite(baseline)&series['strict']
        result[f'global_v2_moving_mean_{r}m']=masked_mean(baseline,mask)
        result[f'global_v2_moving_support_{r}m']=mask.sum(1)
    minimum=int(np.ceil(config['partial_side_fraction']*config['window_radius_frames']))
    result['partial_bidirectional_available']=(result.reliable_past_frames>=minimum)&(result.reliable_future_frames>=minimum)
    return result
