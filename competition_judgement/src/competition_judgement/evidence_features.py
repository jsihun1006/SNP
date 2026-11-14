"""Stage B descriptors only: no contest classes, probabilities or possession."""
import numpy as np
import pandas as pd
from .temporal_features import masked_mean,contiguous_duration
from .partial_window_diagnostics import run_statistics

NAMES={'P1':'P1_rapid_convergence','P2':'P2_local_concentration','P3':'P3_relationship_reconfiguration',
       'P4':'P4_subsequent_separation','N1':'N1_persistent_marking_proxy','N2':'N2_co_movement_proxy','N3':'N3_stable_proximity'}
BANDS=(slice(0,25),slice(25,50),slice(51,76),slice(76,101))


def scope_slice(name):
    return {'past':slice(0,51),'near_past':slice(25,51),'middle':slice(25,76),
            'future':slice(50,101),'whole':slice(0,101),'path':slice(40,61)}[name]


def masked_std(x,valid):
    m=masked_mean(x,valid)
    return np.sqrt(masked_mean((x-m[:,None])**2,valid))


def changed_from_center(values,known,sl):
    return ((values[:,sl]!=values[:,50,None])&known[:,sl]).any(1)


def support_state(s,p,full_mode):
    v=s['reliable'];n=len(v);sl=scope_slice(p['scope']);c=s['observed'][:,50]&s['strict'][:,50]&s['record_support'][:,50]
    local=changed_from_center(s['anchor_n_home'],s['observed'],sl)|changed_from_center(s['anchor_n_away'],s['observed'],sl)
    global_known=s.get('global_count_known',s['record_support'])
    global_change=(~s['global_count_same'][:,sl]&global_known[:,sl]).any(1)
    _,gap,_=run_statistics(~v[:,sl])
    reason=np.full(n,'ok',object)
    def fail(mask,label):
        nonlocal reason
        reason=np.where((reason=='ok')&mask,label,reason)
    fail(~s['record_support'][:,50],'outside_or_no_record')
    fail(~s['observed'][:,50],'current_anchor_unobserved')
    fail(~s['strict'][:,50],'current_strict_qc_failed')
    if full_mode:fail(~v.all(1),'not_full_reliable')
    else:
        for band,minimum in zip(BANDS,p['bands']):fail(v[:,band].sum(1)<minimum,'insufficient_band_support')
        r=p['center_radius'];fail(~v[:,50-r:51+r].all(1),'center_gap')
        fail(gap>p['max_gap'],'long_missing_run')
    if p['global_constant']:fail(global_change,'global_count_composition_ambiguous')
    if p['local_constant']:fail(local,'local_composition_ambiguous')
    return dict(valid=(reason=='ok')&c,reason=reason,local_changed=local,global_changed=global_change,
                observed_n=s['observed'][:,sl].sum(1),valid_n=v[:,sl].sum(1),max_gap=gap)


def compute_evidence(s,radii,config):
    """Input: named [N,101] observed series + bounded geometry path metadata.

    No hidden coordinate values are consumed where masks are false. Returns
    separate full_/partial_ columns; missing Evidence always remains NaN.
    """
    v=s['reliable'];n,t=v.shape
    if t!=101 or config['fps']!=25:raise ValueError('Expected 101 frames at 25fps')
    scale=config['scales'];rad=np.asarray(radii)
    edge=np.where(rad[:,None]==3,s['edges_3m'],s['edges_5m'])
    contact=np.where(rad[:,None]==3,s['contact_size_3m'],s['contact_size_5m'])
    denom=s['anchor_n_home']*s['anchor_n_away']
    density=np.divide(edge,denom,out=np.full(edge.shape,np.nan),where=denom>0)
    S=s['soft'];d=s['nearest_m'];D=s['component_diameter'];R=s['component_radius'];K=s['nearby_count']
    pm=lambda x:masked_mean(x[:,:50],v[:,:50])
    fm=lambda x:masked_mean(x[:,51:],v[:,51:])
    nearp=lambda x:masked_mean(x[:,25:50],v[:,25:50])
    nearf=lambda x:masked_mean(x[:,51:76],v[:,51:76])
    whole=lambda x:masked_mean(x,v)
    pos=lambda x,key:np.clip(x/scale[key],0,1)
    approach=S[:,50]-pm(S);future=S[:,50]-fm(S)
    distance_approach=pm(d)-d[:,50];distance_future=fm(d)-d[:,50]
    long=S[:,50]-masked_mean(S[:,:25],v[:,:25]);short=S[:,50]-nearp(S)
    p1=np.mean([pos(distance_approach,'distance_m'),pos(approach,'soft_change'),pos(long-short,'soft_change')],axis=0)
    geom=v&np.isfinite(D)&np.isfinite(R)
    geom_p=geom[:,25:50].sum(1);geom_f=geom[:,51:].sum(1)
    mpD=masked_mean(D[:,25:50],geom[:,25:50]);mpR=masked_mean(R[:,25:50],geom[:,25:50])
    p2terms=np.stack([pos(mpR-R[:,50],'radius_m'),pos(mpD-D[:,50],'diameter_m'),
        pos(contact[:,50]-nearp(contact),'participants'),pos(np.log1p(edge[:,50])-nearp(np.log1p(edge)),'log_edges'),
        pos(density[:,50]-nearp(density),'density')],1)
    p2=p2terms.mean(1)
    branches=np.maximum(s['prev_candidate_count']-1,0)+np.maximum(s['next_candidate_count']-1,0)
    kchange=(np.abs(K[:,50]-nearp(K))+np.abs(nearf(K)-K[:,50]))/2
    densitychange=(np.abs(density[:,50]-nearp(density))+np.abs(nearf(density)-density[:,50]))/2
    p3=np.mean([pos(kchange,'cluster_count'),pos(densitychange,'density'),np.clip(branches/2,0,1)],axis=0)
    duration=contiguous_duration(edge,v,s['record_support'],3,config['fps'])
    known_end=duration['future_end_reason']=='observed_local_relation_end'
    expansion=pos(masked_mean(D[:,51:],geom[:,51:])-D[:,50],'diameter_m')
    p4terms=np.stack([pos(future,'soft_change'),pos(distance_future,'distance_m'),known_end.astype(float),expansion],1)
    p4use=np.isfinite(p4terms);p4=np.divide(np.where(p4use,p4terms,0).sum(1),p4use.sum(1),out=np.full(n,np.nan),where=p4use.sum(1)>0)
    observed_range=np.max(np.where(v,S,-np.inf),1)-np.min(np.where(v,S,np.inf),1)
    observed_range=np.where(v.any(1),observed_range,np.nan)
    Sstd=masked_std(S,v);Smean=whole(S);peak=np.maximum(S[:,50]-Smean,0)
    occupancy=whole((edge>0).astype(float))
    shape_stability=np.exp(-masked_std(density,v)/scale['density']-masked_std(K,v))
    n1=np.clip(duration['close_duration_lower_bound_s']/scale['duration_s'],0,1)*shape_stability*(1-np.maximum(pos(approach,'soft_change'),pos(future,'soft_change')))
    n3=occupancy*np.exp(-Sstd/scale['soft_std'])*np.maximum(1-observed_range,0)*(1-pos(peak,'soft_change'))
    raw={'P1':p1,'P2':p2,'P3':p3,'P4':p4,'N1':n1,'N2':s['path_score'],'N3':n3}
    out={}
    # Timing metadata use observable frame counts, NEVER slot correspondence.
    same=s['global_count_same'];obs=s['observed'];rec=s['record_support']
    global_known=s.get('global_count_known',rec)
    count_different=global_known&~same
    for label,sl in [('past',slice(0,50)),('future',slice(51,101)),('center21',slice(40,61))]:
        out[f'{label}_record_n']=rec[:,sl].sum(1);out[f'{label}_observed_n']=obs[:,sl].sum(1);out[f'{label}_reliable_n']=v[:,sl].sum(1)
    out['current_strict']=s['strict'][:,50];out['current_observed']=obs[:,50]
    out['surrounding_qc_failed_n']=(rec&~s['strict']).sum(1)
    out['global_count_different_n']=count_different.sum(1)
    out['global_count_first_different_offset']=np.where(count_different.any(1),np.argmax(count_different,1)-50,np.nan)
    for side,sl in [('past',slice(0,50)),('future',slice(51,101))]:out[f'global_count_different_{side}_n']=count_different[:,sl].sum(1)
    pairs=global_known[:,1:]&global_known[:,:-1]
    events=((s['global_n_home'][:,1:]!=s['global_n_home'][:,:-1])|(s['global_n_away'][:,1:]!=s['global_n_away'][:,:-1]))&pairs
    out['global_count_transition_n']=events.sum(1)
    out['global_count_transition_offsets']='' # Filled below as per-row strings.
    out['global_count_transition_offsets']=['|'.join(map(str,np.flatnonzero(x)-49)) for x in events]
    out['global_count_different_offsets']=['|'.join(map(str,np.flatnonzero(x)-50)) for x in count_different]
    out['local_composition_changed']=(changed_from_center(s['anchor_n_home'],obs,slice(0,101))|changed_from_center(s['anchor_n_away'],obs,slice(0,101)))
    out['full_reliable_window']=v.all(1)
    out['window_max_missing_run']=run_statistics(~v)[1]
    for label,sl in zip(('far_past','near_past','near_future','far_future'),BANDS):
        out[label+'_reliable_n']=v[:,sl].sum(1)
    for label,m in [('record',rec),('observed',obs),('reliable',v)]:out[label+'_mask_hex']=[x.tobytes().hex() for x in np.packbits(m,axis=1,bitorder='little')]
    out.update({k:duration[k] for k in ('close_duration_lower_bound_s','onset_censored','end_censored','past_end_reason','future_end_reason')})
    for label,x in [('soft_approach',approach),('distance_approach_m',distance_approach),('soft_short_approach',short),
        ('soft_long_approach',long),('soft_future_drop',future),('distance_future_increase_m',distance_future),
        ('near_past_diameter_mean_m',mpD),('near_past_radius_mean_m',mpR),('soft_std',Sstd),('soft_range',observed_range),
        ('occupancy',occupancy),('nearby_count_change',kchange),('density_change',densitychange)]:out[label]=x
    out['P2_geometry_past_n']=geom_p;out['P4_geometry_future_n']=geom_f
    out['P4_term_mask']=np.sum(p4use.astype(int)*(1<<np.arange(4)),1)
    out['matching_prev_status']=s['prev_status'];out['matching_next_status']=s['next_status']
    out['matching_prev_candidates']=s['prev_candidate_count'];out['matching_next_candidates']=s['next_candidate_count']
    out['N2_path_valid']=s['path_valid'];out['N2_path_reason']=s['path_reason']
    for k in ('path_motion_m','path_diameter_range_m','path_radius_range_m','path_density_range'):out['N2_'+k]=s[k]
    for mode in ('full','partial'):
        for key,name in NAMES.items():
            state=support_state(s,config['policies'][key],mode=='full');reason=state['reason'].copy()
            def fail(bad,label):
                nonlocal reason
                reason=np.where((reason=='ok')&bad,label,reason)
            if key=='P2':fail(geom_p<config['policies']['P2']['geometry_min'],'insufficient_component_geometry')
            if key=='P3':fail(np.isin(s['prev_status'],['observation_ambiguous','boundary_or_outside'])|np.isin(s['next_status'],['observation_ambiguous','boundary_or_outside']),'matching_support_unknown')
            if key=='P4':fail(~np.isin(duration['future_end_reason'],['observed_local_relation_end','window_censored']),'relation_end_ambiguous')
            if key=='N2':fail(~s['path_valid'],'geometry_path_unknown')
            fail(~np.isfinite(raw[key]),'undefined_feature')
            valid=(reason=='ok')&state['valid'];score=np.where(valid,raw[key],np.nan)
            out[f'{mode}_{name}']=score;out[f'{mode}_{key}_valid']=valid;out[f'{mode}_{key}_reason']=reason
            out[f'{mode}_{key}_support']=np.where(valid,np.where(v.all(1),'full_reliable','experimental_partial'),np.where(obs[:,50],'spatial_only','insufficient_observation'))
            out[f'{mode}_{key}_observed_n']=state['observed_n'];out[f'{mode}_{key}_reliable_n']=state['valid_n'];out[f'{mode}_{key}_max_gap']=state['max_gap']
            out[f'{mode}_{key}_global_count_changed']=state['global_changed'];out[f'{mode}_{key}_local_composition_changed']=state['local_changed']
        # Missing components are excluded and EXPLICITLY identified; not zero evidence.
        p=np.stack([out[f'{mode}_{NAMES[k]}'] for k in ('P1','P2','P3','P4')],1)
        pv=np.isfinite(p);pn=pv.sum(1)
        out[f'{mode}_positive_evidence_score']=np.divide(np.where(pv,p,0).sum(1),pn,out=np.full(n,np.nan),where=pn>=config['positive_aggregation']['minimum_components'])
        out[f'{mode}_positive_used_mask']=np.sum(pv.astype(int)*(1<<np.arange(4)),1)
        out[f'{mode}_positive_used_n']=pn
        n1_=out[f'{mode}_{NAMES["N1"]}'];n3_=out[f'{mode}_{NAMES["N3"]}'];n2_=out[f'{mode}_{NAMES["N2"]}']
        persistence=np.where(np.isfinite(n1_),n1_,n3_)
        family=np.stack([persistence,n2_],1);fv=np.isfinite(family);fn=fv.sum(1)
        out[f'{mode}_negative_evidence_score']=np.divide(np.where(fv,family,0).sum(1),fn,out=np.full(n,np.nan),where=fn>=config['negative_aggregation']['minimum_families'])
        out[f'{mode}_negative_persistence_source']=np.where(np.isfinite(n1_),'N1',np.where(np.isfinite(n3_),'N3','none'))
        out[f'{mode}_negative_uses_N2']=np.isfinite(n2_);out[f'{mode}_negative_family_n']=fn
        for label in ('positive','negative'):
            valid=np.isfinite(out[f'{mode}_{label}_evidence_score'])
            out[f'{mode}_{label}_valid']=valid
            out[f'{mode}_{label}_reason']=np.where(valid,'ok',np.where(obs[:,50],'insufficient_components','current_anchor_unobserved'))
            out[f'{mode}_{label}_support']=np.where(valid,np.where(v.all(1),'full_reliable','experimental_partial'),np.where(obs[:,50],'spatial_only','insufficient_observation'))
        out[f'{mode}_N3_extrema_unverified']=~v.all(1)&np.isfinite(out[f'{mode}_{NAMES["N3"]}'])
    return pd.DataFrame(out)


def corrupt_series(series,deleted):
    """Whole-frame deletion with no interpolation; invalidate bounded paths."""
    result=dict(series)
    for key in ('observed','reliable'):result[key]=series[key]&~deleted
    result['global_count_known']=series.get('global_count_known',series['record_support'])&~deleted
    result['path_valid']=series['path_valid']&~deleted[:,40:61].any(1)
    result['path_reason']=np.where(result['path_valid'],'ok',np.where(deleted[:,40:61].any(1),'masked_geometry_path',series['path_reason']))
    for key in ('path_score','path_motion_m','path_diameter_range_m','path_radius_range_m','path_density_range'):
        result[key]=np.where(result['path_valid'],series[key],np.nan)
    return result
