"""The seven frozen Stage B equations, without any cross-Evidence score."""
import numpy as np
from .temporal_features import masked_mean,contiguous_duration
from .evidence_features import masked_std,support_state
from .partial_window_diagnostics import run_statistics

FEATURES=('P1','P2','P3','P4','N1','N2','N3')
REASON_MAP={'ok':'OK','outside_or_no_record':'NO_RECORD','current_anchor_unobserved':'CURRENT_UNOBSERVED',
 'current_strict_qc_failed':'CURRENT_QC','insufficient_band_support':'BAND_SUPPORT','center_gap':'CENTER_GAP',
 'long_missing_run':'LONG_GAP','global_count_composition_ambiguous':'GLOBAL_COMPOSITION',
 'local_composition_ambiguous':'LOCAL_COMPOSITION'}


def equations(s,radii,c):
    """Same scales/formulas/optional P4 terms as evidence_features.py."""
    v=s['reliable'];n=len(v);sc=c['scales'];rad=np.asarray(radii)
    E=np.where(rad[:,None]==3,s['edges_3m'],s['edges_5m'])
    C=np.where(rad[:,None]==3,s['contact_size_3m'],s['contact_size_5m'])
    den=s['anchor_n_home']*s['anchor_n_away']
    rho=np.divide(E,den,out=np.full(E.shape,np.nan),where=den>0)
    S=s['soft'];d=s['nearest_m'];D=s['component_diameter'];R=s['component_radius'];K=s['nearby_count']
    pm=lambda x:masked_mean(x[:,:50],v[:,:50])
    fm=lambda x:masked_mean(x[:,51:],v[:,51:])
    npast=lambda x:masked_mean(x[:,25:50],v[:,25:50])
    nfuture=lambda x:masked_mean(x[:,51:76],v[:,51:76])
    pos=lambda x,key:np.clip(x/sc[key],0,1)
    approach=S[:,50]-pm(S);future=S[:,50]-fm(S)
    da=pm(d)-d[:,50];df=fm(d)-d[:,50]
    long=S[:,50]-masked_mean(S[:,:25],v[:,:25]);short=S[:,50]-npast(S)
    p1=np.mean([pos(da,'distance_m'),pos(approach,'soft_change'),pos(long-short,'soft_change')],axis=0)
    geom=v&np.isfinite(D)&np.isfinite(R)
    mpD=masked_mean(D[:,25:50],geom[:,25:50]);mpR=masked_mean(R[:,25:50],geom[:,25:50])
    p2=np.stack([pos(mpR-R[:,50],'radius_m'),pos(mpD-D[:,50],'diameter_m'),pos(C[:,50]-npast(C),'participants'),
        pos(np.log1p(E[:,50])-npast(np.log1p(E)),'log_edges'),pos(rho[:,50]-npast(rho),'density')],1).mean(1)
    branches=np.maximum(s['prev_candidate_count']-1,0)+np.maximum(s['next_candidate_count']-1,0)
    kc=(np.abs(K[:,50]-npast(K))+np.abs(nfuture(K)-K[:,50]))/2
    rc=(np.abs(rho[:,50]-npast(rho))+np.abs(nfuture(rho)-rho[:,50]))/2
    branchterm=np.clip(branches/2,0,1)
    p3=np.mean([pos(kc,'cluster_count'),pos(rc,'density'),branchterm],axis=0)
    duration=contiguous_duration(E,v,s['record_support'],3,c['fps'])
    end=duration['future_end_reason']=='observed_local_relation_end'
    expansion=pos(masked_mean(D[:,51:],geom[:,51:])-D[:,50],'diameter_m')
    terms=np.stack([pos(future,'soft_change'),pos(df,'distance_m'),end.astype(float),expansion],1)
    use=np.isfinite(terms)
    p4=np.divide(np.where(use,terms,0).sum(1),use.sum(1),out=np.full(n,np.nan),where=use.sum(1)>0)
    sr=np.max(np.where(v,S,-np.inf),1)-np.min(np.where(v,S,np.inf),1)
    sr=np.where(v.any(1),sr,np.nan);std=masked_std(S,v);mean=masked_mean(S,v)
    peak=np.maximum(S[:,50]-mean,0);occupancy=masked_mean((E>0).astype(float),v)
    stability=np.exp(-masked_std(rho,v)/sc['density']-masked_std(K,v))
    n1=np.clip(duration['close_duration_lower_bound_s']/sc['duration_s'],0,1)*stability*(1-np.maximum(pos(approach,'soft_change'),pos(future,'soft_change')))
    n3=occupancy*np.exp(-std/sc['soft_std'])*np.maximum(1-sr,0)*(1-pos(peak,'soft_change'))
    values=dict(P1=p1,P2=p2,P3=p3,P4=p4,N1=n1,N2=s['path_score'],N3=n3)
    raw=dict(E=E,C=C,rho=rho,approach=approach,distance_approach_m=da,future_drop=future,distance_future_m=df,
        radius_contract_m=mpR-R[:,50],diameter_contract_m=mpD-D[:,50],contact_growth=C[:,50]-npast(C),
        edge_growth=E[:,50]-npast(E),k_change=kc,density_change=rc,branch_term=branchterm,
        geom_past_n=geom[:,25:50].sum(1),P4_term_mask=np.sum(use.astype(int)*(1<<np.arange(4)),1),
        soft_range=sr,soft_std=std,occupancy=occupancy,**duration)
    return values,raw


def feature_profiles(s,radii,config,legacy):
    values,raw=equations(s,radii,legacy);v=s['reliable'];n=len(v);out={}
    current=s['record_support'][:,50]&s['observed'][:,50]&s['strict'][:,50]
    for key in FEATURES:
        p=legacy['policies'][key];lo,hi=config['dependencies'][key];sl=slice(lo+50,hi+51)
        policy=support_state(s,p,False)
        support=s['path_support'] if key=='N2' else v[:,sl]
        complete=support.all(1)&current
        if key=='N2':complete=support.all(1)&s['strict'][:,50]&s['record_support'][:,50]
        extra=np.ones(n,bool);extra_reason=np.full(n,'OK',object)
        def guard(condition,label):
            nonlocal extra,extra_reason
            extra_reason=np.where(extra&~condition,label,extra_reason);extra&=condition
        if p['global_constant']:guard(~policy['global_changed'],'GLOBAL_COMPOSITION')
        if p['local_constant']:guard(~policy['local_changed'],'LOCAL_COMPOSITION')
        if key=='P2':guard(raw['geom_past_n']>=p['geometry_min'],'GEOMETRY')
        if key=='P3':guard(~np.isin(s['prev_status'],['observation_ambiguous','boundary_or_outside','probe_mapping_unknown'])&
            ~np.isin(s['next_status'],['observation_ambiguous','boundary_or_outside','probe_mapping_unknown']),'MATCHING')
        if key=='P4':guard(np.isin(raw['future_end_reason'],['observed_local_relation_end','window_censored']),'END_AMBIGUOUS')
        if key=='N2':guard(s['path_valid'],'PATH')
        guard(np.isfinite(values[key]),'UNDEFINED')
        complete&=extra
        partial=~complete&policy['valid']&extra
        state=np.where(complete,'COMPLETE',np.where(partial,'EXPERIMENTAL_PARTIAL','UNKNOWN'))
        reason=np.array([REASON_MAP.get(x,x.upper()) for x in policy['reason']],object)
        reason=np.where(~extra,extra_reason,reason)
        reason=np.where(state!='UNKNOWN','OK',reason)
        reason=np.where(~s['strict'][:,50],'CURRENT_QC',reason)
        reason=np.where(~s['observed'][:,50]& (key!='N2'),'CURRENT_UNOBSERVED',reason)
        reason=np.where(~s['record_support'][:,50],'NO_RECORD',reason)
        out[key]=np.where(state!='UNKNOWN',values[key],np.nan);out[key+'_state']=state;out[key+'_reason']=reason
        out[key+'_reliable_n']=support.sum(1);out[key+'_max_gap']=run_statistics(~support)[1]
        out[key+'_composition_flag']=policy['global_changed'].astype(int)+2*policy['local_changed'].astype(int)
    out['available_mask']=sum((out[k+'_state']!='UNKNOWN').astype(int)<<i for i,k in enumerate(FEATURES))
    return out,raw
