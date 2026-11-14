"""Evidence availability and geometric sequence verification are separate layers."""
import numpy as np
import pandas as pd
from .pattern_values import FEATURES,feature_profiles

PATTERNS=('A','B','C','D')


def without_sequence(evidence,config):
    """Layer 1 import only: scores/means cannot certify ordered geometry."""
    out=evidence.copy()
    for key in FEATURES:
        undefined=out[key].isna();unknown=undefined|(out[key+'_state']=='UNKNOWN')
        if key+'_reason' not in out:out[key+'_reason']='OK'
        out.loc[undefined,key+'_reason']='UNDEFINED'
        out.loc[unknown&(out[key+'_reason']=='OK'),key+'_reason']='EVIDENCE_UNKNOWN'
        out.loc[unknown,key+'_state']='UNKNOWN';out.loc[unknown,key]=np.nan
    available=sum(((out[k+'_state']!='UNKNOWN')&out[k].notna()).to_numpy(int)<<i for i,k in enumerate(FEATURES))
    out['available_mask']=available
    for p,req in config['required_evidence'].items():
        states=out[[k+'_state' for k in req]]
        out[p+'_obs']=np.where((states=='UNKNOWN').any(axis=1),'UNKNOWN',np.where((states=='COMPLETE').all(axis=1),'COMPLETE','EXPERIMENTAL_PARTIAL'))
        out[p+'_seq']='UNVERIFIED';out[p+'_outcome']='UNKNOWN';out[p+'_reason']='NO_TIME_ARRAY'
        out[p+'_missing_mask']=sum(1<<FEATURES.index(k) for k in req)&~available
    return out


def trailing(x,width):return np.median(np.lib.stride_tricks.sliding_window_view(x,width,axis=1),axis=-1)


def profiles(s,radii,config,legacy):
    out,r=feature_profiles(s,radii,config,legacy);n=len(radii);q=config['sequence'];v=s['reliable']
    tol=q['numeric_tolerance'];sd=q['soft_deadband'];dd=q['distance_deadband_m'];lag=q['change_lag_frames'];w=q['smoothing_frames']
    # Values at missing positions never enter a successful temporal verification.
    # These diagnostic filters do not impute; their acceptance masks below require complete support.
    smooth=trailing(s['soft'],w);dist=trailing(s['nearest_m'],w);offset=np.arange(-50+w-1,51)
    rise=(smooth[:,lag:]-smooth[:,:-lag]>=sd-tol)&(dist[:,:-lag]-dist[:,lag:]>=dd-tol)
    fall=(smooth[:,:-lag]-smooth[:,lag:]>=sd-tol)&(dist[:,lag:]-dist[:,:-lag]>=dd-tol)
    start=offset[:-lag];end=offset[lag:]
    past_rise=(rise&(end>=-25)&(end<=0)).any(1)
    conv=(r['approach']>=sd-tol)&(r['distance_approach_m']>=dd-tol)&past_rise
    contracted=(r['radius_contract_m']>=q['contraction_deadband_m']-tol)&(r['diameter_contract_m']>=q['contraction_deadband_m']-tol)
    endcontract=(s['component_radius'][:,45]-s['component_radius'][:,50]>=q['contraction_deadband_m']-tol)&(
        s['component_diameter'][:,45]-s['component_diameter'][:,50]>=q['contraction_deadband_m']-tol)
    grown=(r['C'][:,50]-r['C'][:,45]>=q['count_growth'])&(r['E'][:,50]-r['E'][:,45]>=q['count_growth'])
    conc=(contracted&endcontract)|grown
    peakidx=np.argmax(smooth,1);row=np.arange(n);peak=offset[peakidx]
    pvalue=smooth[row,peakidx];pdist=dist[row,peakidx]
    farpast=np.median(s['soft'][:,:25],axis=1);farfuture=np.median(s['soft'][:,76:],axis=1)
    dpast=np.median(s['nearest_m'][:,:25],axis=1);dfuture=np.median(s['nearest_m'][:,76:],axis=1)
    risetime=np.max(np.where(rise&(end[None,:]<=peak[:,None]),end[None,:],-999),axis=1)
    falltime=np.min(np.where(fall&(start[None,:]>=peak[:,None]),start[None,:],999),axis=1)
    order=(np.abs(peak)<=q['peak_center_radius_frames'])&(risetime>-999)&(falltime<999)&(risetime<=peak)&(peak<=falltime)
    excursion=(pvalue-farpast>=sd-tol)&(pvalue-farfuture>=sd-tol)&(dpast-pdist>=dd-tol)&(dfuture-pdist>=dd-tol)
    structural=(np.ptp(s['nearby_count'],axis=1)>=1)|(np.ptp(r['E'],axis=1)>=1)
    transient=order&excursion&structural
    stable=(r['E']>0).all(1)&(r['soft_range']<=q['stable_soft_range']+tol)&(
        np.ptp(s['nearest_m'],axis=1)<=q['stable_distance_range_m']+tol)&(np.ptp(s['nearby_count'],axis=1)==0)&(np.ptp(r['E'],axis=1)==0)
    scale=legacy['scales']
    move=(s['path_motion_m']>=scale['motion_m']-tol)&(s['path_diameter_range_m']<=scale['motion_diameter_range_m']+tol)&(
        s['path_radius_range_m']<=scale['motion_radius_range_m']+tol)&(s['path_density_range']<=scale['density']+tol)
    checks={'A':conv|conc,'B':transient,'C':stable,'D':move}
    sequence_ready={'A':v[:,:51].all(1),'B':v.all(1),'C':v.all(1),'D':s['path_support'].all(1)&s['path_valid']&s['path_inside_anchor']}
    for p in PATTERNS:
        req=config['required_evidence'][p];states=np.stack([out[k+'_state'] for k in req],1)
        missing=(states=='UNKNOWN').any(1)
        obs=np.where(missing,'UNKNOWN',np.where((states=='COMPLETE').all(1),'COMPLETE','EXPERIMENTAL_PARTIAL'))
        if p=='D':obs=np.where(~s['path_inside_anchor'],'UNKNOWN',obs)
        ready=(obs=='COMPLETE')&sequence_ready[p]
        supported=ready&checks[p]
        out[p+'_obs']=obs;out[p+'_seq']=np.where(supported,'VERIFIED','UNVERIFIED')
        out[p+'_outcome']=np.where(supported,'SUPPORTED',np.where(ready,'NOT_OBSERVED','UNKNOWN'))
        out[p+'_reason']=np.where(missing,'MISSING_EVIDENCE',np.where(obs=='EXPERIMENTAL_PARTIAL','SEQUENCE_PARTIAL',np.where(~sequence_ready[p],'SEQUENCE_UNVERIFIED',np.where(checks[p],'OK','CONDITION_NOT_MET'))))
        if p=='D':out[p+'_reason']=np.where(~s['path_inside_anchor'],'ANCHOR_SCOPE_OR_PATH_UNKNOWN',out[p+'_reason'])
        required=sum(1<<FEATURES.index(k) for k in req)
        out[p+'_missing_mask']=required&~out['available_mask']
    out['A_convergence']=np.where((out['P1_state']=='COMPLETE')&v[:,:51].all(1),conv.astype(float),np.nan)
    out['A_concentration']=np.where((out['P2_state']=='COMPLETE')&v[:,25:51].all(1),conc.astype(float),np.nan)
    full=v.all(1)
    out['B_peak_offset']=np.where(full,peak,np.nan);out['B_rise_end_offset']=np.where(full&(risetime>-999),risetime,np.nan)
    out['B_fall_start_offset']=np.where(full&(falltime<999),falltime,np.nan)
    out['B_structure_changed']=np.where(full,structural.astype(float),np.nan)
    out['B_reapproach']=r['reapproach_observed']
    for name,value in [('approach_s',r['approach']),('approach_m',r['distance_approach_m']),('future_drop_s',r['future_drop']),
        ('radius_contract_m',r['radius_contract_m']),('diameter_contract_m',r['diameter_contract_m']),('edge_growth',r['edge_growth']),
        ('P3_k_change',r['k_change']),('P3_density_change',r['density_change']),('P3_branch_term',r['branch_term']),
        ('soft_range',r['soft_range']),('soft_std',r['soft_std']),('occupancy',r['occupancy'])]:out[name]=value
    out['P3_matching_ambiguity']=(np.isin(s['prev_status'],['geometric_ambiguous','observation_ambiguous','probe_mapping_unknown'])|
        np.isin(s['next_status'],['geometric_ambiguous','observation_ambiguous','probe_mapping_unknown']))
    out['P4_term_mask']=r['P4_term_mask']
    for key in ('close_duration_lower_bound_s','onset_censored','end_censored','future_end_reason'):out[key]=r[key]
    for key in ('path_motion_m','path_diameter_range_m','path_radius_range_m','path_density_range','path_inside_anchor'):out[key]=s[key]
    out['legacy_101_full']=full;out['current_strict']=s['strict'][:,50]
    out['current_anchor_home']=s['anchor_n_home'][:,50];out['current_anchor_away']=s['anchor_n_away'][:,50]
    out['current_soft']=np.where(s['observed'][:,50],s['soft'][:,50],np.nan);out['current_edges']=np.where(s['observed'][:,50],r['E'][:,50],np.nan)
    known=s.get('global_count_known',s['record_support'])
    local=(s['anchor_n_home']!=s['anchor_n_home'][:,50,None])|(s['anchor_n_away']!=s['anchor_n_away'][:,50,None])
    masks=dict(record=s['record_support'],observed=s['observed'],reliable=v,strict=s['strict']&s['record_support'],
        global_changed=~s['global_count_same']&known,local_changed=local&s['observed'],path=s['path_support'])
    for key,mask in masks.items():out[key+'_mask']=[x.tobytes().hex() for x in np.packbits(mask,axis=1,bitorder='little')]
    out['path_reason']=s['path_reason'];out['prev_match']=s['prev_status'];out['next_match']=s['next_status']
    return pd.DataFrame(out)
