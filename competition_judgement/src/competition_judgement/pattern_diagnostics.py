"""Fixed Validation analyses; geometry changes are not labelled prediction errors."""
from collections import Counter
import json
import numpy as np
import pandas as pd
from .pattern_values import FEATURES
from .pattern_profiles import PATTERNS,profiles
from .evidence_features import NAMES,corrupt_series
from .controlled_temporal_masking import position_mask,continuity_mask
from .missingness_experiments import draw_donors


def validation_analysis(val,frame,config,out):
    if not (val.split=='validation').all():raise ValueError('Validation only')
    rows=[];patterns=[];sequence=[];comparison=[]
    oldcols=['frame','radius_m','cluster_id']+[f'{mode}_{name}' for mode in ('full','partial') for name in NAMES.values()]
    old=pd.read_csv(out.parent/'evidence_stage_b_final_20261109/validation_evidence.csv.gz',usecols=oldcols)
    key=['frame','radius_m','cluster_id'];both=val.merge(old,on=key,validate='one_to_one')
    groups={'all':np.ones(len(val),bool),'strict':val.current_strict.to_numpy(bool),'low':~val.current_strict.to_numpy(bool),
        'radius=3':val.radius_m.to_numpy()==3,'radius=5':val.radius_m.to_numpy()==5}
    for group,sel in groups.items():
        for f in FEATURES:
            for state in ('COMPLETE','EXPERIMENTAL_PARTIAL','UNKNOWN'):
                rows.append(dict(group=group,feature=f,state=state,n=int((sel&(val[f+'_state'].to_numpy()==state)).sum()),total=int(sel.sum())))
        for p in PATTERNS:
            sub=val.loc[sel];n=len(sub)
            patterns.append(dict(group=group,pattern=p,total=n,available_n=int((sub[p+'_obs']!='UNKNOWN').sum()),
                supported_n=int((sub[p+'_outcome']=='SUPPORTED').sum()),not_observed_n=int((sub[p+'_outcome']=='NOT_OBSERVED').sum()),
                outcome_unknown_n=int((sub[p+'_outcome']=='UNKNOWN').sum()),observation_unknown_n=int((sub[p+'_obs']=='UNKNOWN').sum())))
            for (obs,seq,result,reason),n_ in sub.groupby([p+'_obs',p+'_seq',p+'_outcome',p+'_reason']).size().items():
                sequence.append(dict(group=group,pattern=p,observation=obs,sequence=seq,outcome=result,reason=reason,n=int(n_)))
    for f,name in NAMES.items():
        for mode in ('full','partial'):
            original=both[mode+'_'+name].to_numpy(float);current=both[f].to_numpy(float)
            common=np.isfinite(original)&np.isfinite(current);diff=np.abs(current[common]-original[common])
            comparison.append(dict(feature=f,legacy_mode=mode,legacy_valid_n=int(np.isfinite(original).sum()),
                current_complete_n=int((both[f+'_state']=='COMPLETE').sum()),current_partial_n=int((both[f+'_state']=='EXPERIMENTAL_PARTIAL').sum()),
                new_complete_prior_held_n=int(((both[f+'_state']=='COMPLETE')&~np.isfinite(original)).sum()),
                comparable_n=int(common.sum()),max_abs_difference=float(diff.max()) if len(diff) else None))
    pd.DataFrame(rows).to_csv(out/'feature_support_summary.csv',index=False)
    pd.DataFrame(patterns).to_csv(out/'pattern_availability.csv',index=False)
    pd.DataFrame(sequence).to_csv(out/'temporal_sequence_validation.csv',index=False)
    pd.DataFrame(comparison).to_csv(out/'legacy_comparison.csv',index=False)
    # Multi-label counts do not force a single type and do not count radius views as events.
    bits=sum((val[p+'_outcome']=='SUPPORTED').to_numpy(int)<<i for i,p in enumerate(PATTERNS))
    pd.Series(bits).value_counts().rename_axis('supported_pattern_mask').reset_index(name='anchor_views').to_csv(out/'pattern_overlap.csv',index=False)
    ids=val.frame.unique();fv=frame.loc[frame.frame.isin(ids)]
    (out/'frame_multilabel_summary.json').write_text(json.dumps(dict(validation_frames_with_clusters=len(fv),
        multi_pattern_supported_frames=int((fv.supported_pattern_types_n>=2).sum()),
        supported_type_hist={str(k):int(v) for k,v in fv.supported_pattern_types_n.value_counts().items()},
        note='Frame-level existence of different pattern types, potentially at different anchors; not one composite event.'),indent=2))
    common=both.N1.notna()&both.N3.notna();x=both.loc[common,'N1'];y=both.loc[common,'N3']
    (out/'alternative_evidence_diagnostic.json').write_text(json.dumps(dict(common_n=int(common.sum()),
        pearson=float(x.corr(y)) if len(x)>1 else None,spearman=float(x.rank().corr(y.rank())) if len(x)>1 else None,
        no_combined_alternative_score=True,
        p3_matching_ambiguous_n=int(both.P3_matching_ambiguity.sum()),
        p3_branch_positive_n=int((both.P3_branch_term>0).sum()),
        p3_branch_only_n=int(((both.P3_branch_term>0)&(both.P3_k_change==0)&(both.P3_density_change==0)).sum())),indent=2))


def mask_series(s,deleted):
    changed=corrupt_series(s,deleted)
    changed['path_support']=s['path_support']&~deleted[:,40:61]
    changed['path_inside_anchor']=s['path_inside_anchor']&changed['path_support'].all(1)
    return changed


def compare(clean,changed,series_a,series_b,label,family,radii,metadata=None,config=None):
    rows=[];n=len(clean)
    for f in FEATURES:
        a=clean[f].to_numpy(float);b=changed[f].to_numpy(float);common=np.isfinite(a)&np.isfinite(b)
        if f=='P4':common&=clean.P4_term_mask.to_numpy()==changed.P4_term_mask.to_numpy()
        aa=series_a['path_support'] if f=='N2' else series_a['reliable'];bb=series_b['path_support'] if f=='N2' else series_b['reliable']
        if f!='N2':
            lo,hi=config['dependencies'][f];aa=aa[:,lo+50:hi+51];bb=bb[:,lo+50:hi+51]
        same=np.all(aa==bb,axis=1);delta=np.abs(a[common]-b[common]);eq=common&same
        row=dict(condition=label,family=family,kind='feature',code=f,n=n,common_n=int(common.sum()),same_mask_n=int(eq.sum()),
            mae=float(delta.mean()) if len(delta) else None,p95_abs=float(np.quantile(delta,.95)) if len(delta) else None,
            same_mask_mae=float(np.abs(a[eq]-b[eq]).mean()) if eq.any() else None,
            complete_n=int((changed[f+'_state']=='COMPLETE').sum()),experimental_partial_n=int((changed[f+'_state']=='EXPERIMENTAL_PARTIAL').sum()),
            unknown_n=int((changed[f+'_state']=='UNKNOWN').sum()),observation_unknown_n=int((changed[f+'_state']=='UNKNOWN').sum()),
            support_changed_n=int((clean[f+'_state'].to_numpy()!=changed[f+'_state'].to_numpy()).sum()))
        row.update(metadata or {});rows.append(row)
    for p in PATTERNS:
        a=clean[p+'_outcome'].to_numpy();b=changed[p+'_outcome'].to_numpy()
        rows.append(dict(condition=label,family=family,kind='pattern',code=p,n=n,
            complete_n=int((changed[p+'_obs']=='COMPLETE').sum()),experimental_partial_n=int((changed[p+'_obs']=='EXPERIMENTAL_PARTIAL').sum()),
            unknown_n=int((b=='UNKNOWN').sum()),outcome_unknown_n=int((b=='UNKNOWN').sum()),
            observation_unknown_n=int((changed[p+'_obs']=='UNKNOWN').sum()),
            supported_n=int((b=='SUPPORTED').sum()),not_observed_n=int((b=='NOT_OBSERVED').sum()),
            lost_supported_n=int(((a=='SUPPORTED')&(b!='SUPPORTED')).sum()),new_supported_n=int(((a!='SUPPORTED')&(b=='SUPPORTED')).sum()),
            sequence_changed_n=int((clean[p+'_seq'].to_numpy()!=changed[p+'_seq'].to_numpy()).sum()),**(metadata or {})))
    # No gap, ambiguous matching or missing Evidence may be called a verified event.
    for p in PATTERNS:
        hit=changed[p+'_seq'].to_numpy()=='VERIFIED'
        assert (changed.loc[hit,p+'_obs']=='COMPLETE').all()
    return rows


def masking_analysis(source,config,legacy,out):
    stage_a=out.parent/'missingness_stage_a_final_20261109'
    keys=pd.read_csv(stage_a/'reference_keys.csv');index=source.clusters.reset_index().set_index(['frame','radius_m','cluster_id'])['index']
    selected=index.loc[pd.MultiIndex.from_frame(keys[['frame','radius_m','cluster_id']])].to_numpy(int)
    pieces=[source.query(selected[i:i+64]) for i in range(0,len(selected),64)]
    s={k:np.concatenate([part[k] for part in pieces]) if k!='offsets' else pieces[0][k] for k in pieces[0]}
    with np.load(stage_a/'reference_temporal_arrays.npz') as archive:
        for k in archive.files:assert np.array_equal(s[k],archive[k],equal_nan=True)
    radii=keys.radius_m.to_numpy(int);clean=profiles(s,radii,config,legacy);n=len(clean);rows=[];conditions=[];donors={}
    with np.load(stage_a/'train_pattern_library.npz') as archive:
        for radius in (3,5):
            for basis in ('record','observed','reliable'):
                packed=np.packbits(archive[f'{radius}_{basis}_masks'],axis=1,bitorder='little')
                donors[(radius,basis)]=Counter({x.tobytes():int(w) for x,w in zip(packed,archive[f'{radius}_{basis}_weights'])})
    def run(label,m,family,**meta):
        if m is None:conditions.append(dict(condition=label,status='INCOMPARABLE',**meta));return
        changed_s=mask_series(s,m);changed=profiles(changed_s,radii,config,legacy)
        rows.extend(compare(clean,changed,s,changed_s,label,family,radii,meta,config))
        conditions.append(dict(condition=label,status='COMPUTED',center_lost_n=int(m[:,50].sum()),**meta))
    locations=('far_past','near_past','center_band','near_future','far_future','uniform','random','empirical','center_past_one_run','center_future_one_run')
    for count in config['robustness']['mask_counts']:
        for center in (False,True):
            for j,loc in enumerate(locations):
                empirical=None
                if loc=='empirical':
                    empirical,available=draw_donors(donors,radii,'reliable',np.random.default_rng(config['seed']+count+int(center)),count-int(center),True)
                    if not available.all():empirical=None
                m=position_mask(n,count,loc,center,config['seed']+100*count+j,empirical)
                run(f'm{count}_{loc}_center_{"lost" if center else "kept"}',m,'mask',missing_count=count,location=loc)
    for side in ('past','future'):
        for runs in ([10],[5,5],[2]*5,[1]*10):run(f'continuity_{side}_{len(runs)}',continuity_mask(n,runs,side),'continuity',missing_count=10,location=side)
    for basis in ('record','observed','reliable'):
        for rep in range(3):
            m,available=draw_donors(donors,radii,basis,np.random.default_rng(config['seed']+1000+rep));assert available.all()
            run(f'empirical_{basis}_{rep}',m,'empirical')
    pd.DataFrame(rows).to_csv(out/'masking_robustness.csv',index=False)
    pd.DataFrame(conditions).to_csv(out/'masking_conditions.csv',index=False)
    return dict(reference_windows=n,blocks=int(keys.support_block.nunique()),stage_a_arrays_bitwise_equal=True,
        computed_conditions=sum(x['status']=='COMPUTED' for x in conditions),incomparable_conditions=sum(x['status']=='INCOMPARABLE' for x in conditions))


def noise_analysis(root,config,legacy,out):
    from mtah_relational.frame_store import open_frame_cache
    from mtah_relational.splits import SplitManifest
    from .temporal_robustness import condition_source
    from .temporal_matching import reciprocal_unique
    from .pattern_source import PatternSource
    temporal=json.loads((root/'competition_judgement/configs/temporal_relations_v2_2.json').read_text())
    local=json.loads((root/'competition_judgement/configs/local_clusters_v2_1.json').read_text())
    store=open_frame_cache(root/'data/derived/relational_temporal_v1_20261109/raw_cache')
    manifest=SplitManifest.load(root/'data/derived/relational_temporal_v1_20261109/split_manifest.json')
    base,c,offsets=condition_source(store,manifest,temporal,local,'clean',0.)
    source=PatternSource.from_memory(base,c,legacy)
    frames=base.ids[(offsets>=50)&(offsets<=100)];selected=np.flatnonzero(c.frame.isin(frames).to_numpy())
    probes=c.iloc[selected].reset_index(drop=True);centers=probes[['cluster_centroid_x','cluster_centroid_y']].to_numpy()
    radii=np.maximum(6,probes.cluster_max_pairwise_distance_m.to_numpy()+3);rc=(probes.radius_m.to_numpy()==5).astype(int)
    clean_s=source.query(selected);clean=profiles(clean_s,probes.radius_m.to_numpy(),config,legacy)
    groups=list(probes.groupby(['frame','radius_m']).indices.items());rows=[];bridge=[]
    for kind,strength in config['robustness']['conditions']:
        pert,c2,_=condition_source(store,manifest,temporal,local,kind,strength)
        changed_source=PatternSource.from_memory(pert,c2,legacy);mapped=np.full(len(probes),-1,int)
        c2groups=c2.groupby(['frame','radius_m']).indices
        for key,left in groups:
            right=c2groups.get(key,np.array([],int));a=probes.iloc[left];b=c2.iloc[right]
            _,pairs=reciprocal_unique(a[['cluster_centroid_x','cluster_centroid_y']].to_numpy(),b[['cluster_centroid_x','cluster_centroid_y']].to_numpy(),
                a.n_players.to_numpy(),b.n_players.to_numpy(),a.cluster_max_pairwise_distance_m.to_numpy(),b.cluster_max_pairwise_distance_m.to_numpy(),temporal)
            for i,j,_ in pairs:mapped[left[i]]=right[j]
        changed_s=changed_source.query_fixed(probes.frame.to_numpy(),centers,radii,rc,mapped)
        changed=profiles(changed_s,probes.radius_m.to_numpy(),config,legacy)
        label=f'{kind}_{strength:g}';rows.extend(compare(clean,changed,clean_s,changed_s,label,'tracking_noise',probes.radius_m.to_numpy(),config=config))
        bridge.append(dict(condition=label,total=len(probes),unique_current_bridge_n=int((mapped>=0).sum())))
        if kind in ('permutation','team_swap'):
            for feature in FEATURES:assert np.allclose(clean[feature],changed[feature],equal_nan=True,atol=1e-10,rtol=0)
            for pattern in PATTERNS:assert np.array_equal(clean[pattern+'_outcome'],changed[pattern+'_outcome'])
        print('B.1 fixed-anchor tracking robustness:',label,flush=True)
    pd.DataFrame(rows).to_csv(out/'tracking_robustness.csv',index=False)
    pd.DataFrame(bridge).to_csv(out/'tracking_probe_correspondence.csv',index=False)
    return dict(validation_support_frames=len(base.ids),center_frames=len(frames),clean_anchor_probes=len(probes),conditions=len(bridge),
        fixed_spatial_anchors=True,no_forced_cluster_identity=True)
