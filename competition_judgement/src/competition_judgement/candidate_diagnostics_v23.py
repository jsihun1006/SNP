"""Validation-only descriptive agreement/sensitivity; v1 is never a label."""
from collections import Counter
import json
import numpy as np
import pandas as pd
from .candidate_decision_v23 import KEY, STATES, CANDIDATES, decide_clusters
from .pattern_profiles import profiles
from .local_temporal_features import local_features
from .local_temporal_source import load_local_source, LocalTemporalSource, mask_local_series
from .controlled_temporal_masking import position_mask, continuity_mask
from .missingness_experiments import draw_donors


def validation_analysis(val, frame, root, out):
    if not (val.split == 'validation').all(): raise ValueError('Validation only')
    distributions = []; reasons = []; supports = []; alternatives = []
    groups = {'all': np.ones(len(val), bool), 'strict': val.current_strict.to_numpy(bool),
              'low_quality': ~val.current_strict.to_numpy(bool), 'radius=3': val.radius_m.to_numpy() == 3, 'radius=5': val.radius_m.to_numpy() == 5}
    for group, selection in groups.items():
        sub = val.loc[selection]
        for state in STATES:
            n = int((sub.decision_state == state).sum())
            distributions.append(dict(unit='cluster_view', group=group, state=state, n=n, total=len(sub), fraction=n/len(sub) if len(sub) else None))
        for (state, reason), count in sub.groupby(['decision_state', 'decision_reason']).size().items():
            reasons.append(dict(group=group, state=state, kind='primary', reason=reason, n=int(count)))
        for bit, name in ((1,'A'), (2,'B'), (4,'short_RF'), (8,'near_RF')):
            reasons.append(dict(group=group, state='ANY_CANDIDATE', kind='basis', reason=name, n=int(((sub.decision_basis_mask.to_numpy(int)&bit)>0).sum())))
        for mask, count in sub.decision_basis_mask.value_counts().items():
            reasons.append(dict(group=group, state='ALL', kind='basis_combination', reason=str(mask), n=int(count)))
        for state, bystate in sub.groupby('decision_state'):
            for feature in ('P1','P2','P3','P4','N1','N2','N3'):
                for support, count in bystate[feature+'_state'].value_counts().items():
                    supports.append(dict(group=group, state=state, feature=feature, support=support, n=int(count), total=len(bystate)))
            for feature in ('N1','N2','N3'):
                x=bystate[feature].to_numpy(float);x=x[np.isfinite(x)]
                alternatives.append(dict(group=group,state=state,feature=feature,n=len(bystate),finite_n=len(x),median=float(np.median(x)) if len(x) else None,p90=float(np.quantile(x,.9)) if len(x) else None))
            for feature in ('C','D'):
                alternatives.append(dict(group=group,state=state,feature=feature,n=len(bystate),supported_n=int((bystate[feature+'_outcome']=='SUPPORTED').sum())))
    v1 = pd.read_csv(out.parent/'stage1_v1_final_20261109/candidate_frames.csv', usecols=['frame','period','split','candidate_state','strict_quality_valid'])
    if not np.array_equal(v1.frame, frame.frame): raise ValueError('v1 frame axis mismatch')
    selected = v1.split == 'validation'; fv=frame.loc[selected].reset_index(drop=True); old=v1.loc[selected].reset_index(drop=True)
    agreement=[]
    for group,sel in {'all':np.ones(len(fv),bool),'strict':fv.current_strict.to_numpy(bool),'low_quality':~fv.current_strict.to_numpy(bool)}.items():
        for (a,b),count in pd.DataFrame(dict(v1=old.candidate_state.to_numpy()[sel],v23=fv.decision_state.to_numpy()[sel])).groupby(['v1','v23']).size().items():
            agreement.append(dict(group=group,v1_state=a,v23_state=b,n=int(count),total=int(sel.sum())))
        for state in STATES:
            count=int((fv.decision_state.to_numpy()[sel]==state).sum())
            distributions.append(dict(unit='frame',group=group,state=state,n=count,total=int(sel.sum()),fraction=count/int(sel.sum()) if sel.any() else None))
    pd.DataFrame(agreement).to_csv(out/'v1_agreement.csv',index=False)
    runs=[]
    for name, states in [('v1',old.candidate_state.to_numpy()),('v23',fv.decision_state.to_numpy())]:
        ids=old.frame.to_numpy();period=old.period.to_numpy();role=old.split.to_numpy()
        starts=np.flatnonzero(np.r_[True,(np.diff(ids)!=1)|(period[1:]!=period[:-1])|(role[1:]!=role[:-1])|(states[1:]!=states[:-1])])
        ends=np.r_[starts[1:],len(ids)];lengths=ends-starts
        for state in np.unique(states):
            values=lengths[states[starts]==state]
            runs.append(dict(model=name,state=state,runs_n=len(values),frames_n=int(values.sum()),mean_s=float(values.mean()/25),median_s=float(np.median(values)/25),p95_s=float(np.quantile(values,.95)/25),max_s=float(values.max()/25)))
    pd.DataFrame(runs).to_csv(out/'consecutive_frame_runs.csv',index=False)
    pd.DataFrame(distributions).to_csv(out/'candidate_distribution.csv',index=False)
    pd.DataFrame(reasons).to_csv(out/'candidate_reason_summary.csv',index=False)
    pd.DataFrame(supports).to_csv(out/'candidate_support_summary.csv',index=False)
    pd.DataFrame(alternatives).to_csv(out/'alternative_evidence_diagnostic.csv',index=False)
    unknown=val[['decision_state','upstream_unknown_reasons']].copy();unknown['upstream_unknown_reasons']=unknown.upstream_unknown_reasons.str.split('|');unknown=unknown.explode('upstream_unknown_reasons')
    unknown.groupby(['decision_state','upstream_unknown_reasons']).size().rename('n').reset_index().to_csv(out/'upstream_unknown_reason_summary.csv',index=False)
    overlap=radius_analysis(val,root,out)
    return dict(validation_clusters=len(val),validation_frames=len(fv),v1_candidate_frames=int((old.candidate_state=='candidate').sum()),
                v23_candidate_frames=int(fv.decision_state.isin(CANDIDATES).sum()),radius_overlap=overlap)


def radius_analysis(val,root,out):
    columns=KEY+['family_id_5m','same_node_set_across_radii','cluster_centroid_x','cluster_centroid_y']
    old=pd.read_csv(root/'data/derived/competition_judgement/local_v2_1_20261109/local_clusters.csv',usecols=columns)
    current=val[KEY+['decision_state']].merge(old,on=KEY,validate='one_to_one')
    a=current[current.radius_m==3].copy(); b=current[current.radius_m==5].copy()
    b=b.drop(columns='family_id_5m').rename(columns={'cluster_id':'family_id_5m','decision_state':'parent_state','cluster_centroid_x':'parent_x','cluster_centroid_y':'parent_y'})
    joined=a.merge(b[['frame','family_id_5m','parent_state','parent_x','parent_y']],on=['frame','family_id_5m'],how='left',validate='many_to_one')
    if joined.parent_state.isna().any(): raise ValueError('Frozen radius family unresolved')
    joined['centroid_distance_m']=np.hypot(joined.cluster_centroid_x-joined.parent_x,joined.cluster_centroid_y-joined.parent_y)
    rows=[]
    for (a_state,b_state,same),sub in joined.groupby(['decision_state','parent_state','same_node_set_across_radii']):
        rows.append(dict(state_3m=a_state,state_5m=b_state,same_node_set=bool(same),n_3m_views=len(sub),
                         unique_5m_families=len(sub[['frame','family_id_5m']].drop_duplicates()),
                         centroid_separation_mean_m=float(sub.centroid_distance_m.mean()),centroid_separation_p95_m=float(sub.centroid_distance_m.quantile(.95))))
    pd.DataFrame(rows).to_csv(out/'radius_overlap_diagnostic.csv',index=False)
    return dict(validation_3m_views=len(a),validation_5m_views=len(b),same_node_set_3m_views=int(a.same_node_set_across_radii.sum()),
                candidate_both_radius_views=int((joined.decision_state.isin(CANDIDATES)&joined.parent_state.isin(CANDIDATES)).sum()),
                no_event_deduplication=True)


def compare_decisions(clean, changed, radii, label, family, metadata=None):
    rows=[];a=clean.decision_state.to_numpy();b=changed.decision_state.to_numpy()
    ca=np.isin(a,CANDIDATES);cb=np.isin(b,CANDIDATES)
    spatial=clean.spatial_gate_valid.to_numpy(bool)&changed.spatial_gate_valid.to_numpy(bool)
    sa=clean.dynamic_support_mask.to_numpy(int);sb=changed.dynamic_support_mask.to_numpy(int)
    for group,selected in [('all',np.ones(len(a),bool)),('radius=3',np.asarray(radii)==3),('radius=5',np.asarray(radii)==5)]:
        for old in STATES:
            for new in STATES:
                sel=selected&(a==old)&(b==new); count=int(sel.sum())
                if not count:continue
                rows.append(dict(condition=label,family=family,group=group,from_state=old,to_state=new,n=count,cohort_n=int(selected.sum()),
                    new_candidate_n=int((sel&~ca&cb).sum()),lost_candidate_n=int((sel&ca&~cb).sum()),state_changed_n=count if old!=new else 0,
                    both_spatial_valid_n=int((sel&spatial).sum()),same_dynamic_support_n=int((sel&spatial&(sa==sb)).sum()),
                    same_nonzero_dynamic_support_n=int((sel&spatial&(sa==sb)&(sa>0)).sum()),both_full_dynamic_n=int((sel&spatial&(sa==63)&(sb==63)).sum()),
                    lost_spatial_gate_n=int((sel&clean.spatial_gate_valid.to_numpy(bool)&~changed.spatial_gate_valid.to_numpy(bool)).sum()),
                    gained_spatial_gate_n=int((sel&~clean.spatial_gate_valid.to_numpy(bool)&changed.spatial_gate_valid.to_numpy(bool)).sum()),
                    **(metadata or {})))
    return rows


def masking_analysis(source, config, b1, local_config, legacy, out):
    stage_a=out.parent/'missingness_stage_a_final_20261109';keys=pd.read_csv(stage_a/'reference_keys.csv')
    index=source.clusters.reset_index().set_index(KEY)['index'];selected=index.loc[pd.MultiIndex.from_frame(keys[KEY])].to_numpy(int)
    if not (source.roles[source.cluster_frame[selected]]=='validation').all():raise ValueError('Validation cohort required')
    pieces=[source.query(selected[i:i+64]) for i in range(0,len(selected),64)]
    s={k:np.concatenate([part[k] for part in pieces]) if k!='offsets' else pieces[0][k] for k in pieces[0]}
    with np.load(stage_a/'reference_temporal_arrays.npz') as archive:
        for key in archive.files:np.testing.assert_array_equal(s[key],archive[key])
    radii=keys.radius_m.to_numpy(int);base=source.clusters.iloc[selected].reset_index(drop=True)
    def calculate(series):return decide_clusters(base,profiles(series,radii,b1,legacy),local_features(series,radii,local_config),config)
    clean=calculate(s);donors={};rows=[];conditions=[];n=len(base)
    with np.load(stage_a/'train_pattern_library.npz') as archive:
        for radius in (3,5):
            for basis in ('record','observed','reliable'):
                packed=np.packbits(archive[f'{radius}_{basis}_masks'],axis=1,bitorder='little')
                donors[(radius,basis)]=Counter({x.tobytes():int(w) for x,w in zip(packed,archive[f'{radius}_{basis}_weights'])})
    def run(label,mask,family,**metadata):
        if mask is None:conditions.append(dict(condition=label,status='INCOMPARABLE',**metadata));return
        changed=calculate(mask_local_series(s,mask))
        rows.extend(compare_decisions(clean,changed,radii,label,family,metadata))
        conditions.append(dict(condition=label,status='COMPUTED',center_lost_n=int(mask[:,50].sum()),**metadata))
    locations=('far_past','near_past','center_band','near_future','far_future','uniform','random','empirical','center_past_one_run','center_future_one_run')
    for count in config['robustness']['mask_counts']:
        for center in (False,True):
            for j,loc in enumerate(locations):
                empirical=None
                if loc=='empirical':
                    empirical,available=draw_donors(donors,radii,'reliable',np.random.default_rng(config['seed']+count+int(center)),count-int(center),True)
                    if not available.all():empirical=None
                mask=position_mask(n,count,loc,center,config['seed']+100*count+j,empirical)
                run(f'm{count}_{loc}_center_{"lost" if center else "kept"}',mask,'mask',missing_count=count,location=loc,center_deleted=center)
    for side in ('past','future'):
        for runs in ([10],[5,5],[2]*5,[1]*10):run(f'continuity_{side}_{len(runs)}',continuity_mask(n,runs,side),'continuity',missing_count=10,location=side)
    for basis in ('record','observed','reliable'):
        for rep in range(3):
            mask,available=draw_donors(donors,radii,basis,np.random.default_rng(config['seed']+1000+rep));assert available.all()
            run(f'empirical_{basis}_{rep}',mask,'empirical')
    pd.DataFrame(rows).to_csv(out/'masking_robustness.csv',index=False)
    pd.DataFrame(conditions).to_csv(out/'masking_conditions.csv',index=False)
    return dict(reference_windows=n,blocks=int(keys.support_block.nunique()),stage_a_arrays_bitwise_equal=True,
                computed_conditions=sum(x['status']=='COMPUTED' for x in conditions),incomparable_conditions=sum(x['status']=='INCOMPARABLE' for x in conditions),
                clean_state_counts={str(k):int(v) for k,v in clean.decision_state.value_counts().items()})


def noise_analysis(root,config,b1,local_config,legacy,out):
    from mtah_relational.frame_store import open_frame_cache
    from mtah_relational.splits import SplitManifest
    from .temporal_robustness import condition_source
    from .temporal_matching import reciprocal_unique
    temporal=json.loads((root/'competition_judgement/configs/temporal_relations_v2_2.json').read_text())
    local=json.loads((root/'competition_judgement/configs/local_clusters_v2_1.json').read_text())
    store=open_frame_cache(root/'data/derived/relational_temporal_v1_20261109/raw_cache')
    manifest=SplitManifest.load(root/'data/derived/relational_temporal_v1_20261109/split_manifest.json')
    base,c,offsets=condition_source(store,manifest,temporal,local,'clean',0.)
    source=LocalTemporalSource.from_memory(base,c,legacy)
    center_frames=base.ids[(offsets>=50)&(offsets<=100)];selected=np.flatnonzero(c.frame.isin(center_frames).to_numpy())
    if not (source.roles[source.cluster_frame[selected]]=='validation').all():raise ValueError('Validation only')
    probes=c.iloc[selected].reset_index(drop=True);r=probes.radius_m.to_numpy();centers=probes[['cluster_centroid_x','cluster_centroid_y']].to_numpy()
    anchor_r=np.maximum(6,probes.cluster_max_pairwise_distance_m.to_numpy()+3);rc=(r==5).astype(int)
    s=source.query(selected);clean=decide_clusters(probes,profiles(s,r,b1,legacy),local_features(s,r,local_config),config)
    groups=list(probes.groupby(['frame','radius_m']).indices.items());rows=[];bridge=[]
    for kind,strength in config['robustness']['conditions']:
        pert,c2,_=condition_source(store,manifest,temporal,local,kind,strength);changed_source=LocalTemporalSource.from_memory(pert,c2,legacy)
        mapped=np.full(len(probes),-1,int);right_groups=c2.groupby(['frame','radius_m']).indices
        for key,left in groups:
            right=right_groups.get(key,np.array([],int));a=probes.iloc[left];b=c2.iloc[right]
            _,pairs=reciprocal_unique(a[['cluster_centroid_x','cluster_centroid_y']].to_numpy(),b[['cluster_centroid_x','cluster_centroid_y']].to_numpy(),
                a.n_players.to_numpy(),b.n_players.to_numpy(),a.cluster_max_pairwise_distance_m.to_numpy(),b.cluster_max_pairwise_distance_m.to_numpy(),temporal)
            for i,j,_ in pairs:mapped[left[i]]=right[j]
        series=changed_source.query_fixed(probes.frame.to_numpy(),centers,anchor_r,rc,mapped)
        # Query keeps CLEAN fixed probes. Only unambiguous bridges supply current cluster geometry.
        current=probes.copy();good=mapped>=0
        for col in ('n_home','n_away','n_players','n_cross_edges','cluster_centroid_x','cluster_centroid_y','cluster_radius_m','cluster_max_pairwise_distance_m','cross_pair_density','contact_strength'):
            values=np.full(len(probes),np.nan);values[good]=c2.iloc[mapped[good]][col].to_numpy();current[col]=values
        position=np.searchsorted(changed_source.ids,probes.frame.to_numpy());current['strict_quality_valid']=changed_source.strict[position]
        current['period']=changed_source.periods[position]
        changed=decide_clusters(current,profiles(series,r,b1,legacy),local_features(series,r,local_config),config,mapped=good)
        label=f'{kind}_{strength:g}';rows.extend(compare_decisions(clean,changed,r,label,'tracking'))
        bridge.append(dict(condition=label,total=len(probes),unique_current_bridge_n=int(good.sum()),current_spatial_valid_n=int(changed.spatial_gate_valid.sum())))
        if kind in ('permutation','team_swap'):
            for key in clean:np.testing.assert_array_equal(clean[key],changed[key])
        print('v2-3 fixed-anchor robustness:',label,flush=True)
    pd.DataFrame(rows).to_csv(out/'tracking_robustness.csv',index=False)
    pd.DataFrame(bridge).to_csv(out/'tracking_probe_correspondence.csv',index=False)
    return dict(validation_support_frames=len(base.ids),center_frames=len(center_frames),clean_anchor_probes=len(probes),conditions=len(bridge),
                clean_state_counts={str(k):int(v) for k,v in clean.decision_state.value_counts().items()},
                no_forced_cluster_identity=True,permutation_and_team_swap_invariant=True)
