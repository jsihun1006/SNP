"""Validation-only descriptive and paired masking analyses. No rule fitting."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .evidence_features import NAMES,compute_evidence,corrupt_series
from .evidence_frames import SCORES
from .controlled_temporal_masking import position_mask,continuity_mask
from .missingness_experiments import draw_donors,paired_block_bootstrap


def distribution(x):
    a=np.asarray(x,float);a=a[np.isfinite(a)]
    if not len(a):return dict(n=0,mean=None,std=None,p05=None,median=None,p95=None)
    return dict(n=len(a),mean=float(a.mean()),std=float(a.std()),p05=float(np.quantile(a,.05)),median=float(np.median(a)),p95=float(np.quantile(a,.95)))


def correlations(table,pairs,group):
    rows=[]
    for a,b in pairs:
        good=table[[a,b]].replace([np.inf,-np.inf],np.nan).dropna()
        enough=len(good)>1 and good[a].std()>0 and good[b].std()>0
        rows.append(dict(group=group,left=a,right=b,n=len(good),pearson=float(good[a].corr(good[b])) if enough else None,
            spearman=float(good[a].rank(method='average').corr(good[b].rank(method='average'))) if enough else None))
    return rows


def validation_analysis(val,frame,original,config,out):
    if not (val.split=='validation').all():raise ValueError('Validation-only diagnostics')
    rows=[];joint=[];corr=[];reasons=[]
    basegroups={'all':np.ones(len(val),bool),'strict':val.strict_quality_valid.to_numpy(bool),'low':~val.strict_quality_valid.to_numpy(bool),
        'radius=3':val.radius_m.to_numpy()==3,'radius=5':val.radius_m.to_numpy()==5}
    for mode in ('full','partial'):
        for group,select in basegroups.items():
            for score in SCORES:
                values=val.loc[select,mode+'_'+score]
                rows.append(dict(mode=mode,group=group,feature=score,total=int(select.sum()),coverage=float(values.notna().mean()),**distribution(values)))
        for group,sel in [('all',np.ones(len(val),bool)),('partial_only',~val.full_reliable_window.to_numpy(bool))]:
            sub=val.loc[sel];p=sub[mode+'_positive_evidence_score'];n=sub[mode+'_negative_evidence_score'];valid=p.notna()&n.notna()
            high=(p>=config['diagnostics']['high_band'])&(n>=config['diagnostics']['high_band'])&valid
            joint.append(dict(mode=mode,group=group,total=len(sub),both_valid_n=int(valid.sum()),both_high_n=int(high.sum()),
                both_high_fraction_of_joint=float(high.sum()/valid.sum()) if valid.any() else None,
                persistence_sources=sub[mode+'_negative_persistence_source'].value_counts().to_dict()))
            targets=[mode+'_'+s for s in SCORES]
            pairs=[(a,b) for i,a in enumerate(targets) for b in targets[i+1:]]
            corr.extend(correlations(sub,pairs,mode+':'+group))
        for key in NAMES:
            count=val[mode+'_'+key+'_reason'].value_counts()
            reasons.extend(dict(mode=mode,feature=key,reason=str(k),n=int(v)) for k,v in count.items())
    # Compare frame-level v1/v2 only on identical Validation frames, no labels.
    signalcols=['v1_candidate_score','max_local_contact_strength_3m','max_local_contact_strength_5m']
    val=val.merge(original[['frame',*signalcols]],on='frame',validate='many_to_one')
    for mode in ('full','partial'):
        corr.extend(correlations(val,[(mode+'_'+s,c) for s in SCORES for c in signalcols],mode+':anchor_weighted_v1_v2'))
    ids=val.frame.unique();fv=frame.loc[frame.frame.isin(ids)].merge(original[['frame',*signalcols]],on='frame',validate='one_to_one')
    for mode in ('full','partial'):
        corr.extend(correlations(fv,[(mode+'_'+s+'_max',c) for s in SCORES for c in signalcols],mode+':unique_validation_frame_v1_v2'))
    pd.DataFrame(rows).to_csv(out/'validation_distributions.csv',index=False)
    pd.DataFrame(corr).to_csv(out/'validation_correlations.csv',index=False)
    pd.DataFrame(reasons).to_csv(out/'validation_reason_counts.csv',index=False)
    pd.DataFrame(joint).to_json(out/'validation_joint_summary.json',orient='records',indent=2)


def identical_components(clean,changed,score):
    """Compare aggregate errors only at identical component/term weights."""
    same=np.ones(len(clean),bool)
    if score=='P4_subsequent_separation':same=clean.P4_term_mask.to_numpy()==changed.P4_term_mask.to_numpy()
    if score=='positive_evidence_score':
        same=clean.full_positive_used_mask.to_numpy()==changed.partial_positive_used_mask.to_numpy()
        p4_used=(clean.full_positive_used_mask.to_numpy(int)&8)>0
        same&=~p4_used|(clean.P4_term_mask.to_numpy()==changed.P4_term_mask.to_numpy())
    if score=='negative_evidence_score':
        same=(clean.full_negative_persistence_source.to_numpy()==changed.partial_negative_persistence_source.to_numpy())&(clean.full_negative_uses_N2.to_numpy()==changed.partial_negative_uses_N2.to_numpy())
    return same


def masking_analysis(source,config,out,stage_a_dir=None):
    stage_a=Path(stage_a_dir) if stage_a_dir is not None else out.parent/'missingness_stage_a_final_20261109'
    reference=pd.read_csv(stage_a/'reference_keys.csv')
    if len(reference)!=1778:raise ValueError('Frozen Stage A cohort mismatch')
    index=source.clusters.reset_index().set_index(['frame','radius_m','cluster_id'])['index']
    selected=index.loc[pd.MultiIndex.from_frame(reference[['frame','radius_m','cluster_id']])].to_numpy(int)
    pieces=[];batch=config['batch_anchors']
    for start in range(0,len(selected),batch):pieces.append(source.query(selected[start:start+batch]))
    series={k:np.concatenate([s[k] for s in pieces]) if isinstance(pieces[0][k],np.ndarray) and pieces[0][k].ndim>0 and k!='offsets' else pieces[0][k] for k in pieces[0]}
    with np.load(stage_a/'reference_temporal_arrays.npz') as old:
        for k in old.files:
            if not np.allclose(series[k],old[k],atol=1e-7,equal_nan=True):raise ValueError('Stage A reference arrays changed')
    radii=reference.radius_m.to_numpy(int);n=len(reference)
    if not (source.roles[source.cluster_frame[selected]]=='validation').all():raise ValueError('Non-Validation reference')
    clean=compute_evidence(series,radii,config)
    clean.to_csv(out/'masking_reference_evidence.csv.gz',index=False,compression='gzip')
    reference.to_csv(out/'masking_reference_keys.csv',index=False)
    donors={}
    with np.load(stage_a/'train_pattern_library.npz') as lib:
        for r in (3,5):
            for basis in ('record','observed','reliable'):
                patterns=np.packbits(lib[f'{r}_{basis}_masks'],axis=1,bitorder='little');weights=lib[f'{r}_{basis}_weights']
                donors[(r,basis)]=Counter({p.tobytes():int(w) for p,w in zip(patterns,weights)})
    rows=[];conditions=[];paired={};groups={'all':np.ones(n,bool),'radius=3':radii==3,'radius=5':radii==5}
    def run(name,deleted,family,count=None,location=None):
        if deleted is None:
            conditions.append(dict(condition=name,status='incomparable',family=family,count=count,location=location));return
        changed=compute_evidence(corrupt_series(series,deleted),radii,config)
        conditions.append(dict(condition=name,status='computed',family=family,count=count,location=location,n=n,center_lost_n=int(deleted[:,50].sum())))
        for score in SCORES:
            target=clean['full_'+score].to_numpy(float);estimate=changed['partial_'+score].to_numpy(float)
            same=identical_components(clean,changed,score)
            for group,sel in groups.items():
                bv=sel&np.isfinite(target);ev=sel&np.isfinite(estimate);common=bv&ev&same
                delta=estimate[common]-target[common];ae=np.abs(delta)
                rows.append(dict(condition=name,family=family,count=count,location=location,group=group,feature=score,
                    references=int(sel.sum()),baseline_valid_n=int(bv.sum()),partial_valid_n=int(ev.sum()),
                    comparable_n=int(common.sum()),lost_valid_n=int((bv&~ev).sum()),newly_valid_n=int((ev&~bv).sum()),
                    composition_changed_n=int((bv&ev&~same).sum()),coverage=float(ev.sum()/sel.sum()),
                    mae=float(ae.mean()) if len(ae) else None,median_ae=float(np.median(ae)) if len(ae) else None,
                    p95_ae=float(np.quantile(ae,.95)) if len(ae) else None,bias=float(delta.mean()) if len(ae) else None,
                    rmse=float(np.sqrt(np.mean(delta**2))) if len(ae) else None,
                    small_error_fraction=float((ae<=config['diagnostics']['small_error']).mean()) if len(ae) else None,
                    # For nonnegative evidence, threshold crossing is descriptive, not a direction flip.
                    high_band_crossing_fraction=float(((estimate[common]>=config['diagnostics']['high_band'])!=(target[common]>=config['diagnostics']['high_band'])).mean()) if len(ae) else None))
            if family=='position' and not deleted[:,50].any():paired[(count,location,score)]=np.where(np.isfinite(target)&np.isfinite(estimate)&same,np.abs(estimate-target),np.nan)
    locations=('far_past','near_past','center_band','near_future','far_future','uniform','random','empirical',
               'center_past_one_run','center_future_one_run')
    for count in config['diagnostics']['mask_counts']:
        for center in (False,True):
            for j,location in enumerate(locations):
                empirical=None
                if location=='empirical':
                    empirical,available=draw_donors(donors,radii,'reliable',np.random.default_rng(config['seed']+count+int(center)),count-int(center),True)
                    if not available.all():empirical=None
                deletion=position_mask(n,count,location,center,config['seed']+100*count+j,empirical)
                run(f'm{count}_{location}_center_{"lost" if center else "kept"}',deletion,'position',count,location)
        print(f'Stage B Stage A paired masks complete: {count}',flush=True)
    for side in ('past','future'):
        for lengths in ([10],[5,5],[2]*5,[1]*10):run(f'continuity_{side}_{len(lengths)}runs',continuity_mask(n,lengths,side),'continuity',10,side)
    for basis in ('record','observed','reliable'):
        for rep in range(3):
            deletion,available=draw_donors(donors,radii,basis,np.random.default_rng(config['seed']+1000+rep))
            if not available.all():raise ValueError('Missing Train donor')
            run(f'empirical_{basis}_rep{rep}',deletion,'empirical',location=basis)
    comparisons=[]
    for count in config['diagnostics']['mask_counts']:
        for score in SCORES:
            for near,far in [('center_past_one_run','far_past'),('center_future_one_run','far_future')]:
                ka=(count,near,score);kb=(count,far,score)
                if ka not in paired or kb not in paired:continue
                comparisons.append(dict(count=count,feature=score,near=near,far=far,
                    **paired_block_bootstrap(paired[ka]-paired[kb],reference.support_block.to_numpy(),
                        config['diagnostics']['bootstrap_replicates'],config['seed'])))
    pd.DataFrame(rows).to_csv(out/'masking_distortion.csv',index=False)
    pd.DataFrame(conditions).to_csv(out/'masking_conditions.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'masking_paired_block_comparisons.csv',index=False)
    return dict(reference_windows=n,support_blocks=int(reference.support_block.nunique()),
        stage_a_keys_arrays_exact=True,conditions=len(conditions),computed_conditions=sum(c['status']=='computed' for c in conditions),
        labels_used=False,original_stage_a_unmodified=True)
