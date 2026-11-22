"""Exact-count, paired Stage A experiments on a fixed Validation cohort."""
import json
import numpy as np
import pandas as pd
from .controlled_temporal_masking import (FEATURES,FULL_ONLY,SIGNED,diagnostic_estimates,
    position_mask,continuity_mask,error_summary,feature_tolerances)
from .partial_window_diagnostics import policy_predicates,run_statistics


def draw_donors(donors,radii,basis,rng,exact_count=None,keep_center=False):
    """Frequency-weighted draws from Train ONLY; no outcome-based selection."""
    out=np.zeros((len(radii),101),bool);available=np.ones(len(radii),bool)
    for r in (3,5):
        select=np.flatnonzero(radii==r);counts=donors[(r,basis)]
        keys=list(counts);weights=np.array([counts[k] for k in keys],float)
        patterns=np.unpackbits(np.frombuffer(b''.join(keys),np.uint8).reshape(-1,13),axis=1,bitorder='little')[:,:101].astype(bool)
        eligible=np.ones(len(keys),bool)
        if exact_count is not None:eligible&=patterns.sum(1)==exact_count
        if keep_center:eligible&=~patterns[:,50]
        if not eligible.any():available[select]=False;continue
        p=weights[eligible];p/=p.sum()
        out[select]=patterns[eligible][rng.choice(int(eligible.sum()),len(select),p=p)]
    return out,available


def paired_block_bootstrap(delta,blocks,replicates,seed):
    """Resample disjoint raw-support blocks, keeping repeated clusters together."""
    valid=np.isfinite(delta);delta=delta[valid];blocks=np.asarray(blocks)[valid]
    unique,idx=np.unique(blocks,return_inverse=True)
    if not len(unique):return dict(n=0,blocks=0,mean=None,ci_low=None,ci_high=None)
    sums=np.bincount(idx,weights=delta);counts=np.bincount(idx)
    rng=np.random.default_rng(seed);pick=rng.integers(len(unique),size=(replicates,len(unique)))
    estimates=sums[pick].sum(1)/counts[pick].sum(1)
    return dict(n=len(delta),blocks=len(unique),mean=float(delta.mean()),
        ci_low=float(np.quantile(estimates,.025)),ci_high=float(np.quantile(estimates,.975)))


def run_experiments(reference,series,donors,config,out):
    radius=reference.radius_m.to_numpy(int);n=len(reference);blocks=reference.support_block.to_numpy()
    full=np.ones((n,101),bool);clean=diagnostic_estimates(series,full,radius)
    if any(not np.allclose(clean[f],reference[f],rtol=1e-6,atol=1e-6,equal_nan=True) for f in FEATURES):
        raise ValueError('Frozen CSV and recomputed clean features differ')
    summary=[];conditions=[];policy_rows=[];support_rows=[];errors_for_pairing={};sweep_ranges={}
    groups={'all':np.ones(n,bool),'radius=3':radius==3,'radius=5':radius==5,
            'size=2':reference.n_players.to_numpy()==2,'size=3+':reference.n_players.to_numpy()>=3}

    def evaluate(name,deleted,*,family,count=None,location=None,compare_policy=False,extra=None):
        if deleted is None:
            conditions.append(dict(condition=name,family=family,count=count,location=location,status='incomparable_band_capacity_or_no_exact_Train_donor'))
            return
        valid=~deleted;center=valid[:,50]
        metadata=dict(condition=name,family=family,count=count,location=location,n=n,
            center_lost_n=int((~center).sum()),status='computed',**(extra or {}))
        conditions.append(metadata)
        est=diagnostic_estimates(series,valid,radius)
        _,maximum,_=run_statistics(deleted)
        # Official full-only features are NaN after any deletion; center loss has no score.
        for group,select in groups.items():
            select=select&center
            for f in FEATURES:
                row=dict(condition=name,family=family,count=count,location=location,group=group,feature=f,
                    **error_summary(clean[f][select],est[f][select],f,config))
                row['official_nan_fraction']=(float((~valid.all(1))[select].mean()) if select.any() else None) if f in FULL_ONLY else row['nan_fraction']
                summary.append(row)
        if family in ('position','continuity'):
            errors_for_pairing[name]={f:np.where(center,np.abs(est[f]-clean[f]),np.nan) for f in FEATURES}
        if family=='one_run_sweep':
            for f in FEATURES:
                error=np.abs(est[f]-clean[f]);key=(count,f)
                if key not in sweep_ranges:sweep_ranges[key]=[error.copy(),error.copy(),est[f].copy(),est[f].copy()]
                else:
                    lo,hi,plo,phi=sweep_ranges[key]
                    np.minimum(lo,error,out=lo);np.maximum(hi,error,out=hi)
                    np.minimum(plo,est[f],out=plo);np.maximum(phi,est[f],out=phi)
        if compare_policy:
            predicates=policy_predicates(valid,config,current_strict=np.ones(n,bool),
                 count_changed=np.zeros(n,bool),center_relation=center)
            for (p,g),eligible in predicates.items():
                # All QC/count gates are constant for deletion-only clean references.
                if g in ('current_strict','constant_global_count','center_relation_valid'):continue
                for f in FEATURES:
                    metric=error_summary(clean[f][eligible],est[f][eligible],f,config)
                    policy_rows.append(dict(condition=name,family=family,feature=f,percent=p,gate=g,n_total=n,
                        coverage=float(eligible.mean()),**metric))
        if family=='empirical_unconditional':
            support=np.minimum(valid[:,:50].mean(1),valid[:,51:].mean(1))
            for low,high in [(0,.6),(.6,.7),(.7,.8),(.8,.9),(.9,1.),(1.,1.01)]:
                sel=center&(support>=low)&(support<high)
                for f in FEATURES:
                    support_rows.append(dict(condition=name,support_low=low,support_high=high,feature=f,
                         **error_summary(clean[f][sel],est[f][sel],f,config)))
        if center.any():
            lo=est['close_duration_lower_bound_s'][center];hi=est['duration_upper_bound_s'][center]
            target=clean['close_duration_lower_bound_s'][center]
            metadata.update(duration_interval_coverage=float(((target>=lo-1e-9)&(target<=hi+1e-9)).mean()),
                duration_interval_width_mean_s=float((hi-lo).mean()),max_missing_run_mean=float(maximum[center].mean()))

    locations=('far_past','near_past','center_band','near_future','far_future','uniform','random','empirical',
               'center_past_one_run','center_future_one_run')
    for count in config['missing_counts']:
        for center_deleted in (False,True):
            for j,location in enumerate(locations):
                empirical=None
                if location=='empirical':
                    empirical,available=draw_donors(donors,radius,'reliable',np.random.default_rng(config['seed']+count+int(center_deleted)),
                        exact_count=count-int(center_deleted),keep_center=True)
                    # Identical cohort across locations: do NOT silently drop unmatched radii.
                    if not available.all():empirical=None
                deletion=position_mask(n,count,location,center_deleted,config['seed']+100*count+j,empirical)
                name=f'position_m{count}_{location}_center_{"lost" if center_deleted else "kept"}'
                evaluate(name,deletion,family='position',count=count,location=location,compare_policy=not center_deleted)
        # Isolate location AND keep continuity fixed: every allowed one-sided single run.
        for side,start in [('past',s) for s in range(51-count)]+[('future',s) for s in range(51-count)]:
            deletion=np.zeros((n,101),bool);absolute=start+(0 if side=='past' else 51)
            deletion[:,absolute:absolute+count]=True
            evaluate(f'sweep_m{count}_{side}_start{start}',deletion,family='one_run_sweep',count=count,location=f'{side}:{start}')
        print(f'Stage A fixed-count and location sweep done: {count}',flush=True)
    for side in ('past','future'):
        for runs in ([10],[5,5],[2]*5,[1]*10):
            evaluate(f'continuity_{side}_{len(runs)}runs',continuity_mask(n,runs,side),family='continuity',count=10,location=side,
                     compare_policy=True,extra={'run_lengths':runs,'mean_offset':-25.5 if side=='past' else 25.5})
    # Unconditional frequency-weighted Train patterns, including clean and center-loss.
    for basis in ('record','observed','reliable'):
        for rep in range(config['empirical_replicates']):
            deletion,available=draw_donors(donors,radius,basis,np.random.default_rng(config['seed']+1000+rep))
            if not available.all():raise ValueError('No empirical Train patterns')
            evaluate(f'empirical_{basis}_rep{rep}',deletion,family='empirical_unconditional',location=basis,compare_policy=True)
    # Paired position comparisons use the SAME reference windows and missing count.
    pairs=[]
    for count in config['missing_counts']:
        if count>25:continue
        for near,far in [('center_past_one_run','far_past'),('center_future_one_run','far_future'),
                         ('near_past','far_past'),('near_future','far_future'),('center_band','far_past')]:
            a=f'position_m{count}_{near}_center_kept';b=f'position_m{count}_{far}_center_kept'
            if a not in errors_for_pairing or b not in errors_for_pairing:continue
            for f in FEATURES:
                delta=errors_for_pairing[a][f]-errors_for_pairing[b][f]
                pairs.append(dict(count=count,near=near,far=far,feature=f,
                    continuity_confounded=near=='center_band' and count>1,
                    **paired_block_bootstrap(delta,blocks,config['bootstrap_replicates'],config['seed'])))
    sweep=[]
    for (count,f),(lo,hi,plo,phi) in sweep_ranges.items():
        for group,select in groups.items():
            sweep.append(dict(count=count,feature=f,group=group,n=int(select.sum()),
               best_location_mae=float(lo[select].mean()),worst_location_mae=float(hi[select].mean()),
               per_window_prediction_range_mean=float((phi-plo)[select].mean()),
               per_window_prediction_range_p95=float(np.quantile((phi-plo)[select],.95))))
    pd.DataFrame(summary).to_csv(out/'controlled_feature_errors.csv',index=False)
    pd.DataFrame(conditions).to_csv(out/'controlled_conditions.csv',index=False)
    pd.DataFrame(policy_rows).to_csv(out/'synthetic_policy_comparison.csv',index=False)
    pd.DataFrame(support_rows).to_csv(out/'empirical_support_errors.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'paired_block_comparisons.csv',index=False)
    pd.DataFrame(sweep).to_csv(out/'one_run_location_sensitivity.csv',index=False)
    # Human-readable donor metadata only; packed masks stay local in a separate NPZ.
    donor_summary=[];save={}
    for (r,basis),counts in donors.items():
        keys=list(counts);patterns=np.unpackbits(np.frombuffer(b''.join(keys),np.uint8).reshape(-1,13),axis=1,bitorder='little')[:,:101]
        weights=np.array([counts[k] for k in keys])
        donor_summary.append(dict(radius=r,basis=basis,train_windows=int(weights.sum()),unique_patterns=len(keys),
            missing_count_weighted_hist=np.bincount(patterns.sum(1).astype(int),weights=weights,minlength=102).astype(int).tolist(),
            center_loss_windows=int(weights[patterns[:,50].astype(bool)].sum())))
        save[f'{r}_{basis}_masks']=patterns;save[f'{r}_{basis}_weights']=weights
    np.savez_compressed(out/'train_pattern_library.npz',**save)
    result=dict(reference_windows=n,reference_support_blocks=len(np.unique(blocks)),radius_counts=reference.radius_m.value_counts().to_dict(),
        conditions=len(conditions),computed_conditions=sum(c['status']=='computed' for c in conditions),
        unavailable_conditions=[c['condition'] for c in conditions if c['status']!='computed'],donors=donor_summary,
        frozen_reference_feature_equality=True,test_used_for_experiments=False)
    (out/'experiment_summary.json').write_text(json.dumps(result,indent=2))
    return result
