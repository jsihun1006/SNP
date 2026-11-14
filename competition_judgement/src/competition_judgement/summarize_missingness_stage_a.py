"""Readable aggregate CSV exports from completed Stage A diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .missingness_diagnostics import CAUSES,REGIONS
from .controlled_temporal_masking import FEATURES,diagnostic_estimates


def hist_quantile(hist,q):
    h=np.asarray(hist);return int(np.searchsorted(h.cumsum(),h.sum()*q)) if h.sum() else None


def summarize(run):
    data=json.loads((run/'missingness_summary.json').read_text())['aggregates']
    causes=[];regions=[];overlap=[];run_rows=[]
    for group,g in data.items():
        n=g['windows'];offsets=n*101
        for cause in CAUSES:
            first=np.array(g[cause+'_first_failure_hist']);hist=np.array(g[cause+'_run_length_hist'])
            causes.append(dict(group=group,cause=cause,n_windows=n,failed_windows=g[cause+'_windows'],
                window_fraction=g[cause+'_windows']/n,failed_offsets=g[cause+'_offsets'],offset_fraction=g[cause+'_offsets']/offsets,
                only_cause_windows=g[cause+'_only_windows'],only_cause_offsets=g[cause+'_only_offsets'],
                remove_only_this_factor_full=g[cause+'_remove_factor_full'],
                first_failure_median_offset=(hist_quantile(first,.5)-50) if first.sum() else None,
                run_length_median=hist_quantile(hist,.5),run_length_p95=hist_quantile(hist,.95),
                run_length_max=int(np.flatnonzero(hist)[-1]) if hist.sum() else None))
        for unit in ('window','offset'):
            for code,count in enumerate(g[f'{unit}_failure_code_hist']):
                overlap.append(dict(group=group,unit=unit,code=code,count=count,
                    causes='|'.join(c for bit,c in enumerate(CAUSES) if code&(1<<bit)) or 'none'))
        for basis in ('record','observed','reliable'):
            for name,sl in REGIONS.items():
                prefix=f'{basis}_{name}_missing';hist=np.array(g[prefix+'_run_length_hist']);size=sl.stop-sl.start
                regions.append(dict(group=group,basis=basis,region=name,n_windows=n,
                    valid_ratio=g[f'{basis}_{name}_valid_offsets']/(n*size),
                    windows_with_missing=g[prefix+'_windows'],missing_runs=int(hist.sum()),
                    run_length_median=hist_quantile(hist,.5),run_length_p95=hist_quantile(hist,.95),
                    run_length_max=int(np.flatnonzero(hist)[-1]) if hist.sum() else 0,
                    count_change_ratio=g[f'count_change_{name}_offsets']/(n*size)))
            h=np.array(g[basis+'_missing_run_length_hist']);maximum=np.array(g[basis+'_missing_max_run_hist'])
            run_rows.append(dict(group=group,basis=basis,n_windows=n,missing_runs=int(h.sum()),
                run_length_median=hist_quantile(h,.5),run_length_p95=hist_quantile(h,.95),
                short_run_fraction=h[1:6].sum()/h.sum() if h.sum() else None,
                short_run_offset_fraction=np.dot(np.arange(1,6),h[1:6])/np.dot(np.arange(102),h) if h.sum() else None,
                window_max_run_median=hist_quantile(maximum,.5),window_max_run_p95=hist_quantile(maximum,.95),
                window_max_run_max=int(np.flatnonzero(maximum)[-1]),
                center21_missing_ratio=g[basis+'_center21_missing']/(n*21),
                outer50_missing_ratio=g[basis+'_outer50_missing']/(n*50),
                future_minus_past_missing_mean=g[basis+'_future_minus_past_missing']/n))
    for filename,rows in [('cause_summary.csv',causes),('cause_overlap.csv',overlap),
                          ('temporal_region_summary.csv',regions),('missing_run_summary.csv',run_rows)]:
        target=run/filename
        if target.exists():raise FileExistsError(target)
        pd.DataFrame(rows).to_csv(target,index=False)


def extras(run):
    """Same-side location control and explicit all-reference denominators."""
    config=json.loads((run/'preregistered_diagnostic_config.json').read_text())
    keys=pd.read_csv(run/'reference_keys.csv');n=len(keys);radii=keys.radius_m.to_numpy(int)
    with np.load(run/'reference_temporal_arrays.npz') as archive:series={k:archive[k] for k in archive.files}
    clean=diagnostic_estimates(series,np.ones((n,101),bool),radii)
    rows=[]
    for count in config['missing_counts']:
        for side,base in [('past',0),('future',51)]:
            best={f:np.full(n,np.inf) for f in FEATURES};worst={f:np.zeros(n) for f in FEATURES}
            low={f:np.full(n,np.inf) for f in FEATURES};high={f:np.full(n,-np.inf) for f in FEATURES}
            fixed={f:[] for f in FEATURES}
            for start in range(51-count):
                valid=np.ones((n,101),bool);valid[:,base+start:base+start+count]=False
                est=diagnostic_estimates(series,valid,radii)
                for f in FEATURES:
                    error=np.abs(est[f]-clean[f]);np.minimum(best[f],error,out=best[f]);np.maximum(worst[f],error,out=worst[f])
                    np.minimum(low[f],est[f],out=low[f]);np.maximum(high[f],est[f],out=high[f]);fixed[f].append(float(error.mean()))
            for f in FEATURES:
                location=int(np.argmax(fixed[f]));ranges=high[f]-low[f]
                rows.append(dict(count=count,side=side,feature=f,n=n,placements=51-count,
                    best_fixed_placement_mae=min(fixed[f]),worst_fixed_placement_mae=max(fixed[f]),
                    worst_fixed_start_offset=base+location-50,worst_fixed_end_offset=base+location+count-1-50,
                    per_window_best_location_mae=float(best[f].mean()),per_window_worst_location_mae=float(worst[f].mean()),
                    prediction_range_mean=float(ranges.mean()),prediction_range_p95=float(np.quantile(ranges,.95))))
    pd.DataFrame(rows).to_csv(run/'same_side_location_sensitivity.csv',index=False)
    errors=pd.read_csv(run/'controlled_feature_errors.csv')
    counts={'all':n,'radius=3':int((radii==3).sum()),'radius=5':int((radii==5).sum()),
        'size=2':int((keys.n_players==2).sum()),'size=3+':int((keys.n_players>=3).sum())}
    errors['n_all_references']=errors.group.map(counts)
    errors['center_available_fraction']=errors.n/errors.n_all_references
    errors['diagnostic_calculable_total_fraction']=errors.finite_n/errors.n_all_references
    errors['official_calculable_total_fraction']=(1-errors.official_nan_fraction)*errors.n/errors.n_all_references
    errors.to_csv(run/'controlled_feature_comparison.csv',index=False)
    region_cooccurrence(run)


def region_cooccurrence(run):
    rows=[]
    with np.load(run/'train_pattern_library.npz') as archive:
        for radius in (3,5):
            for basis in ('record','observed','reliable'):
                patterns=archive[f'{radius}_{basis}_masks'];weights=archive[f'{radius}_{basis}_weights']
                code=np.zeros(len(patterns),int)
                for bit,sl in enumerate(REGIONS.values()):code|=patterns[:,sl].any(1).astype(int)<<bit
                for c in range(32):
                    number=int(weights[code==c].sum())
                    rows.append(dict(radius=radius,basis=basis,region_code=c,train_windows=number,
                        fraction=number/int(weights.sum()),regions='|'.join(name for bit,name in enumerate(REGIONS) if c&(1<<bit)) or 'none'))
    path=run/'train_region_cooccurrence.csv'
    if path.exists():raise FileExistsError(path)
    pd.DataFrame(rows).to_csv(path,index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    p.add_argument('--extras-only',action='store_true')
    p.add_argument('--cooccurrence-only',action='store_true')
    a=p.parse_args();run=Path(a.run_dir)
    if a.cooccurrence_only:region_cooccurrence(run);return
    if not a.extras_only:summarize(run)
    for name in ('same_side_location_sensitivity.csv','controlled_feature_comparison.csv'):
        if (run/name).exists():raise FileExistsError(run/name)
    extras(run)


if __name__=='__main__':main()
