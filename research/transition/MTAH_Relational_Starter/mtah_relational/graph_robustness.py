"""Read-only synthetic tracking interventions, Validation only, frozen targets."""
from dataclasses import replace
import argparse
import json
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
from pathlib import Path
import time

import numpy as np
import torch

from .diagnose_a_training import write_json,restore_variant
from .model_a import load_checkpoint
from .model_b import load_b_checkpoint
from .model_a_data import build_feature_cache,FeatureWindowDataset,sampled_indices
from .graph_windows import VisibleGraphCache,GraphWindowDataset,pack_windows
from .windows import WindowDataset,collate_targets
from .reconstruction import ReconstructionMetrics
from .temporal_cli import open_temporal_dataset
from .corruption import SpanMasker
from .spatial_schema import TARGET_NAMES
from .train_model_a import set_seed,select_device


KINDS={'clean':0,'noise':1,'dropout':2,'permutation':3,'team_swap':4,'composition':5,'physical_shift':6}


def perturb_frame(frame,kind,level,seed,*,active=True):
    """Common random numbers across levels. Never modify source arrays/IDs."""
    if kind=='clean' or not active:return frame
    rng=np.random.default_rng(np.random.SeedSequence([seed,int(frame.frame),KINDS[kind]]))
    xy=frame.xy_m.copy();team=frame.team_codes.copy()
    missing=frame.parse_missing.copy();partial=frame.parse_partial.copy();invalid=frame.parse_invalid.copy()
    seen=np.isfinite(xy).all(axis=1)
    if kind=='noise':xy[seen]+=rng.normal(size=(seen.sum(),2))*level
    elif kind in ('dropout','composition'):
        removed=seen&(rng.random(len(xy))<level)
        xy[removed]=np.nan;missing[removed]=True;partial[removed]=False;invalid[removed]=False
    elif kind=='permutation':
        p=rng.permutation(len(xy));xy=xy[p];team=team[p];missing=missing[p];partial=partial[p];invalid=invalid[p]
    elif kind=='team_swap':team=1-team
    elif kind=='physical_shift':
        home=seen&(team==0);away=seen&(team==1)
        if home.any() and away.any():
            direction=xy[away].mean(0)-xy[home].mean(0)
            xy[home]+=direction/max(np.linalg.norm(direction),1e-12)*level
    else:raise ValueError('Unknown intervention')
    return replace(frame,xy_m=xy,team_codes=team,parse_missing=missing,parse_partial=partial,parse_invalid=invalid)


class PerturbedSource:
    def __init__(self,source,kind,level,seed,*,active_frames=(),forbidden_frames=()):
        self.source,self.kind,self.level,self.seed=source,kind,level,seed
        self.active=frozenset(int(f) for f in active_frames);self.forbidden=frozenset(int(f) for f in forbidden_frames)
    def __getattr__(self,key):return getattr(self.source,key)
    def __len__(self):return len(self.source)
    def frame(self,row):
        f=int(self.source.frame_ids[row])
        if f in self.forbidden:raise AssertionError('A hidden raw observation was accessed during robustness input generation')
        active=self.kind not in ('composition','physical_shift') or f in self.active
        return perturb_frame(self.source.frame(row),self.kind,self.level,self.seed,active=active)


def delta_stats(changed,reference):
    delta=np.linalg.norm(changed-reference,axis=-1)
    spread=float(np.sqrt(np.var(reference,axis=0).sum()))
    cosine=1-np.sum(changed*reference,axis=-1)/np.maximum(np.linalg.norm(changed,axis=-1)*np.linalg.norm(reference,axis=-1),1e-12)
    return dict(mean_l2=float(delta.mean()),p95_l2=float(np.percentile(delta,95)),
                mean_l2_per_reference_std=float(delta.mean()/max(spread,1e-12)),
                mean_cosine_distance=float(np.clip(cosine,0,2).mean()),
                max_absolute=float(np.max(np.abs(changed-reference))))


def scenarios(config):
    cases=[('clean',0.)]
    cases += [('noise',float(x)) for x in config['coordinate_noise_sd_m']]
    cases += [('dropout',float(x)) for x in config['player_dropout_probability']]
    cases += [('permutation',0.),('team_swap',0.),('composition',config['composition_remove_fraction']),
              ('physical_shift',config['physical_team_shift_m'])]
    return cases


def run(run_dir,data_dir,target_cache_path):
    root=Path(run_dir);config=json.loads((root/'robustness_config.json').read_text())
    output=root/'robustness'
    if output.exists():raise FileExistsError('Choose a new robustness output')
    output.mkdir()
    set_seed(2027,4);device,info=select_device('auto')
    base=open_temporal_dataset(data_dir,split='validation')
    base.masker=SpanMasker(seed=2027)
    cache=build_feature_cache(base,target_cache_path)
    full=sampled_indices(base.index,25);positions=full[::config['validation_window_stride']]
    active=[]
    for i in positions:
        routing,_=base._routing(int(i));a,b=config['affected_temporal_span'];active.extend(routing.frame_ids[a:b])
    models={}
    models['A12']=load_checkpoint('data/derived/model_a_run_v2_20261109/best.pt',device)[0]
    models['A20']=restore_variant('data/derived/a_diagnostic_once_20261109/continued_short/best.pt',device)[0]
    models['B12']=load_b_checkpoint(root/'b12_best.pt',device)[0]
    models['B20']=load_b_checkpoint(root/'b20_best.pt',device)[0]
    for model in models.values():model.eval()
    results={};reference={};reference_graph={};reference_team={}
    started=time.perf_counter()
    for kind,level in scenarios(config):
        key=f'{kind}_{level:g}';result={}
        for masked in (False,True):
            if kind=='physical_shift' and masked:continue  # changed physical geometry is not a noise-invariant target
            if masked:base.masker=SpanMasker(seed=2027)
            else:
                from .diagnose_a_representation import NoArtificialMask
                base.masker=NoArtificialMask()
            forbidden=[]
            for i in positions:
                routing,_=base._routing(int(i));forbidden.extend(routing.frame_ids[base._mask(routing)])
            source=PerturbedSource(base.source,kind,level,config['seed'],active_frames=active,forbidden_frames=forbidden)
            altered=WindowDataset(source,base.index,base.normalizer,
                     __import__('mtah_relational.splits',fromlist=['SplitManifest']).SplitManifest.load(Path(data_dir)/'split_manifest.json'),
                     policy=base.policy,masker=base.masker)
            graphs=GraphWindowDataset(altered,VisibleGraphCache(altered),cache)
            clean_accessor=FeatureWindowDataset(base,cache)
            accum={name:[] for name in models};gacc={name:[] for name in ('B12','B20')};tacc={name:[] for name in ('B12','B20')}
            metrics={name:ReconstructionMetrics() for name in models}
            for first in range(0,len(positions),16):
                chosen=positions[first:first+16]
                # Both streams originate from the SAME perturbed visible observations.
                graph_windows=[graphs[int(i)] for i in chosen]
                batch=pack_windows(graph_windows).to(device)
                x=np.stack([altered[int(i)].inputs.dense_features() for i in chosen])
                target=collate_targets([clean_accessor.get_targets(int(i),window=w) for i,w in zip(chosen,graph_windows)]) if masked else None
                with torch.inference_mode():
                    for name,model in models.items():
                        if name.startswith('B'):
                            out=model(batch);prediction=out.prediction;h=out.embedding
                            if not masked:
                                gacc[name].append(out.graph_embedding.cpu().numpy())
                                tacc[name].append(out.team_embedding.cpu().numpy())
                        else:prediction,h=model(torch.as_tensor(x,device=device),batch.padding)
                        if not masked:accum[name].append(h[:,75].cpu().numpy())
                        else:metrics[name].update(prediction.cpu().numpy(),target.values,target.observed,target.quality_weight,target.artificial_mask)
            if masked:
                result['reconstruction']={name:m.report(TARGET_NAMES,base.normalizer.scale[:13]) for name,m in metrics.items()}
            else:
                accum={name:np.concatenate(v) for name,v in accum.items()}
                gacc={name:np.concatenate(v) for name,v in gacc.items()};tacc={name:np.concatenate(v) for name,v in tacc.items()}
                if kind=='clean':reference=accum;reference_graph=gacc;reference_team=tacc
                result['temporal_embedding_shift']={name:delta_stats(h,reference[name]) for name,h in accum.items()}
                result['graph_embedding_shift']={name:delta_stats(h.reshape(-1,64),reference_graph[name].reshape(-1,64)) for name,h in gacc.items()}
                if kind in ('permutation','team_swap'):
                    result['invariance']={}
                    for name in ('B12','B20'):
                        expected=reference_team[name][:,:,::-1,:] if kind=='team_swap' else reference_team[name]
                        result['invariance'][name]=dict(scene_pass=bool(np.allclose(gacc[name],reference_graph[name],atol=5e-5,rtol=1e-4)),
                            temporal_pass=bool(np.allclose(accum[name],reference[name],atol=5e-5,rtol=1e-4)),
                            ordered_team_pass=bool(np.allclose(tacc[name],expected,atol=5e-5,rtol=1e-4)))
        results[key]=result
        write_json(output/'partial.json',results)
        print(f'Robustness complete: {key}',flush=True)
    report=dict(config=config,device=info,validation_windows=len(positions),results=results,seconds=time.perf_counter()-started,
                test_used=False,clean_targets_unchanged=True,original_modified=False,
                caveat='Synthetic input sensitivity on one already-noisy match, not true tracking accuracy or contest accuracy',
                observation_filter='Same existing 2m bounds exclusion after perturbation for A and B; no clipping',
                common_random_numbers=True,hidden_raw_read_guard_passed=True)
    write_json(output/'report.json',report)
    plot_report(root,report)
    return report


def plot_report(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(root)/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for column,(kind,levels) in enumerate((('noise',report['config']['coordinate_noise_sd_m']),
                                         ('dropout',report['config']['player_dropout_probability']))):
        for model in ('A12','A20','B12','B20'):
            values=[report['results'][f'{kind}_{x:g}']['temporal_embedding_shift'][model]['mean_l2_per_reference_std'] for x in levels]
            axes[0,column].plot([0,*levels],[0,*values],marker='o',label=model)
        for model in ('B12','B20'):
            values=[report['results'][f'{kind}_{x:g}']['graph_embedding_shift'][model]['mean_l2_per_reference_std'] for x in levels]
            axes[1,column].plot([0,*levels],[0,*values],marker='o',label=model)
        for row in (0,1):
            axes[row,column].set(xlabel='Coordinate Gaussian SD (m)' if kind=='noise' else 'Player dropout probability',
                ylabel=('Temporal' if row==0 else 'Graph')+' mean L2 shift / reference spread')
            axes[row,column].grid(alpha=.2);axes[row,column].legend()
    fig.suptitle('Validation-only synthetic tracking sensitivity; not tracking/contest accuracy')
    fig.tight_layout();fig.savefig(folder/'tracking_noise_sensitivity.png',dpi=170);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    p.add_argument('--data',default='data/derived/relational_temporal_v1_20261109')
    p.add_argument('--target-cache',default='data/derived/model_a_feature_cache_v1')
    p.add_argument('--plot-only',action='store_true')
    a=p.parse_args()
    if a.plot_only:plot_report(a.run_dir,json.loads((Path(a.run_dir)/'robustness/report.json').read_text()))
    else:run(a.run_dir,a.data,a.target_cache)


if __name__=='__main__':main()
