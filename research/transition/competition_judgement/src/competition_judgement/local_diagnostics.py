"""Paired synthetic observation errors on the original v1 Validation supports."""
import numpy as np
import pandas as pd

from mtah_relational.data_quality import assess_observations
from .quality import POLICY, observation_row
from .detector import compute_evidence, apply_policy
from .diagnostics import validation_supports, perturb
from .local_clusters import connected_mixed_sets
from .local_features import extract_local_frame


def distribution(values):
    a = np.asarray(values,dtype=float)
    a = a[np.isfinite(a)]
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                p05=float(np.quantile(a,.05)) if len(a) else None,
                p95=float(np.quantile(a,.95)) if len(a) else None,
                max=float(a.max()) if len(a) else None)


def deletion_trials(graph, radius):
    """Remove each component member once in ONE frame, not cross-frame tracking.

    Count destruction separately from splitting into >=2 surviving MIXED groups.
    Also count splitting with isolated surviving observations included.
    """
    result = dict(member_deletions=0,trials_original_size_at_least3=0,
                  split_into_multiple_mixed=0,split_including_isolated=0,
                  no_surviving_mixed_component=0)
    for c in graph.components[radius]:
        ids = np.asarray(c.nodes)
        team = graph.team[ids]
        adjacency = (graph.distance_m[np.ix_(ids,ids)] <= radius) & (team[:,None] != team[None,:])
        for removed in range(len(ids)):
            keep = np.arange(len(ids)) != removed
            a = adjacency[np.ix_(keep,keep)]
            pieces = connected_mixed_sets(a)
            all_count = len(pieces)+int((~a.any(axis=1)).sum())
            result["member_deletions"] += 1
            result["trials_original_size_at_least3"] += int(len(ids)>=3)
            result["split_into_multiple_mixed"] += int(len(pieces)>=2)
            result["split_including_isolated"] += int(all_count>=2)
            result["no_surviving_mixed_component"] += int(not pieces)
    return result


def paired_change(clean, changed, names, subset=None):
    if not np.array_equal(clean.frame,changed.frame):
        raise ValueError("Paired diagnostic frame axes differ")
    selected = np.ones(len(clean),bool) if subset is None else np.asarray(subset,dtype=bool)
    result = {}
    for name in names:
        a,b = clean[name].to_numpy(dtype=float),changed[name].to_numpy(dtype=float)
        joint = selected & np.isfinite(a) & np.isfinite(b)
        result[name] = dict(absolute_change=distribution(np.abs(a[joint]-b[joint])),
                            changed_fraction=float((~np.isclose(a[joint],b[joint],atol=1e-12,rtol=0)).mean()) if joint.any() else None,
                            lost_definition=int((selected & np.isfinite(a) & ~np.isfinite(b)).sum()),
                            gained_definition=int((selected & ~np.isfinite(a) & np.isfinite(b)).sum()))
    return result


def raw_edges(frame,pitch,radius):
    xy,_ = assess_observations(frame,pitch,POLICY)
    valid = np.isfinite(xy).all(axis=1)
    d = np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
    return np.triu((frame.team_codes[:,None] != frame.team_codes[None,:]) &
                   valid[:,None] & valid[None,:] & (d<=radius),1)


def run_local_robustness(store,manifest,fitted,config):
    supports = validation_supports(manifest)
    conditions = [("clean",0.),("jitter",.1),("jitter",.5),("jitter",1.),
                  ("dropout",.1),("dropout",.2),("dropout",.3),
                  ("permutation",0.),("team_swap",0.),("composition",.2)]
    clean_table = None
    reports = {}
    deletion = {str(int(r)):dict(member_deletions=0,trials_original_size_at_least3=0,
                  split_into_multiple_mixed=0,split_including_isolated=0,no_surviving_mixed_component=0) for r in (3.,5.)}
    for kind,strength in conditions:
        parts = []
        edge_change = {r:[] for r in (3.,5.)}
        for first,last in supports:
            local,geometry = [],[]
            for offset,frame_id in enumerate(range(first,last+1)):
                original = store.frame(store.position(frame_id))
                frame = perturb(original,kind,strength,seed=config["seed"],active=65<=offset<=85)
                summary,_,graph = extract_local_frame(frame,store.pitch,config)
                local.append(summary)
                geometry.append(observation_row(frame,store.pitch))
                if kind=="clean":
                    for radius in (3.,5.):
                        for name,value in deletion_trials(graph,radius).items():
                            deletion[str(int(radius))][name] += value
                # QA pairing within the same observation only, never person IDs.
                if kind not in ("permutation",):
                    for radius in (3.,5.):
                        before,after=raw_edges(original,store.pitch,radius),raw_edges(frame,store.pitch,radius)
                        union=(before|after).sum()
                        edge_change[radius].append(float((before^after).sum()/union) if union else 0.)
            v1=apply_policy(compute_evidence(pd.DataFrame(geometry),manifest),fitted,5)
            t=pd.DataFrame(local)
            t["v1_score"]=v1.candidate_score.to_numpy()
            parts.append(t)
        table=pd.concat(parts,ignore_index=True)
        if kind=="clean":
            clean_table=table
        shared_strict=np.isfinite(table.v1_score) & np.isfinite(clean_table.v1_score)
        names=[f"{name}_{r}m" for r in (3,5) for name in ("n_local_clusters","max_local_cluster_size",
                 "max_local_contact_strength","n_compact_mixed_clusters","total_unique_players_in_mixed_clusters")]
        reports[f"{kind}_{strength:g}"]=dict(
            frames=len(table),local_available=int(table.local_metric_available.sum()),
            strict_frames=int(table.strict_quality_valid.sum()),v1_score_available=int(np.isfinite(table.v1_score).sum()),
            all_available_features=paired_change(clean_table,table,names),
            shared_strict_scores=paired_change(clean_table,table,["v1_score","max_local_contact_strength_3m",
                                                                 "max_local_contact_strength_5m"],shared_strict),
            symmetric_edge_difference_over_union={str(int(r)):distribution(x) for r,x in edge_change.items()},
            local_availability_lost=int((clean_table.local_metric_available & ~table.local_metric_available).sum()))
    return dict(seed=config["seed"],supports=[list(s) for s in supports],role="validation",n_frames=len(clean_table),
                threshold_refit=False,one_member_deletion=deletion,conditions=reports,
                notes="All changes compare noisy observations, not true contests. Shared-strict support is condition-specific.")
