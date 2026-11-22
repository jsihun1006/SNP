"""Synthetic only: no licensed files, videos, labels, ball or Smart Tagging."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from games import half_video_blocks
from mtah_relational.data_quality import PlayerFrame, period_of
from mtah_relational.candidate_baseline import CausalContestedIndex
from mtah_relational.features import geometry_features, max_bipartite_cluster_size
from mtah_relational.splits import make_split_manifest
from mtah_relational.zones import zone_metrics
from competition_judgement.quality import observation_row, quality_fields, read_features, validate_table
from competition_judgement.detector import compute_evidence, fit_threshold, classify, apply_policy
from competition_judgement.artifacts import segments, summarize, attach_zones
from competition_judgement.diagnostics import perturb, compare_frames
from competition_judgement.cli import validate_config, verify_preserved, execute

PITCH = (105.3,68.)
BLOCKS = [(1,0,299),(2,320,619)]
MANIFEST = make_split_manifest(BLOCKS)


def frame(i=0, home=8, away=8, offset=1.):
    h = np.array([[20+10*(k%4),15+20*(k//4)] for k in range(home)],float).reshape(-1,2)
    a = np.array([[20+10*(k%4)+offset,15+20*(k//4)] for k in range(away)],float).reshape(-1,2)
    xy = np.concatenate([h,a])
    flags = np.zeros(len(xy),bool)
    return PlayerFrame(i,period_of(i,BLOCKS),xy,np.r_[np.zeros(home,int),np.ones(away,int)],flags,flags.copy(),flags.copy())


def evidence(n=80):
    return compute_evidence(pd.DataFrame([observation_row(frame(i,offset=1+i/100),PITCH) for i in range(n)]),MANIFEST)


@pytest.mark.parametrize("radius",[3,5,10])
def test_distance_inclusive(radius):
    f = geometry_features([[10,10]],[[10+radius,10],[10+radius+.001,10]])
    assert f[f"cross_pairs_{radius}m"] == 1


def test_unique_contact_and_chain_component():
    f = geometry_features([[10,10]],[[11,10],[12,10]])
    assert f["cross_pairs_3m"] == 2 and f["cross_contact_fraction_3m"] == 1
    chain = np.array([[1,0],[1,1],[0,1]],bool)
    assert max_bipartite_cluster_size(chain) == 5
    assert max_bipartite_cluster_size(np.zeros((3,2),bool)) == 0


def test_zone_boundary_does_not_gate_global_pair():
    h,a = np.array([[105.3/8-.1,10]]),np.array([[105.3/8+.1,10]])
    assert geometry_features(h,a)["cross_pairs_3m"] == 1
    z=zone_metrics(np.r_[h,a],np.array([0,1]),PITCH)
    assert sum(x["within_zone_pairs_3m"] for x in z) == 0


def test_baseline_formula_parity_and_temporal():
    e = evidence(50)
    index = CausalContestedIndex()
    for i,row in enumerate(e.to_dict("records")):
        old = index.step(i,1,row,True)
        assert row["baseline_score"] == old["contested_index"]
        assert row["candidate_score"] == old["contested_index"]
        assert row["temporal_available"] == (i>=25)
    r=e.iloc[30]
    spatial=.5*r.cross_contact_fraction_3m+.3*r.cross_cluster_fraction_3m+.2*np.exp(-r.cross_min_m/3)
    assert r.spatial_contact_index == pytest.approx(spatial)


@pytest.mark.parametrize("kind",["permutation","team_swap"])
def test_order_invariance_and_ordered_structure(kind):
    f=frame(home=8,away=9)
    a=observation_row(f,PITCH)
    b=observation_row(perturb(f,kind,0),PITCH)
    for key in ("cross_min_m","cross_density_3m","cross_nearest_mean_m","cross_cluster_fraction_3m"):
        assert a[key] == pytest.approx(b[key])
    for name in ("cx","cy","width","length","radius"):
        t="away" if kind=="team_swap" else "home"
        assert a[f"home_{name}"] == pytest.approx(b[f"{t}_{name}"])
    x=compute_evidence(pd.DataFrame([a]),MANIFEST)
    y=compute_evidence(pd.DataFrame([b]),MANIFEST)
    np.testing.assert_allclose(x.candidate_score,y.candidate_score,equal_nan=True)


@pytest.mark.parametrize("home,away",[(0,0),(0,8),(1,1),(7,8)])
def test_insufficient_is_not_negative(home,away):
    row=observation_row(frame(home=home,away=away),PITCH)
    e=compute_evidence(pd.DataFrame([row]),MANIFEST)
    out=apply_policy(e,{"threshold":.1})
    assert out.candidate_state.iloc[0] == "insufficient_observation"
    assert np.isnan(out.candidate_score.iloc[0])
    assert out.metric_valid.iloc[0] == bool(home and away)
    if not home or not away:
        assert np.isnan(out.cross_pairs_3m.iloc[0])


@pytest.mark.parametrize("kind",["partial","infinite","bounds","duplicate","missing"])
def test_bad_observations_flagged(kind):
    f=frame();xy=f.xy_m.copy()
    flags={k:getattr(f,k).copy() for k in ("parse_missing","parse_partial","parse_invalid")}
    if kind=="partial": xy[0,0]=np.nan;flags["parse_partial"][0]=True
    if kind=="infinite": xy[0,0]=np.inf;flags["parse_invalid"][0]=True
    if kind=="bounds": xy[0,0]=-3
    if kind=="duplicate": xy[0]=xy[1]
    if kind=="missing": xy[0]=np.nan;flags["parse_missing"][0]=True
    row=observation_row(replace(f,xy_m=xy,**flags),PITCH)
    q=quality_fields(row)
    assert not q["strict_quality_valid"] and q["metric_valid"]
    assert q["quality_tier"]=="low"
    assert np.array_equal(f.xy_m,frame().xy_m)


def test_train_only_fit_unique_frames_and_holdout_mutation():
    e=evidence(300)
    fitted=fit_threshold(e)
    changed=e.copy()
    changed.loc[changed.split!="train","candidate_score"]=1e6
    assert fit_threshold(changed)==fitted
    with pytest.raises(ValueError): fit_threshold(pd.concat([e,e]))


def test_causal_gate_has_delay_and_no_backdating():
    e=evidence(20)
    out=apply_policy(e,{"threshold":0},5)
    assert out.candidate_state.iloc[:4].eq("not_candidate").all()
    assert out.candidate_state.iloc[4:].eq("candidate").all()
    assert out.candidate_pending.iloc[:4].all()
    shorter=apply_policy(e.iloc[:3],{"threshold":0},5)
    assert np.array_equal(shorter.candidate_state,out.candidate_state.iloc[:3])


def test_future_changes_do_not_affect_past_with_frozen_policy():
    table=pd.DataFrame([observation_row(frame(i),PITCH) for i in range(70)])
    changed=table.copy()
    changed.loc[50:,"cross_min_m"]=50
    a=apply_policy(compute_evidence(table,MANIFEST),{"threshold":.5})
    b=apply_policy(compute_evidence(changed,MANIFEST),{"threshold":.5})
    np.testing.assert_equal(a.candidate_score.iloc[:50].to_numpy(),b.candidate_score.iloc[:50].to_numpy())
    assert np.array_equal(a.candidate_state.iloc[:50],b.candidate_state.iloc[:50])


def test_quality_dropout_resets_history_and_run():
    table=pd.DataFrame([observation_row(frame(i,home=7 if i==30 else 8),PITCH) for i in range(70)])
    out=apply_policy(compute_evidence(table,MANIFEST),{"threshold":0})
    assert out.candidate_state.iloc[30]=="insufficient_observation"
    assert out.candidate_state.iloc[31:35].eq("not_candidate").all()
    assert not out.temporal_available.iloc[31:56].any()
    assert out.temporal_available.iloc[56]


def test_half_gap_split_and_period_zero():
    ids=list(range(170,240))+list(range(290,350))
    e=compute_evidence(pd.DataFrame([observation_row(frame(i),PITCH) for i in ids]),MANIFEST)
    assert not e.loc[e.frame.isin([210,240,290,320]),"temporal_available"].any()
    assert e.loc[e.frame.between(300,319),"candidate_score"].isna().all()
    out=apply_policy(e,{"threshold":0})
    s=segments(out)
    assert not any((r.first_frame<300 and r.last_frame>=320) for r in s.itertuples())
    # Split edge: original baseline remains available while routed one restarts.
    assert e.loc[e.frame==210,"baseline_temporal_available"].all()
    assert not e.loc[e.frame==210,"temporal_available"].any()


def test_full_alignment_contract_and_malformed_rows():
    n=199319
    row=observation_row(frame(),PITCH)
    table=pd.DataFrame([row]).loc[np.zeros(n,int)].reset_index(drop=True)
    table["frame"]=np.arange(n)
    blocks=half_video_blocks("BAR-ATH")
    table["period"]=[period_of(i,blocks) for i in range(n)]
    validate_table(table,n,blocks)
    table.loc[100,"frame"]=99
    with pytest.raises(ValueError):validate_table(table,n,blocks)


def test_allowlist_excludes_ball_tag_gt_and_future_delta(tmp_path):
    table=pd.DataFrame([observation_row(frame(i),PITCH) for i in range(10)])
    for key in ("ball_x","smart_tagging","gt_label","foo_delta_next_1s"):
        table[key]="decoy"
    path=tmp_path/"features.csv";table.to_csv(path,index=False)
    a=read_features(path,10,[(1,0,9)])
    table["ball_x"]=999;table["gt_label"]="H"
    table.to_csv(path,index=False)
    b=read_features(path,10,[(1,0,9)])
    pd.testing.assert_frame_equal(a,b)
    assert set(a).isdisjoint({"ball_x","smart_tagging","gt_label","foo_delta_next_1s"})


def test_seeded_perturbations_and_nested_dropout():
    f=frame();a=perturb(f,"jitter",.1);b=perturb(f,"jitter",1.)
    np.testing.assert_allclose((a.xy_m-f.xy_m)*10,b.xy_m-f.xy_m)
    a=perturb(f,"dropout",.1);b=perturb(f,"dropout",.3)
    assert (np.isnan(a.xy_m).all(1) <= np.isnan(b.xy_m).all(1)).all()
    np.testing.assert_equal(b.xy_m,perturb(f,"dropout",.3).xy_m)
    assert np.isfinite(f.xy_m).all()


def test_segments_summary_noise_comparison_and_csv_roundtrip(tmp_path):
    e=evidence()
    a=apply_policy(e,fit_threshold(e));s=segments(a)
    assert s.n_frames.sum()==len(a)
    assert compare_frames(a,a)["score_abs_change_p95"]==0
    assert summarize(a)["states"]["candidate"]>=0
    a.to_csv(tmp_path/"candidate_frames.csv",index=False)
    b=pd.read_csv(tmp_path/"candidate_frames.csv")
    assert set(b.candidate_state).issubset({"candidate","not_candidate","insufficient_observation"})
    assert "label" not in b


def test_zone_samples_are_not_forward_filled(tmp_path):
    e=evidence(10);f=frame()
    z=pd.DataFrame([dict(frame=0,period=1,**v) for v in zone_metrics(f.xy_m,f.team_codes,PITCH)])
    path=tmp_path/"zones.csv";z.to_csv(path,index=False)
    out=attach_zones(e,path)
    assert out.zone_sample_available.sum()==1
    assert out.zone_max_mixing.iloc[1:].isna().all()


def test_config_and_preservation_checks(tmp_path):
    config=json.loads((Path(__file__).parents[1]/"configs/initial_policy.json").read_text())
    validate_config(config)
    config["candidate_rule"]["bridge_unknown_gap_frames"]=1
    with pytest.raises(ValueError):validate_config(config)
    p=tmp_path/"data";p.write_text("original")
    snap=dict(files={},artifacts={str(p):[p.stat().st_size,p.stat().st_mtime_ns]})
    verify_preserved(snap)
    p.write_text("changed")
    with pytest.raises(RuntimeError):verify_preserved(snap)


def test_output_overwrite_rejected_before_reading_inputs(tmp_path):
    out=tmp_path/"data/derived/competition_judgement/existing"
    out.mkdir(parents=True)
    f=out/"keep.txt";f.write_text("keep")
    with pytest.raises(FileExistsError):
        execute(root=tmp_path,output=out,features="absent",cache="absent",
                split_path="absent",config_path="absent")
    assert f.read_text()=="keep"


def test_deterministic_state_pipeline_and_signed_departure():
    table=pd.DataFrame([observation_row(frame(i,offset=1+i/100),PITCH) for i in range(80)])
    a=compute_evidence(table,MANIFEST);b=compute_evidence(table,MANIFEST)
    fa,fb=fit_threshold(a),fit_threshold(b)
    assert fa==fb
    pd.testing.assert_frame_equal(apply_policy(a,fa),apply_policy(b,fb))
    assert a.nearest_approach_m_per_s.iloc[30] < 0
    assert a.home_cx_delta_prev_1s.iloc[30] == 0


def test_team_shape_invalidity_and_unknown_metrics_remain_nan():
    row=observation_row(frame(home=1,away=1),PITCH)
    e=compute_evidence(pd.DataFrame([row]),MANIFEST)
    assert not e.valid_home_width.iloc[0]
    assert np.isnan(e.home_width.iloc[0])
    assert e.valid_home_cx.iloc[0]
    row=observation_row(frame(home=0,away=1),PITCH)
    e=compute_evidence(pd.DataFrame([row]),MANIFEST)
    assert np.isnan(e.max_cross_cluster_3m.iloc[0]) and not e.valid_max_cross_cluster_3m.iloc[0]
