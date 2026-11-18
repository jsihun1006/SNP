"""Re-evaluate a saved TranSPORTmer holdout with IFAB hard boundary constraints."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_SRC), str(_SRC.parent)]

import argparse
from dataclasses import asdict
import hashlib
import json
import shlex
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from audit_transportmer_deadball_misses import audit_half
import compare_transportmer_folds as run
import dfl_load
import features
from rescore_transportmer_ball_location import metrics
import simulate
from transportmer_ifab_rules import (
    STAGES, Lock, Rules, apply_decoder, build_rules, constrained_path, constraints, crossings,
    observed_kicks, observed_line_out_locks,
)
from config import CONFIG


def intervals(mask, context, variant, source, dest, labels, evidence, conflict):
    first, last = np.asarray(context.frame_id), np.asarray(context.timestamp_s)
    change = np.flatnonzero(np.r_[
        True, (labels[1:] != labels[:-1]) | (evidence[1:] != evidence[:-1])
        | (dest[1:] != dest[:-1]) | (source[1:] != source[:-1]) | (conflict[1:] != conflict[:-1]), True,
    ])
    for a, b in zip(change[:-1], change[1:], strict=True):
        if mask[a]:
            yield {"match_id": context.match_id.iloc[0], "half": int(context.half.iloc[0]),
                   "variant": variant, "start_frame": first[a], "end_frame_inclusive": first[b-1],
                   "start_s": last[a], "end_s_inclusive": last[b-1], "frames": b-a,
                   "direction": "인플레이 → D" if dest[a] else "D → 인플레이",
                   "rule": labels[a], "observed_evidence": evidence[a], "conflict_result": conflict[a]}


def load_automatic(path, match):
    if path is None:
        return {}, None
    data = json.loads(path.read_text())
    if data["match_id"] != match or not data.get("provenance"):
        raise ValueError("Automatic signals require matching match_id and detector provenance.")
    return data["halves"], data["provenance"]


def self_test():
    cfg = Rules()
    pitch = (105., 68.)
    ball = np.tile([20., 2.], (70, 1))
    ball[7:11, 1] = [.9, .3, -.4, -.8]
    ball[11:30, 1] = -.8
    ball[30:36, 1] = [.15, .5, .8, 1.2, 1.5, 1.8]
    players = np.full((70, 12, 2), np.nan)
    players[29,0] = [20, -1]
    movement = {k: np.zeros(70, dtype=bool) for k in ("carry","pass","reliable")}
    movement["speed"] = np.full(70, np.nan)
    marks, _, _, hints = build_rules(ball, players, np.repeat([0,1],6), pitch, np.zeros(70, bool),
                                        movement, {}, cfg)
    locked, _, _ = constraints(marks, 70, 1)
    assert (locked[9:30] == 1).all() and locked[30] == 0 and hints == [(30,0)]
    line_out, _, _ = constraints(observed_line_out_locks(marks), 70, 4)
    assert (line_out[9:30] == 1).all() and line_out[30] == -1
    occluded = ball.copy()
    occluded[8] = np.nan
    assert any(not c.exact and c.frame == 9 and c.previous == 7 for c in crossings(occluded,pitch,cfg))
    with_play = dict(movement, reliable=np.ones(70,bool), **{"pass": np.arange(70) == 28})
    still = ball.copy()
    still[28,1] = .03
    paired, _, rejected, _ = build_rules(still, players, np.repeat([0,1],6), pitch, np.zeros(70,bool),
                                            with_play, {}, cfg)
    assert not any(m.rule == "throw_in_pair" for m in paired) and rejected
    cropped = crossings(ball[11:],pitch,cfg)
    assert not any(not c.inward for c in cropped)
    kick = np.tile([20., 20.], (30, 1))
    kick[12:,0] += np.minimum(np.arange(1,19),4)*.085
    assert [t for t,*_ in observed_kicks(kick,np.tile(kick[:,None,:],(1,1,1)),cfg)] == [12]
    strong = Lock(1,3,True,6,1,"out","observed")
    weak = Lock(1,4,False,3,1,"pass","observed")
    other = Lock(2,3,False,6,1,"restart","observed")
    state, _, blocked = constraints([weak,strong,other],5,1)
    np.testing.assert_array_equal(state, np.array([-1,1,-1,0,-1]))
    assert blocked[2]

    def legacy_cost(log_odds, enter, leave):
        return np.ones(len(log_odds), dtype=bool)

    def legacy(p, game, condition, cases, settings):
        decoded = viterbi_cost(p, np.ones(len(p)), np.ones(len(p)))  # noqa: F821
        decoded[3] = True  # imitate the legacy line-out pass undoing a restart
        return decoded, {}

    legacy = FunctionType(legacy.__code__, dict(globals(), viterbi_cost=legacy_cost))
    final = apply_decoder(legacy,np.ones(5),{}, {},state,[])
    assert final[1] and not final[3]
    dp = constrained_path(np.ones(5),np.ones(5),np.ones(5),state)
    assert dp[1] and not dp[3]
    print("Synthetic checks passed for crossing/re-entry, gaps, short kicks, conflicts, half edges and restart locks.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path, default=run.ROOT / "data/external/dfl")
    parser.add_argument("--device", choices=("mps","cpu","cuda"))
    parser.add_argument("--automatic-signals", type=Path, help="JSON with provenance and per-half source-frame events")
    parser.add_argument("--line-out-only", action="store_true",
                        help="Compare the existing six rules with observed line-out locks only")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.source is None or args.output is None or args.output.exists():
        parser.error("Provide --source and a fresh --output directory.")
    start = perf_counter()
    cfg = Rules()
    stages = {"six_rules": 0, "line_out_only": 4} if args.line_out_only else STAGES
    torch.set_num_threads(4)
    source = json.loads((args.source/"manifest.json").read_text())
    match = source["splits"]["test_match"]
    reference = pd.read_csv(args.source/"summary.csv").query("stage == 'after_rules'").iloc[0]
    weights = args.source/"ball_restorer.pt"
    weight_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    if weight_hash != source["ball_checkpoint"]["sha256"]:
        raise ValueError("Ball-restorer checksum mismatch.")
    device = args.device or source["device"]
    imputer = run.TranSPORTmer("ball", motion_input=source["motion_input"]).to(device)
    imputer.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    imputer.eval()
    prediction_args = SimpleNamespace(device=device, microbatch=source["microbatch"])
    movement, decode, settings = run.load_rules()
    if settings != source["rule_settings"] or list(run.prior.RULES) != source["rules"]:
        raise ValueError("The current six-rule configuration differs from the source run.")
    if source["evaluation_fps"] != cfg.fps:
        raise ValueError("Expected the original 25 fps axis.")
    CONFIG["dfl_cache"] = str(args.cache)
    tracking, events, metadata = dfl_load.load_match(match)
    raw = features.match_from_dfl(tracking, metadata)
    simulated = simulate.simulate(raw, seed=CONFIG["sim_seed"])
    pitch = np.asarray(raw.pitch, dtype=np.float32)
    index = dfl_load.MATCHES.index(match)
    auto, provenance = load_automatic(args.automatic_signals, match)
    scores_ready, changes, proposals, rejected, marks_by_rule = {}, [], [], [], []
    inputs = []
    for half in (1, 2):
        path = args.source/"predictions"/f"{match}_H{half}_restored_none.csv"
        inputs.append(str(path.resolve()))
        data = pd.read_csv(path,float_precision="round_trip",
                           usecols=["match_id","half","frame_id","timestamp_s","dead_probability","after_rules"])
        keep = raw.period == half
        np.testing.assert_array_equal(data.frame_id, raw.frame[keep])
        np.testing.assert_array_equal(data.timestamp_s, tracking.loc[keep,"timestamp_s"])
        np.testing.assert_array_equal(np.diff(data.frame_id), np.ones(len(data)-1))
        if not data.match_id.eq(match).all() or not data.half.eq(half).all() or not data.after_rules.isin([0,1]).all():
            raise ValueError("Unexpected stored prediction labels or identity.")
        rng = np.random.default_rng(source["ball_mask_seed_test"]+2*index+half)
        ball = simulate.ball_dropout(raw.ball[keep]/pitch, rng, source["ball_missing_target"],
                                     source["ball_mask_mean_gap_seconds"])
        players = simulated.players[keep]
        test = run.arrays((players/pitch,ball,raw.team))
        restored = run.complete(imputer,test,prediction_args)
        game = run.correction_game(restored,raw.pitch,movement)
        prob = data.dead_probability.to_numpy()
        base, _ = decode(prob,game,"original",run.prior.RULES,settings)
        np.testing.assert_array_equal(base,data.after_rules.to_numpy(dtype=bool),
                                      err_msg=f"Six-rule baseline mismatch on H{half}")
        observed = ball*pitch
        signal = movement(players,raw.team,observed,raw.pitch)
        frame_lookup = {frame:i for i,frame in enumerate(data.frame_id)}
        auto_half = {key:[frame_lookup[int(frame)] for frame in values]
                     for key,values in auto.get(str(half),{}).items()}
        marks, soft, blocked_pairs, hints = build_rules(observed,players,raw.team,raw.pitch,base,
                                                       signal,auto_half,cfg)
        if args.line_out_only:
            marks = observed_line_out_locks(marks)
            soft, blocked_pairs, hints = [], [], []
        for a,b,name,why in blocked_pairs:
            rejected.append({"half":half,"start_frame":data.frame_id.iloc[a],
                             "end_frame":data.frame_id.iloc[b],"rule":name,"reason":why})
        for mark in marks:
            proposals.append({"half":half,"start_frame":data.frame_id.iloc[mark.start],
                              "end_frame":data.frame_id.iloc[mark.end-1],"forced_dead":mark.dead,
                              "stage":mark.stage,"priority":mark.priority,"rule":mark.rule,"evidence":mark.evidence})
        for frame, kind, delta, rule in soft:
            proposals.append({"half":half,"start_frame":data.frame_id.iloc[frame],
                              "end_frame":data.frame_id.iloc[frame],"stage":4,"rule":rule,
                              "evidence":f"{kind} cost adjustment {delta:.4f}"})
        data["throw_in_team_hint"] = ""
        for frame, team in hints:
            data.loc[frame,"throw_in_team_hint"] = ("H","A")[team]
        masks = {}
        for variant, stage in stages.items():
            state, why, conflicts = constraints(marks,len(data),stage)
            extras = soft if stage == 4 else []
            result = base if stage == 0 else apply_decoder(decode,prob,game,settings,state,extras)
            np.testing.assert_array_equal(result[state>=0], state[state>=0].astype(bool))
            effective = result != base
            labels = np.where(effective, "viterbi_context", "").astype(object)
            evidence = np.where(effective, "sequence decoded with boundary constraints", "").astype(object)
            for i in np.unique(why[why>=0]):
                mask = why == i
                labels[mask], evidence[mask] = marks[i].rule, marks[i].evidence
                marks_by_rule.append({"half":half,"variant":variant,"rule":marks[i].rule,
                                      "locked_frames":int(mask.sum()),"changed_frames":int((mask & effective).sum())})
            for t,kind,delta,rule in extras:
                if effective[t] and why[t] < 0:
                    labels[t], evidence[t] = rule,f"{kind} cost {delta:.4f}"
            statuses = np.where(conflicts,"equal priority, lock deferred",np.where(why>=0,"priority resolved",""))
            masks[variant] = result
            data[variant] = result.astype(np.int8)
            data[f"{variant}_rule"] = labels
            data[f"{variant}_evidence"] = evidence
            data[f"{variant}_conflict"] = statuses
            changes.extend(intervals(effective,data,variant,base,result,labels,evidence,statuses))
        scores_ready[half] = (data, masks, keep)
        print(f"H{half}: reproduced baseline; {len(marks)} lock proposals and {len(soft)} cost adjustments",flush=True)

    # Scores and DFL event labels are consumed only after both halves are decoded.
    rows, stoppages = [], []
    for half,(data,masks,keep) in scores_ready.items():
        truth = run.prior.dfl_labels.dead_label(tracking.loc[keep]).astype(bool)
        data["truth_dead"] = truth.astype(np.int8)
        for variant,state in masks.items():
            score = metrics(truth,state,raw.period[keep])
            audit_score, misses = audit_half(tracking.loc[keep],events.loc[events.period_id==half],state)
            np.testing.assert_allclose(audit_score,score["dead_spell_f1"],atol=1e-12)
            score["missed_throw_in_iou_0_5"] = sum(r["situation"]=="throw_in" and r["miss_iou_0.5"] for r in misses)
            score["inplay_false_dead_seconds"] = score["inplay_false_dead_frames"]/cfg.fps
            rows.append({"half":half,"variant":variant,**score})
            stoppages.extend({"half":half,"variant":variant,**r} for r in misses)
    by_half = pd.DataFrame(rows)
    summary = by_half.drop(columns="half").groupby("variant",sort=False).agg(
        {c:"mean" if c.endswith("_f1") else "sum" for c in by_half if c not in {"half","variant"}})
    for metric in ("dead_spell_f1","alive_spell_f1"):
        np.testing.assert_allclose(summary.loc["six_rules",metric],reference[metric],atol=1e-12)
    if match == "J03WR9":
        assert summary.loc["six_rules","inplay_false_dead_frames"] == 2131
        assert summary.loc["six_rules","missed_dead_iou_0_5"] == 7
    args.output.mkdir(parents=True)
    by_half.to_csv(args.output/"by_half.csv",index=False)
    summary.to_csv(args.output/"summary.csv")
    columns = ["match_id","half","variant","start_frame","end_frame_inclusive","start_s","end_s_inclusive",
               "frames","direction","rule","observed_evidence","conflict_result"]
    pd.DataFrame(changes,columns=columns).to_csv(args.output/"changed_intervals.csv",index=False)
    pd.DataFrame(proposals).to_csv(args.output/"rule_evidence.csv",index=False)
    pd.DataFrame(rejected,columns=["half","start_frame","end_frame","rule","reason"]).to_csv(
        args.output/"rejected_pairs.csv",index=False)
    pd.DataFrame(stoppages).to_csv(args.output/"stoppages.csv",index=False)
    count_cols = ["half","variant","rule","locked_frames","changed_frames"]
    pd.DataFrame(marks_by_rule,columns=count_cols).groupby(count_cols[:3],as_index=False)[count_cols[3:]].sum().to_csv(
        args.output/"by_rule.csv",index=False)
    for half,(data,_,_) in scores_ready.items():
        data.to_csv(args.output/f"{match}_H{half}_predictions.csv",index=False)
    active = sorted({p["rule"] for p in proposals})
    auto_keys = {key for entries in auto.values() for key,values in entries.items() if values}
    inactive = [key for key in ("referee_stop","drop_ground_contact","waiting") if key not in auto_keys]
    manifest = {
        "source":str(args.source.resolve()),"input_predictions":inputs,"test_match":match,
        "fixed_weights":{"ball_restorer":str(weights.resolve()),"sha256":weight_hash,
                         "dead_classifier":str((args.source/"dead_best.pt").resolve())},
        "device":device,"fps":cfg.fps,"seed":CONFIG["sim_seed"],"ball_seed_base":source["ball_mask_seed_test"],
        "thresholds":asdict(cfg),"ifab_edition":"2024/25","six_rule_settings":settings,
        "baseline_frame_and_mask_equality":True,"stages":stages,"rules_with_evidence":active,
        "line_out_only":args.line_out_only,
        "inactive_automatic_signals":inactive,"automatic_signal_provenance":provenance,
        "runtime_s":perf_counter()-start,
        "reproduce":shlex.join([sys.executable,*sys.argv]),
    }
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(summary.to_string())


if __name__ == "__main__":
    main()
