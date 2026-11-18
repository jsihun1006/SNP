"""Enumerate every error in the saved 5 Hz versus 5+25 Hz DFL holdout."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from transportmer_ha_data import contiguous_runs
from evaluation import possession_metrics as scores

LABEL = {-1: "미확인", 0: "H", 1: "A", 2: "D", 3: "X"}
MODELS = {"5fps": "base", "5+25fps": "refined"}
CATEGORIES = {
    "missed_transition": "누락 전환",
    "spurious_transition": "가짜 전환",
    "timing_error": "매칭 전환 시점 차이",
    "missed_spell": "점유 구간 미검출",
    "spurious_spell": "가짜·매칭 실패 예측 점유 구간",
    "short_missed_spell": "2초 미만 점유 미검출",
    "wrong_run": "연속 오답 구간",
    "label_pair_error": "정답·예측 쌍별 오답",
}


def clock(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes):02d}:{seconds:05.2f}"


def compact_sequence(values):
    if not len(values):
        return ""
    changed = np.r_[True, values[1:] != values[:-1]]
    return "→".join(LABEL[int(x)] for x in values[changed])


def save_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


class Audit:
    def __init__(self, args):
        self.args = args
        self.rows = []
        self.counter = Counter()
        self.transitions = []
        self.comparison = []
        self.confusion = []
        self.summary = []
        self.halves = []
        self.truth_spells = []
        self.predicted_spells = []
        events = pd.read_parquet(args.events) if args.events.is_file() else pd.DataFrame()
        tracking = pd.read_parquet(args.cache / "J03WR9_tracking.parquet").set_index("frame_id", verify_integrity=True)
        roster = pd.read_parquet(args.cache / "J03WR9_meta.parquet")
        player_team = dict(zip(roster.player_id, roster.team))
        player_ids = [p for p in player_team if p + "_x" in tracking]
        pitch = roster[["pitch_length", "pitch_width"]].iloc[0].to_numpy()
        for half in (1, 2):
            table = pd.read_parquet(args.run / f"estimated_D_primary_J03WR9_H{half}.parquet")
            frame = table.frame.to_numpy()
            raw = tracking.reindex(frame)
            np.testing.assert_allclose(raw.timestamp_s, table.seconds, atol=1e-6, rtol=0)
            np.testing.assert_array_equal(raw.period, table.half)
            labels = raw.ball_owning_team.map({"home": 0, "away": 1}).fillna(-1).to_numpy(int)
            labels[raw.ball_state.to_numpy() != 1] = 2
            np.testing.assert_array_equal(labels, table.truth)
            ball = raw[["ball_x", "ball_y"]].to_numpy(float) * pitch
            players = raw[[p + "_" + axis for p in player_ids for axis in ("x", "y")]]
            players = players.to_numpy().reshape(len(raw), len(player_ids), 2) * pitch
            team = np.asarray([player_team[p] for p in player_ids])
            near = []
            for code in ("home", "away"):
                distance = np.linalg.norm(players[:, team == code] - ball[:, None], axis=-1)
                near.append(np.any(distance <= 3, axis=1))
            contest = near[0] & near[1] & np.isfinite(ball).all(axis=1)
            restart = np.zeros(len(table), bool)
            for a, b in contiguous_runs(np.isin(labels, (0, 1)), frame):
                if a and labels[a - 1] == 2:
                    restart[a:min(a + 50, b)] = True
            self.halves.append({
                "half": half, "table": table, "frame": frame, "time": table.seconds.to_numpy(),
                "truth": labels, "base": table.base.to_numpy(), "refined": table.refined.to_numpy(),
                "candidate": table.candidate.to_numpy(bool), "contest": contest,
                "ball_missing": ~np.isfinite(ball).all(axis=1), "restart": restart,
                "events": events.loc[events["period_id"].eq(half) & events["frame_id"].notna()]
                          .sort_values("frame_id") if len(events) else pd.DataFrame(),
            })

    def location(self, half, a, b, pred):
        n = len(half["frame"])
        ca, cb = max(0, a - 50), min(n, b + 50)
        table = half["table"]
        truth = half["truth"]
        mask_error = ((truth == 2) != (half["base"] == 2))
        fraction = float(half["candidate"][a:b].mean())
        tags = ["25fps 대상 안" if fraction == 1 else "25fps 대상 밖" if fraction == 0 else "25fps 대상 경계"]
        if half["contest"][ca:cb].any():
            tags.append("주변 양 팀이 볼 3m 이내인 경합 후보")
        if half["ball_missing"][ca:cb].any():
            tags.append("주변 볼 결측")
        if half["restart"][ca:cb].any():
            tags.append("주변 재개 후 2초")
        if mask_error[ca:cb].any():
            tags.append("주변 데드볼 마스크 오류")
        events = half["events"]
        nearby = events.loc[events["frame_id"].between(half["frame"][a] - 25, half["frame"][b - 1] + 25)] if len(events) else events
        event_types = sorted(nearby["spadl_type"].dropna().unique()) if len(nearby) else []
        event_frames = ",".join(str(int(x)) for x in nearby["frame_id"].to_numpy()) if len(nearby) else ""
        return {
            "match": "J03WR9", "half": half["half"], "start_row": a, "stop_row_exclusive": b,
            "start_frame": int(half["frame"][a]), "end_frame_inclusive": int(half["frame"][b - 1]),
            "start_seconds": float(half["time"][a]), "end_seconds_inclusive": float(half["time"][b - 1]),
            "half_start_clock": clock(half["time"][a]), "half_end_clock": clock(half["time"][b - 1]),
            "duration_seconds": (b - a) / 25, "frame_count": b - a,
            "truth_sequence": compact_sequence(truth[a:b]), "prediction_sequence": compact_sequence(pred[a:b]),
            "context_start_frame": int(half["frame"][ca]), "context_end_frame": int(half["frame"][cb - 1]),
            "context_truth": compact_sequence(truth[ca:cb]), "context_prediction": compact_sequence(pred[ca:cb]),
            "refinement_candidate_fraction": fraction,
            "ball_missing_frames": int(half["ball_missing"][a:b].sum()),
            "geometry_contest_frames": int(half["contest"][a:b].sum()),
            "context_tags": " | ".join(tags),
            "nearby_DFL_event_types": " | ".join(event_types) or "미확정",
            "nearby_DFL_event_frames": event_frames,
        }

    def record(self, half, model, category, subtype, a, b, **fields):
        column = MODELS[model]
        self.counter[model, category] += 1
        identifier = f"{model}-{category}-{self.counter[model, category]:03d}"
        row = {"id": identifier, "model": model, "category": category,
               "category_ko": CATEGORIES[category], "subtype": subtype,
               **self.location(half, a, b, half[column]), **fields}
        self.rows.append(row)
        return row

    def audit_transitions(self, half, model):
        truth = half["truth"]
        pred = half[MODELS[model]]
        matched_all, missed_all, spurious_all = [], [], []
        for a, b in contiguous_runs(np.isin(truth, (0, 1)), half["frame"]):
            gt = scores.transitions(truth, [(a, b)])
            observed = [t for t in scores.transitions(pred, [(a, b)]) if {t[1], t[2]} == {0, 1}]
            matched, missed, spurious = scores.match_transitions(gt, observed, 25, typed="pair")
            matched_all.extend(matched)
            missed_all.extend(missed)
            spurious_all.extend(spurious)
            pairs = {g: p for g, p in matched}
            for event in gt:
                t, old, new = event
                p = pairs.get(event)
                next_stop = next((x[0] for x in gt if x[0] > t), b)
                entry = {"model": model, "match": "J03WR9", "half": half["half"],
                         "truth_row": t, "truth_frame": int(half["frame"][t]),
                         "truth_seconds": float(half["time"][t]), "truth_half_clock": clock(half["time"][t]),
                         "direction": f"{LABEL[old]}→{LABEL[new]}", "matched": p is not None,
                         "prediction_frame": int(half["frame"][p[0]]) if p else None,
                         "signed_delay_ms": (p[0] - t) * 40 if p else None,
                         "new_truth_spell_seconds": (next_stop - t) / 25,
                         "prediction_at_true_change": LABEL[int(pred[t])],
                         "refinement_candidate": bool(half["candidate"][t])}
                self.transitions.append(entry)
                if p and p[0] != t:
                    delay = (p[0] - t) * 40
                    self.record(half, model, "timing_error", "지연" if delay > 0 else "선행",
                                min(t, p[0]), max(t, p[0]) + 1,
                                truth_frame=int(half["frame"][t]), prediction_frame=int(half["frame"][p[0]]),
                                truth_half_clock=clock(half["time"][t]),
                                prediction_half_clock=clock(half["time"][p[0]]),
                                direction=entry["direction"], signed_delay_ms=delay,
                                absolute_delay_ms=abs(delay))
            for t, old, new in missed:
                proposals = [p for p in observed if p[1:] == (old, new)]
                nearest = min(proposals, key=lambda p: abs(p[0] - t)) if proposals else None
                if pred[t] == 2:
                    subtype = "예측 D가 실제 전환을 가림"
                elif nearest is None or abs(nearest[0] - t) > 125:
                    subtype = "전후 5초에 같은 방향 전환 없음"
                elif abs(nearest[0] - t) > 25:
                    subtype = "같은 방향 전환이 있으나 시점 차이 1초 초과"
                else:
                    subtype = "근접 후보가 다른 정답 전환에 매칭됨"
                next_stop = next((g[0] for g in gt if g[0] > t), b)
                self.record(half, model, "missed_transition", subtype, t, t + 1,
                            direction=f"{LABEL[old]}→{LABEL[new]}",
                            nearest_same_direction_prediction_frame=int(half["frame"][nearest[0]]) if nearest else None,
                            nearest_prediction_delay_ms=(nearest[0] - t) * 40 if nearest else None,
                            new_truth_spell_seconds=(next_stop - t) / 25,
                            prediction_at_change=LABEL[int(pred[t])])
            for t, old, new in spurious:
                nearest = min(gt, key=lambda g: abs(g[0] - t)) if gt else None
                typed = [g for g in gt if g[1:] == (old, new)]
                same = min(typed, key=lambda g: abs(g[0] - t)) if typed else None
                if nearest and abs(nearest[0] - t) <= 25 and nearest[1:] != (old, new):
                    subtype = "근접 정답 전환과 방향이 다름"
                elif same and abs(same[0] - t) <= 25:
                    subtype = "같은 방향 정답에 중복 전환"
                elif same and abs(same[0] - t) <= 125:
                    subtype = "같은 방향 정답에서 1초 초과 시점 이동"
                else:
                    subtype = "전후 5초에 같은 방향 정답 전환 없음"
                self.record(half, model, "spurious_transition", subtype, t, t + 1,
                            direction=f"{LABEL[old]}→{LABEL[new]}",
                            nearest_truth_frame=int(half["frame"][nearest[0]]) if nearest else None,
                            nearest_same_direction_truth_frame=int(half["frame"][same[0]]) if same else None,
                            delay_from_nearest_same_direction_ms=(t - same[0]) * 40 if same else None)
        return matched_all, missed_all, spurious_all

    def audit_spells(self, half, model):
        truth, pred = half["truth"], half[MODELS[model]]
        runs = contiguous_runs(truth >= 0, half["frame"])
        actual = scores.possession_spells(truth, runs)
        estimated = scores.possession_spells(pred, runs)
        overlaps = scores.spell_overlaps(actual, estimated)
        flags = {threshold: scores.matched_gt_flags(overlaps, len(actual), threshold)
                 for threshold in scores.SPELL_IOUS}
        reverse = sorted([(iou, p, g) for iou, g, p in overlaps],
                         key=lambda item: (-item[0], item[2], item[1]))
        predicted_flags = {threshold: scores.matched_gt_flags(reverse, len(estimated), threshold)
                           for threshold in scores.SPELL_IOUS}
        for index, (a, b, team) in enumerate(actual):
            self.truth_spells.append({"model": model, "truth_spell_index": index,
                                      **self.location(half, a, b, pred),
                                      **{f"matched_iou_{t:.1f}": bool(flags[t][index]) for t in flags}})
            if flags[.5][index]:
                continue
            same = [(iou, p) for iou, g, p in overlaps if g == index]
            best_iou, best_pred = max(same, default=(0., -1))
            inside = pred[a:b]
            pieces = [(x, y, k) for x, y, k in estimated if k == team and x < b and y > a]
            if not (inside == team).any():
                subtype = "실제 점유팀 프레임을 전부 놓침"
            elif len(pieces) > 1:
                subtype = "실제 점유를 여러 조각으로 분할"
            elif best_pred >= 0 and (estimated[best_pred][1] - estimated[best_pred][0]) > b - a:
                subtype = "예측 점유가 이웃 구간까지 과도하게 확장"
            else:
                subtype = "경계 불일치 또는 구간 과소 검출"
            fields = {
                "truth_team": LABEL[team], "best_same_team_iou": best_iou,
                "correct_team_fraction": float((inside == team).mean()),
                "predicted_D_fraction": float((inside == 2).mean()),
                "same_team_predicted_fragments": len(pieces),
                "best_prediction_start_frame": int(half["frame"][estimated[best_pred][0]]) if best_pred >= 0 else None,
                "best_prediction_end_frame": int(half["frame"][estimated[best_pred][1] - 1]) if best_pred >= 0 else None,
                **{f"matched_iou_{threshold:.1f}": bool(flags[threshold][index]) for threshold in flags},
            }
            row = self.record(half, model, "missed_spell", subtype, a, b, **fields)
            if b - a < 50:
                self.record(half, model, "short_missed_spell", subtype, a, b,
                            parent_spell_id=row["id"], **fields)
        for index, (a, b, team) in enumerate(estimated):
            same = [(iou, g) for iou, g, p in overlaps if p == index]
            best_iou, best_gt = max(same, default=(0., -1))
            fields = {"predicted_team": LABEL[team], "best_same_team_iou": best_iou,
                      "best_truth_start_frame": int(half["frame"][actual[best_gt][0]]) if best_gt >= 0 else None,
                      "best_truth_end_frame": int(half["frame"][actual[best_gt][1] - 1]) if best_gt >= 0 else None,
                      **{f"matched_iou_{t:.1f}": bool(predicted_flags[t][index]) for t in predicted_flags}}
            self.predicted_spells.append({"model": model, "predicted_spell_index": index,
                                          **self.location(half, a, b, pred), **fields})
            if predicted_flags[.5][index]:
                continue
            if not (truth[a:b] == team).any():
                kind = "정답에 해당 팀 점유가 없는 예측 구간"
            elif best_gt >= 0 and (b - a) > actual[best_gt][1] - actual[best_gt][0]:
                kind = "이웃 점유까지 합친 과도한 확장"
            else:
                kind = "짧게 분할되거나 경계가 어긋난 예측 구간"
            self.record(half, model, "spurious_spell", kind, a, b, **fields)
        short = [b - a < 50 for a, b, _ in actual]
        return len(actual), len(estimated), int(sum(short)), int(np.count_nonzero(flags[.5] & short))

    def audit_frames(self, half, model):
        truth, pred = half["truth"], half[MODELS[model]]
        known = truth >= 0
        wrong = known & (truth != pred)
        parents = []
        for a, b in contiguous_runs(wrong, half["frame"]):
            pair = (truth[a:b], pred[a:b])
            if np.all(pair[0] == 2):
                kind = "실제 D를 H/A로 오분류"
            elif np.all(pair[1] == 2):
                kind = "실제 H/A를 D로 오분류"
            elif np.isin(pair[0], (0, 1)).all() and np.isin(pair[1], (0, 1)).all():
                kind = "H/A 팀 오분류"
            else:
                kind = "팀 오분류와 D 오류 혼합"
            parent = self.record(half, model, "wrong_run", kind, a, b)
            parents.append(parent)
            cuts = np.r_[a, np.flatnonzero((np.diff(truth[a:b]) != 0) | (np.diff(pred[a:b]) != 0)) + a + 1, b]
            for start, end in zip(cuts[:-1], cuts[1:], strict=True):
                self.record(half, model, "label_pair_error", f"{LABEL[int(truth[start])]}→{LABEL[int(pred[start])]}",
                            int(start), int(end), parent_wrong_run_id=parent["id"])
        for gt in (0, 1, 2):
            for estimate in (0, 1, 2):
                mask = (truth == gt) & (pred == estimate)
                self.confusion.append({"model": model, "half": half["half"], "truth": LABEL[gt],
                                       "prediction": LABEL[estimate], "frames": int(mask.sum()),
                                       "seconds": float(mask.sum() / 25), "wrong": gt != estimate})
        return parents

    def compare_models(self):
        for half in self.halves:
            truth, base, refined = half["truth"], half["base"], half["refined"]
            valid = truth >= 0
            masks = {
                "5fps 오답을 25fps가 해결": valid & (base != truth) & (refined == truth),
                "5fps 정답을 25fps가 오답으로 변경": valid & (base == truth) & (refined != truth),
                "둘 다 틀렸고 예측도 변경": valid & (base != truth) & (refined != truth) & (base != refined),
                "둘 다 같은 오답 유지": valid & (base != truth) & (refined != truth) & (base == refined),
            }
            for kind, mask in masks.items():
                for a, b in contiguous_runs(mask, half["frame"]):
                    self.comparison.append({"kind": kind, **self.location(half, a, b, refined),
                                            "base_sequence": compact_sequence(base[a:b]),
                                            "refined_sequence": compact_sequence(refined[a:b])})

    def execute(self):
        for model in MODELS:
            for half in self.halves:
                matched, missed, spurious = self.audit_transitions(half, model)
                ngt, npred, short, short_matched = self.audit_spells(half, model)
                wrong = self.audit_frames(half, model)
                summary = {"model": model, "half": half["half"], "truth_transitions": len(matched) + len(missed),
                           "matched_transitions": len(matched), "exact_transitions": sum(g[0] == p[0] for g, p in matched),
                           "timing_errors": sum(g[0] != p[0] for g, p in matched),
                           "missed_transitions": len(missed), "spurious_transitions": len(spurious),
                           "truth_spells": ngt, "predicted_spells": npred, "short_spells": short,
                           "short_missed": short - short_matched, "wrong_runs": len(wrong),
                           "wrong_frames": sum(r["frame_count"] for r in wrong)}
                self.summary.append(summary)
        self.compare_models()
        out = self.args.output
        out.mkdir(parents=True, exist_ok=True)
        save_csv(out / "all_errors.csv", self.rows)
        save_csv(out / "summary_by_half.csv", self.summary)
        save_csv(out / "all_true_transitions.csv", self.transitions)
        save_csv(out / "refinement_changed_intervals.csv", self.comparison)
        save_csv(out / "frame_confusion.csv", self.confusion)
        save_csv(out / "all_truth_spells.csv", self.truth_spells)
        save_csv(out / "all_predicted_spells.csv", self.predicted_spells)
        for category in CATEGORIES:
            save_csv(out / f"{category}.csv", [r for r in self.rows if r["category"] == category])
        cross = pd.DataFrame(self.transitions).pivot(index=["match", "half", "truth_row", "truth_frame",
                                                          "truth_seconds", "truth_half_clock", "direction"],
                                                     columns="model", values="matched").reset_index()
        cross["comparison"] = np.select(
            [cross["5fps"] & ~cross["5+25fps"], ~cross["5fps"] & cross["5+25fps"],
             cross["5fps"] & cross["5+25fps"]],
            ["5fps 검출·5+25fps 누락", "5fps 누락·5+25fps 검출", "둘 다 검출"], default="둘 다 누락")
        cross.to_csv(out / "transition_comparison.csv", index=False, encoding="utf-8-sig")
        for name, label in (("lost_by_refinement", "5fps 검출·5+25fps 누락"),
                            ("recovered_by_refinement", "5fps 누락·5+25fps 검출"),
                            ("missed_by_both", "둘 다 누락")):
            cross.loc[cross.comparison == label].to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
        for model in MODELS:
            folder = out / ("5fps" if model == "5fps" else "5plus25fps")
            folder.mkdir(exist_ok=True)
            for category in CATEGORIES:
                save_csv(folder / f"{category}.csv", [r for r in self.rows
                                                    if r["model"] == model and r["category"] == category])
        type_counts = (pd.DataFrame(self.rows).groupby(["model", "category", "subtype"]).size()
                       .reset_index(name="count"))
        type_counts.to_csv(out / "error_subtype_counts.csv", index=False, encoding="utf-8-sig")
        (type_counts.pivot(index=["category", "subtype"], columns="model", values="count")
         .fillna(0).astype(int).reset_index()
         .to_csv(out / "error_subtype_comparison.csv", index=False, encoding="utf-8-sig"))
        expected = json.loads((self.args.run / "holdout.json").read_text())["estimated_D_primary"]
        for model, field in MODELS.items():
            key = "base" if field == "base" else "refined"
            group = [r for r in self.summary if r["model"] == model]
            for a, b in (("missed_transitions", "missed_transitions"), ("spurious_transitions", "spurious_transitions"),
                         ("matched_transitions", "matched_transitions"), ("short_missed", None)):
                total = sum(r[a] for r in group)
                target = expected[key][b] if b else expected[key]["short_spells"] - expected[key]["short_matched"]
                if total != target:
                    raise AssertionError(f"{model} {a}: {total} != saved {target}")
        self.write_markdown(cross)
        print(pd.DataFrame(self.summary).to_string(index=False))
        print("Cross-model transitions", cross.comparison.value_counts().to_dict())
        print("Saved", out)

    def write_markdown(self, cross):
        out = self.args.output
        summary = pd.DataFrame(self.summary).groupby("model").sum(numeric_only=True)
        manifest_path = self.args.run / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        run_description = ("5fps를 고정하고 후보 구간만 25fps로 학습한 결과" if manifest.get("candidate_only_loss")
                           else "5fps base와 5+25fps refined 비교 결과")
        source = self.args.run.relative_to(ROOT) if self.args.run.is_relative_to(ROOT) else self.args.run
        header = [
            "# 5fps와 5+25fps 전체 오류 목록", "",
            f"분석 원본은 `{source}`다. J03WR9 전후반, 동일한 예측 D 마스크를 사용한 {run_description}다. 별도의 5fps 사후 보정 실험은 포함하지 않았다.",
            "", "시간은 전반·후반 각각 00:00.00부터 센 DFL 하프 경과 시간이다. 원본 frame_id도 각 항목에 적었다. 구간 끝 프레임은 포함한다.",
            "", "전환은 실제 H/A 구간 안의 H↔A만 센다. 방향이 같은 전환을 ±25프레임(1초)에서 일대일로 매칭한다. 1초 밖의 이동은 누락과 가짜 전환으로 각각 집계될 수 있다.",
            "", "시점 차이는 매칭된 전환 중 0ms가 아닌 사례다. 점유 미검출은 저장소 평가 코드의 일대일 구간 매칭에서 IoU 0.5로 매칭되지 않은 정답 점유다. 가짜·매칭 실패 점유는 같은 기준에서 매칭되지 않은 예측 점유다. 2초 미만 점유 목록은 미검출 점유 중 50프레임 미만인 부분집합이다. 전체 Spell F1은 IoU 0.3~0.7 평균이며 all_truth_spells.csv와 all_predicted_spells.csv에는 모든 구간의 각 기준별 검출 여부를 기록했다.",
            "", "각 종류는 같은 사건을 다른 관점에서 기록할 수 있다. 연속 오답은 겹치지 않는 프레임 구간이며, label_pair_error.csv는 이를 정답·예측 쌍으로 세분한 목록이다.",
            "", "경합 후보 태그는 주변 2초 안에 양 팀 선수가 모두 볼 3m 이내로 관측됐다는 뜻이다. 실제 태클·가로채기 여부는 미확정이다. 볼 결측과 재개 태그도 관측/DFL 라벨 기반 진단이다.",
            "", "CSV에는 오류 구간 전후 1초의 동기화된 DFL 이벤트 종류와 원본 프레임도 함께 적었다. 주변 이벤트는 원인 확정 라벨이 아니며 연결된 이벤트가 없으면 미확정으로 표시한다.",
            "", "| 모델 | 누락 전환 | 가짜 전환 | 시점 차이 | 짧은 점유 미검출 | 연속 오답 구간 | 오답 프레임 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for model in MODELS:
            r = summary.loc[model]
            header.append(f"| {model} | {int(r.missed_transitions)} | {int(r.spurious_transitions)} | {int(r.timing_errors)} | {int(r.short_missed)} | {int(r.wrong_runs)} | {int(r.wrong_frames)} |")
        header.extend(["", "## 전환 검출의 모델 간 변화", ""])
        header.extend(f"- {name} {count}개" for name, count in cross.comparison.value_counts().items())
        header.extend(["", "전체 상세는 [5fps 오류 목록](5fps_errors.md), [5+25fps 오류 목록](5plus25fps_errors.md), [모든 오류 CSV](all_errors.csv)에 있다.", ""])
        counts = pd.DataFrame(self.rows).groupby(["category", "subtype", "model"]).size().unstack("model", fill_value=0)
        header.extend(["## 유형별 전체 집계", "", "각 표의 건수는 오류 구간 또는 전환 건수다. 서로 다른 종류는 중복될 수 있다.", "",
                       "| 종류 | 세부 유형 | 5fps | 5+25fps |", "|---|---|---:|---:|"])
        for (category, subtype), row in counts.iterrows():
            header.append(f"| {CATEGORIES[category]} | {subtype} | {int(row.get('5fps', 0))} | {int(row.get('5+25fps', 0))} |")
        header.extend(["", "## 가장 긴 연속 오답", "", "시간은 각 하프 경과 시간이며 아래는 모델별 상위 5개다. 전체 목록은 위 모델별 문서에 있다.", "",
                       "| 모델 | 하프 | 시간 | 원본 프레임 | 길이(초) | 정답 / 예측 |", "|---|---:|---|---|---:|---|"])
        for model in MODELS:
            longest = sorted([r for r in self.rows if r["model"] == model and r["category"] == "wrong_run"],
                             key=lambda r: r["frame_count"], reverse=True)[:5]
            for r in longest:
                header.append(f"| {model} | {r['half']} | {r['half_start_clock']}~{r['half_end_clock']} | {r['start_frame']}~{r['end_frame_inclusive']} | {r['duration_seconds']:.2f} | {r['truth_sequence']} / {r['prediction_sequence']} |")
        header.extend(["", "## 재생성", "", "저장소 루트에서 실행한다. 기존 모델 학습이나 예측은 수행하지 않는다.", "",
                       "```bash", f'python "src/possession/pathcrf_refiner/audit_errors.py" --run "{source}"', "```", ""])
        (out / "README.md").write_text("\n".join(header))
        for model in MODELS:
            parts = [f"# {model} 전체 오류 목록", "", "[집계 기준과 요약](README.md)", ""]
            for category, title in CATEGORIES.items():
                selected = sorted([r for r in self.rows if r["model"] == model and r["category"] == category],
                                  key=lambda r: (r["half"], r["start_frame"], r["end_frame_inclusive"]))
                parts.extend([f"## {title} · {len(selected)}건", "",
                              "| ID | 하프 | 하프 경과 시각 | 원본 프레임 | 정답 / 예측 | 세부 유형 | 시점 차이(ms) | 상황 태그 | 주변 DFL 이벤트 |",
                              "|---|---:|---|---|---|---|---:|---|---|"])
                for row in selected:
                    detail = row["subtype"]
                    if row.get("direction"):
                        detail = row["direction"] + " · " + detail
                    parts.append(f"| {row['id']} | {row['half']} | {row['half_start_clock']}~{row['half_end_clock']} | "
                                 f"{row['start_frame']}~{row['end_frame_inclusive']} | {row['truth_sequence']} / "
                                 f"{row['prediction_sequence']} | {detail} | {row.get('signed_delay_ms', '')} | "
                                 f"{row['context_tags'].replace(' | ', ', ')} | "
                                 f"{row['nearby_DFL_event_types'].replace(' | ', ', ')} |")
                parts.append("")
            name = "5fps_errors.md" if model == "5fps" else "5plus25fps_errors.md"
            (out / name).write_text("\n".join(parts))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/possession/pathcrf/dfl_5train_1val_1test")
    p.add_argument("--cache", type=Path, default=ROOT / "data/external/dfl")
    p.add_argument("--output", type=Path)
    p.add_argument("--events", type=Path, default=Path("/private/tmp/pathcrf-official-dfl/data/sportec_synced/J03WR9.parquet"))
    args = p.parse_args()
    args.output = args.output or args.run / "error_audit"
    Audit(args).execute()


if __name__ == "__main__":
    main()
