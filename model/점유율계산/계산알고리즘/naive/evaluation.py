"""Official public scoring plus frame/error/transition diagnostics. No tuning."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (HERE, ROOT, FPS, PRIMARY, checked_labels, metrics as pm,
                    official_evaluation as scorer, timestamp, write_json)


def error_segments(frames, gt, pred):
    rows = []
    start = None
    # Include both the GT and prediction label in the grouping key.
    for i in range(len(frames) + 1):
        key = (gt[i], pred[i]) if i < len(frames) and gt[i] != "X" and gt[i] != pred[i] else None
        if start is not None and (key != old_key or i == len(frames) or frames[i] != frames[i - 1] + 1):
            end = i - 1
            rows.append({"start_frame": int(frames[start]), "end_frame_inclusive": int(frames[end]),
                         "end_frame_exclusive": int(frames[end]) + 1,
                         "start_time": timestamp(frames[start]),
                         "end_time_exclusive": timestamp(frames[end] + 1),
                         "duration_seconds": (frames[end] - frames[start] + 1) / FPS,
                         "gt_label": old_key[0], "pred_label": old_key[1]})
            start = None
        if key is not None and start is None:
            start = i
        old_key = key
    return pd.DataFrame(rows, columns=("start_frame", "end_frame_inclusive", "end_frame_exclusive",
                                      "start_time", "end_time_exclusive", "duration_seconds",
                                      "gt_label", "pred_label"))


def diagnostics(out, gt, entries, primary):
    folder = out / "diagnostics"
    folder.mkdir(parents=True, exist_ok=True)
    frames = np.arange(len(gt))
    alphabet = np.array(pm.LABELS)
    gt_labels = alphabet[gt]
    graded = gt != pm.X
    table = pd.DataFrame({"frame": frames, "time_seconds": frames / FPS,
                          "video_time": [timestamp(f) for f in frames],
                          "gt_label": gt_labels, "is_graded": graded.astype(int)})
    confusion, per_class, labels = [], [], []
    predictions = {}
    for entry in entries:
        name, result = entry["name"], entry["result"]
        _, codes = checked_labels(entry["path"])
        predictions[name] = alphabet[codes]
        table[name] = predictions[name]
        cm = np.array(result["confusion"])[:3, :]
        for i, truth in enumerate(pm.SCORED):
            per_class.append({"model": name, "label": truth, **result["per_class"][truth]})
            for j, prediction in enumerate(pm.LABELS):
                confusion.append({"model": name, "gt_label": truth, "pred_label": prediction,
                                  "frames": int(cm[i, j])})
        for scope, mask in (("full_video", np.ones(len(gt), dtype=bool)), ("public_graded", graded)):
            for i, label in enumerate(pm.LABELS):
                labels.append({"model": name, "scope": scope, "label": label,
                               "frames": int(np.sum(codes[mask] == i))})
    pred = predictions[primary]
    table["primary_model"] = primary
    table["primary_correct"] = np.where(graded, (pred == gt_labels).astype(int), -1)
    gt_transition = np.zeros(len(gt), dtype=int)
    gt_transition[1:] = graded[1:] & graded[:-1] & (gt[1:] != gt[:-1])
    pred_transition = np.zeros(len(gt), dtype=int)
    pred_transition[1:] = pred[1:] != pred[:-1]
    table["gt_transition"] = gt_transition
    table["primary_transition"] = pred_transition
    table.to_csv(folder / "frame_diagnostics.csv", index=False)
    table.loc[graded].to_csv(folder / "public_frame_diagnostics.csv", index=False)
    errors = error_segments(frames, gt_labels, pred)
    errors.to_csv(folder / "error_segments.csv", index=False)
    pd.DataFrame(confusion).to_csv(folder / "confusion.csv", index=False)
    pd.DataFrame(per_class).to_csv(folder / "per_class_metrics.csv", index=False)
    pd.DataFrame(labels).to_csv(folder / "label_counts.csv", index=False)
    return errors


def write_summary(out, entries, gt, primary, errors, bootstrap):
    selected = next(e for e in entries if e["name"] == primary)
    result = selected["result"]
    manifest_path = out / "run_manifest.json"
    import json
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    graded = np.flatnonzero(gt != pm.X)
    lines = ["# Naive baseline 실행 결과", "",
             f"- 입력 방식: `{manifest.get('input_mode', 'not recorded')}`.",
             f"- {manifest.get('note', 'See input provenance separately.')}",
             f"- 전체 제출: {len(gt):,}프레임. 실제 평가: {len(graded):,}프레임 "
             f"({graded[0]}–{graded[-1]}, {timestamp(graded[0])}–{timestamp(graded[-1]+1)}; 끝 시각 제외).",
             f"- 기본 모델: `{primary}`. 주최 측 고정 기준이며 공개 GT로 선택·튜닝하지 않음.",
             f"- bootstrap {bootstrap}회; 구간은 공식 scorer의 5–95 백분위(90% 구간).", "",
             "| 모델 | spellF1 | macroF1 | 정확도 | 점유 구간 수 | 전환 F1 ±25프레임 |",
             "|---|---:|---:|---:|---:|---:|"]
    for e in entries:
        r = e["result"]
        lines.append(f"| {e['name']} | {r['spell_f1_mean']:.4f} | {r['macro_f1']:.4f} | "
                     f"{r['accuracy']:.4f} | {r['n_pred_spells']} | {r['transitions'][25]['f1']:.4f} |")
    lines += ["", f"공개 정답 점유 구간은 {result['n_gt_spells']}개. 기본 모델 오답은 "
              f"{round((1-result['accuracy'])*result['n_graded']):,}프레임이며 "
              f"진단용 연속 오답 구간은 {len(errors)}개다.", "",
              "## 기본 모델의 클래스별 결과", "",
              "| 라벨 | precision | recall | F1 | 정답 프레임 수 |",
              "|---|---:|---:|---:|---:|"]
    for label, row in result["per_class"].items():
        lines.append(f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | "
                     f"{row['f1']:.4f} | {row['support']} |")
    lines += ["", "## 해석과 파일", "",
              "- 이 점수는 공개 5분에만 해당한다. 전체 경기 성능이나 모델 우열로 일반화하지 않는다.",
              "- 기본 모델은 최근접 선수 팀에 H/A를 부여하고 ±50프레임에서 H/A만 투표한다. "
              "추적 답이 없을 때 D로 처리하므로 실제 데드볼과 관측 누락을 구분하지 못한다.",
              "- 공개 정답은 파이프라인 검증에만 사용하며 학습·임계값·윈도 선택에 사용하지 않았다.",
              "- `scoreboard.csv`: 공식 평가 수치. `reports/`: 공식 JSON과 전환 CSV.",
              "- `diagnostics/frame_diagnostics.csv`: 전체 프레임의 GT·4개 예측·시간. "
              "`primary_correct=-1`은 GT X로 평가 대상이 아님.",
              "- `diagnostics/public_frame_diagnostics.csv`: 실제 평가 대상 7,500프레임만 추출.",
              "- `diagnostics/error_segments.csv`: 기본 모델의 연속 오답(동일 GT/예측 쌍) 구간.",
              "- `diagnostics/confusion.csv`, `per_class_metrics.csv`, `label_counts.csv`: 혼동·클래스·분포.",
              "- 모든 시간은 25fps 전술 영상 시작 기준이며 경기 시계가 아니다. "
              "MP4의 편집·프레임 일치 여부를 이번 실행에서 디코딩으로 확인하지 않았다.",
              ("- ATD TXT/XML의 실제 좌표에서 최근접 팀 관측을 새로 생성했다. 속도는 사용하지 않았다."
               if manifest.get("input_mode") == "atd" else
               "- 이번 실행은 공개 raw 예측에서 시작했으며 좌표 추론을 새로 수행하지 않았다."), ""]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=HERE / "predictions")
    parser.add_argument("--gt", type=Path, default=ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv")
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    _, gt = checked_labels(args.gt)
    if not np.any(gt != pm.X):
        parser.error("Ground truth has no graded frames.")
    files = sorted(args.predictions.glob("baseline_*.csv"))
    if not files or not (args.predictions / f"{PRIMARY}.csv").is_file():
        parser.error("Generate the four fixed baseline predictions first.")
    entries = []
    for path in files:
        result, reason = scorer.score_submission(str(path), gt, pm.TAUS, args.bootstrap)
        if reason:
            parser.error(f"{path}: {reason}")
        entries.append({"name": path.stem, "path": str(path), "result": result, "reason": None})
    entries.sort(key=lambda e: scorer.rank_key(e, 25))
    reports = args.out / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    scorer.write_csv(str(args.out / "scoreboard.csv"), scorer.scoreboard_columns(pm.TAUS),
                     [scorer.scoreboard_row(i, e, pm.TAUS) for i, e in enumerate(entries, 1)])
    for entry in entries:
        report = scorer.json_safe(entry["result"])
        report.update(submission=entry["name"], ground_truth=args.gt.name, taus=list(pm.TAUS))
        write_json(reports / f"{entry['name']}.json", report)
        transitions_path = reports / f"{entry['name']}_transitions.csv"
        scorer.write_transitions_csv(str(transitions_path), entry["result"], 25)
        transitions = pd.read_csv(transitions_path)
        for col in ("gt_frame", "pred_frame"):
            transitions[col.replace("frame", "video_time")] = [
                timestamp(value) if pd.notna(value) else "" for value in transitions[col]]
        transitions.to_csv(reports / f"{entry['name']}_transitions_timed.csv", index=False)
    errors = diagnostics(args.out, gt, entries, PRIMARY)
    write_summary(args.out, entries, gt, PRIMARY, errors, args.bootstrap)
    scorer.print_scoreboard(entries, [], str(args.gt), scorer.load_meta(str(args.gt)), 25)
    print(f"\nFixed primary: {PRIMARY}; detailed results: {args.out / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
