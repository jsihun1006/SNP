"""One public evaluation after freezing model1; compare to the same-input baseline."""
import argparse
import json
from pathlib import Path

import pandas as pd

from common import (HERE, ROOT, checked_labels, metrics as pm,
                    official_evaluation as scorer, naive_helper, sha256, timestamp, write_json)


def summary(out, entries, manifest, errors):
    by_name = {e["name"]: e["result"] for e in entries}
    model, baseline = by_name["model1"], by_name["baseline_smooth50"]
    delta = model["spell_f1_mean"] - baseline["spell_f1_mean"]
    scope_note = ("- 좌표 TXT/XML이 없어 현재 실행에서는 거리·속도·선수 제어 특성이나 MP4를 사용하지 않았다."
                  if manifest["input_mode"] == "published-raw" else
                  "- ATD 좌표로 최근접 팀 관측을 새로 생성했다. 시퀀스 추론은 팀/누락만 사용하며 속도·영상·태그는 사용하지 않았다.")
    lines = ["# model1 실행 결과", "",
             "## 입력과 실행 범위", "",
             f"- 입력: `{manifest['input_mode']}`. 현재 raw 모드는 주최 측 최근접 팀 예측에서 시작한다.",
             scope_note,
             "- 알고리즘: 관측 불일치 비용 + 상태 전환 비용을 최소화하는 H/A/D 오프라인 시퀀스 추론.",
             "- 설정은 예측 전에 해시와 함께 저장했다. 공개 GT로 학습·탐색·재조정하지 않았다.",
             "- 공개 평가 범위: 11185–18684 프레임, 7,500프레임(5분). 나머지 GT X는 채점 제외.",
             "- 순위 기준 spellF1, 보조 macroF1. bootstrap 1000회, 5–95 백분위의 90% 구간.", "",
             "## 같은 입력으로 비교", "",
             "| 모델 | spellF1 | macroF1 | 정확도 | 점유 구간 수 | 전환 F1 ±25프레임 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in ("baseline_smooth50", "model1"):
        r = by_name[name]
        lines.append(f"| {name} | {r['spell_f1_mean']:.4f} | {r['macro_f1']:.4f} | "
                     f"{r['accuracy']:.4f} | {r['n_pred_spells']} | {r['transitions'][25]['f1']:.4f} |")
    lines += ["", f"model1 − baseline: spellF1 {delta:+.4f}, "
              f"macroF1 {model['macro_f1']-baseline['macro_f1']:+.4f}. "
              f"공개 정답 점유 구간은 {model['n_gt_spells']}개.", "",
              f"model1 spellF1의 공식 90% bootstrap 구간: "
              f"[{model['spell_bootstrap_low']:.4f}, {model['spell_bootstrap_high']:.4f}]. "
              f"baseline 구간: [{baseline['spell_bootstrap_low']:.4f}, {baseline['spell_bootstrap_high']:.4f}].",
              "공개 표본이 작고 구간은 모델별 추정치다. 차이의 통계 검정이나 전체 경기 성능의 증거가 아니다.", "",
              "## model1 클래스별 결과", "",
              "| 라벨 | precision | recall | F1 | 정답 프레임 수 |",
              "|---|---:|---:|---:|---:|"]
    for label, r in model["per_class"].items():
        lines.append(f"| {label} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['support']} |")
    lines += ["", "## 오류 진단", "",
              f"- 오답 프레임 {round((1-model['accuracy'])*model['n_graded']):,}개, "
              f"동일 정답/예측 쌍의 연속 오답 구간 {len(errors)}개.",
              f"- ±25프레임 전환: 일치 {model['transitions'][25]['n_matched']}개, "
              f"누락 {model['transitions'][25]['n_missed']}개, "
              f"불필요 전환 {model['transitions'][25]['n_spurious']}개.",
              "- 짧은 누락을 D로 확정하지 않고 앞뒤 팀 관측을 함께 본다. 짧은 상대 팀 관측도 바로 점유 전환으로 확정하지 않는다.",
              "- 긴 누락은 여전히 D의 대리 신호일 뿐이다. 실제 데드볼 판별기나 실제 볼 컨트롤 판별기는 아니다.",
              "- 시간은 원본 tactical 영상 시작 기준이다. MP4 디코딩/정렬 확인은 수행하지 않았다.", "",
              "## 산출물", "",
              "- `predictions/model1.csv`: 제출 형식의 전체 프레임 예측.",
              "- `diagnostics/inference_evidence.csv`: GT 없이 생성한 관측·누락 길이·상태별 비용.",
              "- `diagnostics/frame_diagnostics.csv`, `public_frame_diagnostics.csv`: 정답·두 모델 예측·정오·시간.",
              "- `diagnostics/error_segments.csv`, `confusion.csv`, `per_class_metrics.csv`: 오답 구간과 클래스 진단.",
              "- `scoreboard.csv`, `reports/`: 공식 점수·JSON·전환 CSV. `validation.json`: 공식 형식 검사.",
              "- `pre_evaluation_manifest.json`, `run_manifest.json`: 설정·입력·코드·예측 해시와 출처.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap != 1000:
        parser.error("This fixed first evaluation uses 1000 bootstrap draws.")
    manifest = json.loads((args.out / "run_manifest.json").read_text())
    prediction = args.out / "predictions/model1.csv"
    if sha256(prediction) != manifest["prediction_sha256"]:
        parser.error("Prediction differs from its inference manifest. Regenerate it.")
    gt_path = ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"
    _, gt = checked_labels(gt_path)
    entries = []
    for name in ("model1", "baseline_smooth50"):
        path = args.out / "predictions" / f"{name}.csv"
        result, reason = scorer.score_submission(str(path), gt, pm.TAUS, args.bootstrap)
        if reason:
            parser.error(f"{name}: {reason}")
        entries.append({"name": name, "path": str(path), "result": result, "reason": None})
    entries.sort(key=lambda entry: scorer.rank_key(entry, 25))
    scorer.write_csv(str(args.out / "scoreboard.csv"), scorer.scoreboard_columns(pm.TAUS),
                     [scorer.scoreboard_row(i, e, pm.TAUS) for i, e in enumerate(entries, 1)])
    reports = args.out / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        report = scorer.json_safe(entry["result"])
        report.update(submission=entry["name"], ground_truth=gt_path.name, taus=list(pm.TAUS))
        write_json(reports / f"{entry['name']}.json", report)
        path = reports / f"{entry['name']}_transitions.csv"
        scorer.write_transitions_csv(str(path), entry["result"], 25)
        timed = pd.read_csv(path)
        for col in ("gt_frame", "pred_frame"):
            timed[col.replace("frame", "video_time")] = [timestamp(v) if pd.notna(v) else "" for v in timed[col]]
        timed.to_csv(reports / f"{entry['name']}_transitions_timed.csv", index=False)
    helper = naive_helper("evaluation.py")
    errors = helper.diagnostics(args.out, gt, entries, "model1")
    summary(args.out, entries, manifest, errors)
    scorer.print_scoreboard(entries, [], str(gt_path), scorer.load_meta(str(gt_path)), 25)
    print(f"\nResults: {args.out / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
