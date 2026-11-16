"""공식 평가 코드로 점수를 낸다(공개 5분 GT, 파이프라인 확인용. 학습·튜닝·모델 선택에 쓰지 않는다).

    python model/점유율계산/계산알고리즘/model1_dead_ball/evaluation.py                # 가장 최근 실행
    python model/점유율계산/계산알고리즘/model1_dead_ball/evaluation.py --run-id <실행ID>

출력: model/점유율계산/성능지표/model1_dead_ball/<실행ID>/ (scoreboard.csv, reports/, RESULTS.md, evaluation_manifest.json)
그리고 model/점유율계산/성능지표/모델_성능_비교.csv에 model1_dead_ball__<실행ID> 줄을 넣는다(같은 이름이 있으면 바꾼다).
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model1"))
from common import (ROOT, checked_labels, metrics as pm, official_evaluation as scorer, sha256,  # noqa: E402
                    write_json)
sys.path.insert(0, str(HERE))   # model1에도 validate.py가 있어 이 폴더를 앞에 둔다
from validate import run_dir  # noqa: E402

NAME = "model1_dead_ball"
SCORE_ROOT = ROOT / "model" / "점유율계산" / "성능지표"
COLUMNS = ["submission", "spell_f1_mean", "macro_f1", "accuracy", "f1_H", "f1_A", "f1_D", "trans_f1_25", "poss",
           "n_graded"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    out = run_dir(args.run_id)
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    prediction = out / "prediction.csv"
    if sha256(prediction) != manifest["prediction_sha256"]:
        parser.error("prediction.csv differs from its run_manifest; run possession.py again")
    gt_path = ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"
    _, gt = checked_labels(gt_path)
    result, reason = scorer.score_submission(str(prediction), gt, pm.TAUS, args.bootstrap)
    if reason:
        parser.error(reason)
    entry = {"name": f"{NAME}__{out.name}", "path": str(prediction), "result": result, "reason": None}
    dest = SCORE_ROOT / NAME / out.name
    (dest / "reports").mkdir(parents=True, exist_ok=True)
    scorer.write_csv(str(dest / "scoreboard.csv"), scorer.scoreboard_columns(pm.TAUS),
                     [scorer.scoreboard_row(1, entry, pm.TAUS)])
    report = scorer.json_safe(result)
    report.update(submission=entry["name"], ground_truth=gt_path.name, taus=list(pm.TAUS))
    write_json(dest / "reports" / f"{NAME}.json", report)
    scorer.write_transitions_csv(str(dest / "reports" / f"{NAME}_transitions.csv"), result, 25)
    per_class = result["per_class"]
    row = {"submission": entry["name"], "spell_f1_mean": round(result["spell_f1_mean"], 6),
           "macro_f1": round(result["macro_f1"], 6), "accuracy": round(result["accuracy"], 6),
           "f1_H": round(per_class["H"]["f1"], 6), "f1_A": round(per_class["A"]["f1"], 6),
           "f1_D": round(per_class["D"]["f1"], 6), "trans_f1_25": round(result["transitions"][25]["f1"], 6),
           "poss": result["n_pred_spells"], "n_graded": result["n_graded"]}
    lines = [f"# {NAME} 실행 결과 ({out.name})", "",
             "- D: dead_ball Phase 1 결과 그대로. H/A: model1의 최근접 팀 관측 + 전환 비용 디코더.",
             f"- 입력 dead_ball 실행: `{manifest['phase1_input']['run_id']}`.",
             "- 공개 5분 GT는 파이프라인 확인용이다. 이 점수로 설정을 고르거나 고치지 않는다.", "",
             "| spellF1 | macroF1 | 정확도 | F1 H | F1 A | F1 D | 전환 F1 ±25 | 점유 구간 수 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|",
             f"| {row['spell_f1_mean']:.4f} | {row['macro_f1']:.4f} | {row['accuracy']:.4f} | {row['f1_H']:.4f} | "
             f"{row['f1_A']:.4f} | {row['f1_D']:.4f} | {row['trans_f1_25']:.4f} | {row['poss']} |", ""]
    (dest / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(dest / "evaluation_manifest.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "prediction": str(prediction),
        "prediction_sha256": sha256(prediction), "ground_truth": gt_path.name, "bootstrap": args.bootstrap,
        "evaluation_code_sha256": sha256(ROOT / "evaluation" / "run_evaluation.py"), "note": "public 5-minute pipeline check"})
    table = SCORE_ROOT / "모델_성능_비교.csv"
    rows = []
    if table.is_file():
        with table.open(encoding="utf-8") as handle:
            rows = [r for r in csv.DictReader(handle) if r["submission"] != row["submission"]]
    rows.append({k: row[k] for k in COLUMNS})
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print("\n".join(lines))
    print(f"wrote {dest.relative_to(ROOT)} and updated {table.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
