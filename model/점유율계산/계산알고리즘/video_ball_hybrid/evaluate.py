"""Official fixed-model pipeline validation against the public possession GT.

No model configuration or threshold is selected from this output. The detector
input for the evaluated prediction must be the uniform full-video sample; a
dense diagnostic clip cannot be selectively inserted into the public window.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def root_path(explicit):
    if explicit:
        candidates = [Path(explicit)]
    else:
        here = Path(__file__).resolve().parent
        candidates = [here, *here.parents, Path.cwd()]
    result = next((p.resolve() for p in candidates if (p / "games.py").is_file()), None)
    if result is None:
        raise ValueError("Cannot find repository containing games.py; supply --repo")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True, help="fixed hybrid prediction.csv")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--gt", type=Path, help="custom GT for synthetic testing; default is public GT")
    parser.add_argument("--baseline", type=Path, help="custom baseline for synthetic testing")
    parser.add_argument("--model1", type=Path, help="custom model1 for synthetic testing")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap must be nonnegative")
    root = root_path(args.repo)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "evaluation"))
    import possession_metrics as pm
    import run_evaluation as scorer
    from games import GAME, GAMES

    public_gt = root / "evaluation/gt/BAR-ATH_possession_gt_public.csv"
    gt_path = args.gt or public_gt
    inputs = {
        "baseline_smooth50": args.baseline or root / "model/모델별결과csv/naive/predictions/baseline_smooth50.csv",
        "model1": args.model1 or root / "model/모델별결과csv/model1/predictions/model1.csv",
        "video_ball_hybrid": args.prediction,
    }
    for path in [gt_path, *inputs.values()]:
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    uniform_sampling_verified = False
    source_manifest_path = args.prediction.parent / "run_manifest.json"
    if gt_path.resolve() == public_gt.resolve():
        if not source_manifest_path.is_file():
            raise ValueError("Hybrid provenance missing; cannot verify uniform detector sampling")
        source_manifest = json.loads(source_manifest_path.read_text())
        if digest(args.prediction) != source_manifest.get("prediction_sha256"):
            raise ValueError("Prediction does not match its generation manifest")
        if source_manifest.get("ground_truth_used") is not False:
            raise ValueError("Hybrid manifest does not attest GT-free inference")
        detector_path = Path(source_manifest["inputs"][0]["path"])
        detector_manifest_path = detector_path.parent / "run_manifest.json"
        if not detector_manifest_path.is_file():
            raise ValueError("Detector manifest missing; cannot verify uniform sampling")
        detector_manifest = json.loads(detector_manifest_path.read_text())
        detector_config = detector_manifest["config"]
        if (detector_manifest.get("status") != "complete" or detector_config.get("stride") != 25
                or detector_config.get("start_frame") != 0
                or detector_config.get("end_frame_exclusive") != GAMES[GAME]["n_video_frames"]):
            raise ValueError("Require complete, uniform 1fps samples for entire tactical video")
        if digest(detector_path) != source_manifest["inputs"][0]["sha256"]:
            raise ValueError("Detector data differs from the hybrid inference input")
        status = pd.read_csv(detector_path.parent / "frame_status.csv", usecols=["frame"])
        if not np.array_equal(status.frame.to_numpy(), np.arange(0, GAMES[GAME]["n_video_frames"], 25)):
            raise ValueError("Detector frame status is not the expected uniform 1fps source frame sequence")
        uniform_sampling_verified = True
    if (args.out / "scoreboard.csv").exists():
        parser.error("Refusing to overwrite prior score output; use a new --out")
    args.out.mkdir(parents=True, exist_ok=True)
    gt_table = pm.read_label_csv(str(gt_path))
    n_expected = GAMES[GAME]["n_video_frames"]
    reason = pm.validate_submission(gt_table, n_expected)
    if reason:
        raise ValueError(f"Ground truth invalid: {reason}")
    gt_codes = pm.label_codes(gt_table)
    graded = np.flatnonzero(gt_codes != pm.X)
    if not len(graded):
        raise ValueError("No graded GT frames")
    meta = scorer.load_meta(str(gt_path))
    if meta.get("n_video_frames") not in (None, len(gt_codes)):
        raise ValueError("GT metadata does not match video length")

    entries = []
    for name, path in inputs.items():
        result, reason = scorer.score_submission(str(path), gt_codes, pm.TAUS, args.bootstrap)
        if reason:
            raise ValueError(f"Invalid {name} submission: {reason}")
        entries.append({"name": name, "path": str(path), "result": result, "reason": None})
    entries.sort(key=lambda entry: scorer.rank_key(entry, max(pm.TAUS)))
    scores = [scorer.scoreboard_row(rank, entry, pm.TAUS) for rank, entry in enumerate(entries, start=1)]
    scorer.write_csv(str(args.out / "scoreboard.csv"), scorer.scoreboard_columns(pm.TAUS), scores)
    reports = args.out / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        report = scorer.json_safe(entry["result"])
        report.update(submission=entry["name"], ground_truth=gt_path.name, taus=list(pm.TAUS))
        (reports / f"{entry['name']}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        scorer.write_transitions_csv(str(reports / f"{entry['name']}_transitions.csv"),
                                     entry["result"], max(pm.TAUS))
    provenance = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "ground_truth_scope": "public_5min" if gt_path.resolve() == public_gt.resolve() else "custom",
                  "graded_frames": len(graded), "first_graded_frame": int(graded[0]),
                  "last_graded_frame": int(graded[-1]), "bootstrap_draws": args.bootstrap,
                  "no_gt_tuning": True,
                  "sampling_policy": "uniform full-video detector sample; dense diagnostic clip excluded",
                  "uniform_sampling_verified": uniform_sampling_verified,
                  "ground_truth": {"path": str(gt_path.resolve()), "sha256": digest(gt_path)},
                  "predictions": {name: {"path": str(path.resolve()), "sha256": digest(path)} for name, path in inputs.items()},
                  "official_metrics_sha256": digest(root / "evaluation/possession_metrics.py"),
                  "official_evaluator_sha256": digest(root / "evaluation/run_evaluation.py"),
                  "evaluation_code_sha256": digest(Path(__file__))}
    (args.out / "evaluation_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+"\n")

    mapping = {entry["name"]: entry["result"] for entry in entries}
    primary = mapping["video_ball_hybrid"]
    lines = ["# 영상 공 검출 + ATD 점유 추론 결과", "",
             f"- 평가 대상: {len(graded):,}프레임 (원본 영상 {graded[0]}–{graded[-1]}).",
             "- 공개 5분 정답은 파이프라인 확인용이다. 학습·설정 선택·튜닝에 쓰지 않았다.",
             "- 입력 예측은 전체 영상의 균일 1fps 영상 검출이다. 별도 25fps 구간은 이 평가에 섞지 않았다.",
             "- 검출 공 후보는 confidence 0.10 이상·피치 내 최고 confidence를 선택한 뒤 최대 12프레임 유지한다.",
             "- 영상 공 정보가 없으면 ATD 공 좌표를 쓰며, 두 경우 모두 ATD 선수 좌표와 팀 정보로 최근접 팀을 구한다.",
             "- 기존 고정 model1 디코더가 시간 흐름으로 H/A/D를 추론한다. D는 관측 누락 길이에 기반한 휴리스틱이며 실제 데드볼 확정이 아니다.",
             "- 낮은 confidence의 잘못된 공 검출과 고정 좌표 유지, 선수·공의 자동 추적 오차가 영향을 줄 수 있다.", "",
             "| 모델 | spellF1 | macroF1 | 정확도 | 전환 F1 ±25프레임 | 예측 점유 구간 수 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in ("baseline_smooth50", "model1", "video_ball_hybrid"):
        result = mapping[name]
        lines.append(f"| {name} | {result['spell_f1_mean']:.4f} | {result['macro_f1']:.4f} | "
                     f"{result['accuracy']:.4f} | {result['transitions'][max(pm.TAUS)]['f1']:.4f} | {result['n_pred_spells']} |")
    lines += ["", f"정답 점유 구간 {primary['n_gt_spells']}개. 점수는 공개 일부 구간에서만 계산됐다.",
              "공개 표본이 작아 전체 경기 성능이나 모델 우열을 확정할 수 없다.", "",
              "## 하이브리드 모델 클래스별 지표", "",
              "| 라벨 | precision | recall | F1 | 정답 프레임 |", "|---|---:|---:|---:|---:|"]
    for label, result in primary["per_class"].items():
        lines.append(f"| {label} | {result['precision']:.4f} | {result['recall']:.4f} | {result['f1']:.4f} | {result['support']} |")
    if args.bootstrap and primary.get("spell_bootstrap_low") is not None:
        lines += ["", f"공식 spellF1 bootstrap 5–95백분위 구간: "
                  f"{primary['spell_bootstrap_low']:.4f}–{primary['spell_bootstrap_high']:.4f} "
                  f"({args.bootstrap}회; 90% 구간)."]
    lines += ["", "`scoreboard.csv`는 공식 평가 함수의 출력이다. `reports/`에 모델별 JSON과 전환 CSV를 저장한다.",
              "원본 영상·좌표와 프레임 단위 파생 CSV는 로컬에 보관한다.", ""]
    (args.out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print((args.out / "RESULTS.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
