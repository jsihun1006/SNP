"""Assess detector coverage and agreement with UNCORRECTED ATD, not accuracy.

No possession ground truth is used. The single ball candidate is selected by
detector confidence before reading the reference coordinates for comparison.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

PITCH = np.array([105.3, 68.0], dtype=float)
PERSON_RADIUS_M = 3.0  # fixed diagnostic radius, not tuned against any label


def discover_root():
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents, Path.cwd()]:
        if (candidate / "games.py").is_file():
            return candidate
    return Path("/Users/jsh/Desktop/공모전")  # scratch execution on the current workstation


def truthy(values):
    return np.array([str(value).strip().lower() in {"true", "1", "1.0"} for value in values])


def point(group):
    fields = group.split(",")
    try:
        xy = np.array([float(fields[0]), float(fields[1])], dtype=float)
    except (ValueError, IndexError):
        return np.array([np.nan, np.nan])
    return xy if np.isfinite(xy).all() else np.array([np.nan, np.nan])


def load_atd_selected(path, sampled_frames):
    """Stream the TXT; keep only requested zero-based video frames."""
    requested = set(int(frame) for frame in sampled_frames)
    records = {}
    malformed = 0
    with Path(path).open() as stream:
        for line in stream:
            parts = line.strip().split(":")
            if len(parts) != 3:
                malformed += 1
                continue
            try:
                frame = int(parts[0]) - 1
            except ValueError:
                malformed += 1
                continue
            if frame not in requested:
                continue
            players = np.array([point(group) for group in parts[1].split(";")], dtype=float).reshape(-1, 2)
            players = players[np.isfinite(players).all(axis=1)]
            records[frame] = {"players": players * PITCH, "ball": point(parts[2]) * PITCH}
    return records, {"malformed_lines": malformed,
                     "requested_frames": len(requested), "found_frames": len(records),
                     "missing_requested_frames": sorted(requested - set(records))}


def greedy_matches(predicted, reference, radius=PERSON_RADIUS_M):
    """One-to-one spatial pairs within radius; NOT persistent identity matches."""
    predicted = np.asarray(predicted, dtype=float).reshape(-1, 2)
    reference = np.asarray(reference, dtype=float).reshape(-1, 2)
    if not len(predicted) or not len(reference):
        return []
    distances = np.linalg.norm(predicted[:, None] - reference[None, :], axis=2)
    candidates = np.argwhere(distances <= radius)
    order = sorted(candidates.tolist(), key=lambda pair: (distances[tuple(pair)], pair[0], pair[1]))
    used_pred, used_ref, matches = set(), set(), []
    for i, j in order:
        if i in used_pred or j in used_ref:
            continue
        used_pred.add(i)
        used_ref.add(j)
        matches.append((i, j, float(distances[i, j])))
    return matches


def select_detections(detections):
    """Select in-pitch person points and one confidence-selected ball per frame."""
    selected = {}
    if not len(detections):
        return selected
    required = {"frame", "class_id", "confidence", "x_m", "y_m", "projection_valid", "in_pitch"}
    if not required <= set(detections.columns):
        raise ValueError(f"Detection columns missing: {sorted(required-set(detections.columns))}")
    usable = truthy(detections.projection_valid) & truthy(detections.in_pitch)
    usable &= np.isfinite(detections[["x_m", "y_m"]].to_numpy(dtype=float)).all(axis=1)
    usable_table = detections.loc[usable].copy()
    for frame, group in usable_table.groupby("frame", sort=False):
        people = group.loc[group.class_id.isin([1, 2, 3]), ["x_m", "y_m"]].to_numpy(dtype=float)
        balls = group.loc[group.class_id == 0]
        if len(balls):
            best = balls.sort_values("confidence", ascending=False, kind="stable").iloc[0]
            ball = best[["x_m", "y_m"]].to_numpy(dtype=float)
            confidence = float(best.confidence)
        else:
            ball, confidence = np.array([np.nan, np.nan]), np.nan
        selected[int(frame)] = {"players": people, "ball": ball, "ball_confidence": confidence}
    return selected


def compare_frames(status, selected, references):
    rows = []
    for entry in status.itertuples(index=False):
        frame = int(entry.frame)
        det = selected.get(frame, {"players": np.empty((0, 2)), "ball": np.array([np.nan, np.nan]), "ball_confidence": np.nan})
        ref = references.get(frame)
        players = ref["players"] if ref is not None else np.empty((0, 2))
        ref_ball = ref["ball"] if ref is not None else np.array([np.nan, np.nan])
        det_ball = det["ball"]
        observed = bool(np.isfinite(ref_ball).all())
        detected = bool(np.isfinite(det_ball).all())
        matches = greedy_matches(det["players"], players)
        rows.append({"frame": frame, "time_seconds": float(entry.time_seconds),
                     "atd_record_present": ref is not None,
                     "det_person_in_pitch": len(det["players"]), "atd_person_slots_finite": len(players),
                     "person_pairs_within_3m": len(matches),
                     "person_pair_mean_distance_m": float(np.mean([m[2] for m in matches])) if matches else np.nan,
                     "det_ball_in_pitch": detected, "atd_ball_present": observed,
                     "det_ball_confidence": det["ball_confidence"],
                     "det_ball_x_m": det_ball[0], "det_ball_y_m": det_ball[1],
                     "atd_ball_x_m": ref_ball[0], "atd_ball_y_m": ref_ball[1],
                     "ball_reference_distance_m": float(np.linalg.norm(det_ball-ref_ball)) if observed and detected else np.nan,
                     "ball_reference_within_3m": bool(np.linalg.norm(det_ball-ref_ball) <= 3.0) if observed and detected else None})
    return pd.DataFrame(rows)


def ratio(a, b):
    return float(a / b) if b else None


def summarize_scope(comparison, status):
    n = len(comparison)
    distances = comparison.ball_reference_distance_m.dropna().to_numpy(dtype=float)
    matches = int(comparison.person_pairs_within_3m.sum())
    det_person = int(comparison.det_person_in_pitch.sum())
    ref_person = int(comparison.atd_person_slots_finite.sum())
    atd_ball = int(comparison.atd_ball_present.sum())
    det_ball = int(comparison.det_ball_in_pitch.sum())
    return {"sampled_frames": n,
            "person_detection_count_all_image": int(status.n_person.sum()),
            "ball_detection_count_all_image": int(status.n_ball.sum()),
            "person_detections_in_pitch": det_person,
            "atd_finite_person_slots": ref_person,
            "person_pairs_within_3m": matches,
            "person_detector_points_paired_fraction": ratio(matches, det_person),
            "person_atd_points_paired_fraction": ratio(matches, ref_person),
            "mean_person_detections_in_pitch_per_frame": ratio(det_person, n),
            "mean_atd_person_slots_per_frame": ratio(ref_person, n),
            "frames_with_person_in_pitch": int((comparison.det_person_in_pitch > 0).sum()),
            "frames_with_ball_in_pitch": det_ball,
            "detector_ball_coverage": ratio(det_ball, n),
            "atd_ball_present_frames": atd_ball,
            "atd_ball_coverage": ratio(atd_ball, n),
            "frames_with_both_ball_positions": len(distances),
            "frames_atd_ball_present_but_detector_absent": int((comparison.atd_ball_present & ~comparison.det_ball_in_pitch).sum()),
            "frames_detector_ball_present_but_atd_absent": int((comparison.det_ball_in_pitch & ~comparison.atd_ball_present).sum()),
            "ball_reference_distance_mean_m": float(np.mean(distances)) if len(distances) else None,
            "ball_reference_distance_median_m": float(np.median(distances)) if len(distances) else None,
            "ball_reference_distance_p90_m": float(np.percentile(distances, 90)) if len(distances) else None,
            "ball_reference_within_3m_fraction_when_both_present": ratio(int((distances <= 3.0).sum()), len(distances))}


def human(value, percent=False):
    if value is None:
        return "측정 불가"
    return f"{100*value:.2f}%" if percent else f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--root", type=Path, default=discover_root())
    parser.add_argument("--atd", type=Path)
    parser.add_argument("--homography", type=Path)
    args = parser.parse_args()
    atd = args.atd or args.root / "data/tactical/BAR-ATH_tactical_atd.txt"
    homography = args.homography or args.root / "data/tactical/BAR-ATH_tactical_homography.csv"
    status = pd.read_csv(args.out / "frame_status.csv")
    detections = pd.read_csv(args.out / "detections.csv")
    if not len(status) or status.frame.duplicated().any() or not status.frame.is_monotonic_increasing:
        raise ValueError("Frame status must be nonempty, sorted and unique")
    if not np.issubdtype(status.frame.dtype, np.integer):
        raise ValueError("Sampled frame indices must be integers")
    if len(detections) and not detections.frame.isin(status.frame).all():
        raise ValueError("Detection refers to a frame absent from frame_status")
    manifest_path = args.out / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    fps = manifest.get("video", {}).get("fps")
    if fps is not None and not np.isclose(float(fps), 25):
        raise ValueError("Only the 25 fps tactical timeline is supported")
    references, atd_status = load_atd_selected(atd, status.frame)
    selected = select_detections(detections)
    comparison = compare_frames(status, selected, references)
    comparison.to_csv(args.out / "reference_comparison.csv", index=False)
    game_spec = importlib.util.spec_from_file_location("_video_naive_games", args.root / "games.py")
    games = importlib.util.module_from_spec(game_spec)
    game_spec.loader.exec_module(games)
    blocks = games.half_video_blocks(games.GAME)
    half_masks = {f"half_{half}_sampled": comparison.frame.between(start, stop).to_numpy()
                  for half, start, stop in blocks}
    match_mask = np.zeros(len(comparison), dtype=bool)
    for mask in half_masks.values():
        match_mask |= mask
    masks = {"all_sampled": np.ones(len(comparison), dtype=bool),
             "match_halves_sampled": match_mask, **half_masks}
    summaries = {name: summarize_scope(comparison.loc[mask], status.loc[mask]) for name, mask in masks.items()}
    homography_table = pd.read_csv(homography, usecols=["frame", "valid", "reprojection_error"])
    h = homography_table.set_index("frame").reindex(status.frame)
    output = {"assessment_kind": "coverage_and_uncorrected_ATD_agreement_NOT_accuracy",
              "person_matching": "greedy one-to-one position pairs within fixed 3 m; no identities",
              "ball_selection": "highest confidence in-pitch candidate, selected without reference",
              "coordinate_units": "metres; normalized pitch multiplied by 105.3 x 68.0",
              "timing_notes": {
                  "inference_seconds": "predict 호출 구간 합계. 비동기 GPU 완료를 보장하지 않고 CPU 좌표 변환·내보내기 전체 시간도 포함하지 않음.",
                  "total_seconds": "추출 실행의 실제 경과 시간(wall time). 처리량 판단에는 이 값을 사용."
              },
              "atd_path": str(atd.resolve()), "atd_read_status": atd_status,
              "sampled_homography_valid_count": int((h.valid == 1).sum()),
              "sampled_homography_reprojection_error_median": float(h.reprojection_error.median()),
              "scopes": summaries,
              "limitations": ["ATD is an uncorrected automatic reference, not coordinate/box ground truth.",
                              "No detector precision, recall, mAP or true coordinate accuracy can be measured here.",
                              "Football classes include referee and goalkeeper; detections provide no team identities.",
                              "Person bottom-center and ball center are ground-plane approximations; airborne balls violate the plane.",
                              "Provided homography is used; this is not video-only camera calibration.",
                              "Sampled observations are not detections on skipped frames and are not a full 25 fps tracking export."]}
    (args.out / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    lines = ["# 영상 기반 좌표 추출 결과", "",
             "사전학습 객체 검출기로 영상의 사람·공 후보를 검출하고, 제공 homography의 역행렬로 피치 좌표를 구했다.",
             "사람은 bbox 하단 중앙, 공은 bbox 중앙을 사용한다. 이 과정은 영상 검출 + 제공 카메라 보정의 조합이다.", "",
             "## 실행 범위", "",
             f"- 실제 검출한 샘플 프레임: {len(status):,}개. 원본 frame 번호를 유지했다.",
             f"- 설정: `{json.dumps(manifest.get('config', {}), ensure_ascii=False)}`.",
             f"- 영상 정보: `{json.dumps(manifest.get('video', {}), ensure_ascii=False)}`.",
             f"- predict 호출 구간 합계: {manifest.get('inference_seconds', '미기록')}초 "
             "(비동기 GPU 완료를 보장하지 않으며 CPU 좌표 변환·내보내기 전체 시간도 아님).",
             f"- 추출 전체 실제 경과 시간(wall time): {manifest.get('total_seconds', '미기록')}초. 처리량 판단에는 이 값을 사용한다.", "",
             "## 검출 커버리지와 참고 데이터 비교", "",
             "| 범위 | 샘플 프레임 | 평균 피치 내 사람 수 | 공 검출 커버리지 | ATD 공 존재 비율 | 둘 다 있는 공 좌표의 거리 중앙값(m) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, row in summaries.items():
        lines.append(f"| {name} | {row['sampled_frames']} | {human(row['mean_person_detections_in_pitch_per_frame'])} | "
                     f"{human(row['detector_ball_coverage'],True)} | {human(row['atd_ball_coverage'],True)} | "
                     f"{human(row['ball_reference_distance_median_m'])} |")
    lines += ["", "**위 수치는 정확도나 recall이 아니다.** ATD 자체가 보정되지 않은 자동 추적이므로 거리 차이는 참고 데이터와의 일치도일 뿐이다.",
              "공 좌표 비교에는 ATD와 검출기 모두 공 위치가 있는 프레임만 들어가므로 누락된 어려운 프레임의 성능을 대표하지 못한다.",
              "공 후보는 ATD를 보기 전에 confidence가 가장 높은 피치 내 후보로 고정했다.", "",
              "## 파일", "",
              "- `outputs/detections.csv`: 실제 검출 bbox·confidence·픽셀 기준점·정규화 좌표·미터 좌표·투영 플래그.",
              "- `outputs/frame_status.csv`: 검출을 실행한 모든 샘플 프레임의 사람/공 개수. 무검출 프레임도 포함.",
              "- `outputs/reference_comparison.csv`: 프레임별 ATD 존재 여부·사람 3m 공간 짝 수·공 좌표 차이.",
              "- `outputs/summary.json`: 전체/경기 구간/전후반 별 집계 및 한계.", "",
              "## 해석상 한계", "",
              "- 사람 후보에는 심판·골키퍼와 오검출이 포함될 수 있다. 팀 분류와 지속적인 선수 추적 ID는 별도 문제다.",
              "- 작은 공은 기본 객체 검출기가 놓치기 쉽다. 공 검출률이 낮으면 점유 추론의 대체 입력으로 충분하지 않다.",
              "- 피치 homography는 지면 변환이다. 공중에 뜬 공이나 발이 잘린 사람 bbox의 투영에는 오차가 생긴다.",
              "- 샘플링한 프레임만 추론했다. 건너뛴 프레임에 검출이 있었다고 간주하거나 보간 좌표를 실제 검출로 보고하지 않는다.",
              "- 이번 산출물은 좌표 검출 결과다. 팀 분류 없이 H/A 점유 예측을 완성하거나 spellF1으로 평가한 것은 아니다.", ""]
    (args.out / "assessment.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
