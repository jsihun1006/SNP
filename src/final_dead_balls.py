"""최종 데드볼(파이프라인 2/2): 모델 확률에 라인 아웃 처리와 공개 킥오프를 더한다.

    python src/final_dead_balls.py      # 직접 연 터미널에서 실행할 것(대회 트래킹을 읽는다)

입력: src/apply_competition.py의 결과(data/dead_balls/model_<특징 세트>/frames.csv, segments.csv)와 대회 ATD
출력: data/dead_balls/final_model/ (커밋 안 함)
    frames.csv         frame, dead(최종 판정, 채점 구간만)
    segments.csv       최종 데드볼 구간: start, end(포함), start_clock, seconds,
                       kind(model | model+out_start | forced_line_out | public_kickoff)
    frame_signals.csv  frame, eps(선수 평균 속도), ball_speed, seen, inside — 공 보임 여부 등(Phase 1 내보내기와 로컬 뷰어용)
    viewer.csv         로컬 확인용 뷰어(dead_ball_viewer.html, 저장소에 없음)가 읽는 구간 목록
    missed_tags.csv, viewer_missed.csv   최종 구간과 겹치지 않는 SET PIECES 태그(진단용)
    summary.txt        화면 출력과 같은 집계

순서:
  1. 모델 확률의 2상태 비터비(벌점 40). 진짜 라인 아웃이 보인 프레임에서는 인플레이 → 데드 전환 벌점이 0이다.
     느린 빌드업 보정(slow_buildup.py, JSH ⑤): 패스·수신이나 공과 선수의 동행이 보이는 구간은 데드 진입 비용을
     높이고 데드 쪽 점수를 낮춘다. Metrica 3경기 0.809 → 0.820, SkillCorner 20경기 0.729 → 0.748.
  2. 확실한 라인 아웃은 무조건 데드볼: 모델이 인플레이로 본 라인 아웃 프레임부터, 공이 경기장 안에서 다시 보이고
     모델이 인플레이로 볼 때까지(force_line_outs).
  3. 시작 보정: 데드볼 시작 앞뒤 out_start_window_s 안에 라인 아웃이 보이면 그 프레임을 시작으로 한다.
  4. 공개 킥오프(FAQ.md): 후반 채점 시작~킥오프는 D이고 킥오프에서 끝난다. 전반 킥오프 프레임에서 시작하는 구간은 뺀다.
라인 아웃 = 공이 경기장 안(보임)에서 밖(보임)으로 넘고, 넘기 직전 1초 안에 굴러가고 있었으며, 넘은 뒤
out_confirm_s 안에 경기장 안에서 빠른 공이 보이지 않고, out_min_restart_s(공이 나간 뒤 가장 빠른 재개,
DFL에서 잼) 안에 경기장 안에서 공이 (속도와 상관없이) 보이지 않는 것.
진짜 아웃이면 공을 가져와 다시 차거나 던질 때까지 그보다 빨리 안으로 돌아올 수 없다. 더 빨리 돌아오면
높이 뜬 공이 2D 투영으로 밖에 찍힌 것으로 본다.
DFL 7경기 검증(src/eval_crossing.py): 인플레이 구간 F1 0.845(모델만) → 0.850(1) → 0.877(1+2).
스마트 태깅은 결과 보고에만 쓴다. 상수는 src/config.py의 final 항목에 있다.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]
from config import CONFIG  # noqa: E402
from features import runs_of  # noqa: E402

FPS = CONFIG["fps"]
DEAD_DIR = os.path.join(ROOT, "data", "dead_balls")
F = CONFIG["final"]


# --- 공 신호(라인 아웃 판정) ---

def ball_speed(ball, smooth, gap):
    """공 속도(m/s): smooth프레임 이동평균 위치(모두 관측된 곳만)의 gap프레임 차분."""
    n = len(ball)
    seen = np.isfinite(ball).all(axis=1)
    half = smooth // 2
    values = np.where(seen[:, None], ball, 0.0)
    total = np.vstack((np.zeros(2), np.cumsum(values, axis=0)))
    count = np.concatenate(([0], np.cumsum(seen)))
    index = np.arange(n)
    low, high = np.maximum(index - half, 0), np.minimum(index + half + 1, n)
    full = (count[high] - count[low]) == smooth
    mean = np.full((n, 2), np.nan)
    mean[full] = (total[high] - total[low])[full] / smooth
    speed = np.full(n, np.nan)
    step = gap // 2
    if n > 2 * step:
        diff = mean[2 * step:] - mean[:-2 * step]
        speed[step:-step] = np.hypot(diff[:, 0], diff[:, 1]) * FPS / gap
    return speed


def ball_signals(match):
    """라인 아웃 판정에 쓰는 공 신호: 보임, 경기장 안/밖(경계 여유 포함), 빠르게 움직임, 굴러감, 속도."""
    length, width = match.pitch
    ball = match.ball.astype(np.float64)
    seen = np.isfinite(ball).all(axis=1)
    m = F["boundary_margin_m"]
    with np.errstate(invalid="ignore"):
        inside = seen & (ball[:, 0] >= -m) & (ball[:, 0] <= length + m) & (ball[:, 1] >= -m) & (ball[:, 1] <= width + m)
        speed = ball_speed(ball, F["ball_smooth_frames"], F["ball_speed_gap_frames"])
    return {"seen": seen, "inside": inside, "outside": seen & ~inside, "speed": speed,
            "moving": seen & (speed > F["ball_moving_mps"]), "rolling": seen & (speed > F["line_out_moving_mps"])}


def line_crossings(sig):
    """공이 안(보임)에서 다음 프레임 밖(보임)으로 넘은 곳 중, 넘기 직전 1초 안에 공이 움직였고 넘은 뒤
    (a) out_confirm_s 안에 경기장 안에서 빠르게 움직이는 공이 없고 (b) out_min_restart_s 안에 경기장 안에서
    공이 보이지 않는 것(속도 무관). 반환은 밖으로 나간 첫 프레임."""
    n = len(sig["seen"])
    t = np.flatnonzero(sig["inside"][:-1] & sig["outside"][1:])
    if not len(t):
        return t
    rolled = np.concatenate(([0], np.cumsum(sig["rolling"])))
    before = rolled[t + 1] - rolled[np.maximum(t + 1 - FPS, 0)]
    fast = np.concatenate(([0], np.cumsum(sig["inside"] & sig["moving"])))
    confirm = int(F["out_confirm_s"] * FPS)
    returned_fast = fast[np.minimum(t + 2 + confirm, n)] - fast[np.minimum(t + 2, n)]
    any_in = np.concatenate(([0], np.cumsum(sig["inside"])))
    restart = int(F["out_min_restart_s"] * FPS)
    returned_soon = any_in[np.minimum(t + 2 + restart, n)] - any_in[np.minimum(t + 2, n)]
    return t[(before > 0) & (returned_fast == 0) & (returned_soon == 0)] + 1


def viewer_rows(table, blocks):
    """로컬 뷰어(dead_ball_viewer.html)가 읽는 구간 목록 형식. review 열이 출처(kind)다.
    모델에는 단계(회수·세팅·킥)가 없어서 S = 구간 시작, K = R = 구간 끝 다음 프레임으로 둔다."""
    out = []
    for number, row in enumerate(table.itertuples(), start=1):
        stop = int(row.end) + 1
        out.append({"dead_id": number, "restart_type": row.kind, "half": 1 if row.start < blocks[1][0] else 2,
                    "source": row.kind, "t_stop": int(row.start), "t_recover": int(row.start),
                    "t_setup": int(row.start), "t_kick": stop, "t_receive": stop, "dead_len_s": row.seconds,
                    "stop_method": row.kind, "restart_method": "-", "reception_method": "-",
                    "recover_method": "-", "setup_method": "-", "t_line_out": -1, "t_vanish": -1,
                    "t_slowdown": -1, "tag_start": "", "tag_end": "", "review": row.kind})
    return pd.DataFrame(out)


def write_viewer_signals(path, match, sig):
    """뷰어의 그래프용 프레임 신호(frame_signals.csv 형식): 선수 평균 속도, 공 속도, 공 보임, 경기장 안."""
    import features as fx
    speeds = fx.central_speed(match.players.astype(np.float64), np.asarray(match.period),
                              CONFIG["speed_half_window"], CONFIG["max_player_speed"])
    eps = fx._nan_stat(np.nanmean, speeds, 1)
    frame = pd.DataFrame({"frame": np.arange(len(eps)), "eps": np.round(eps, 2),
                          "ball_speed": np.round(sig["speed"], 2), "near": "",
                          "seen": sig["seen"].astype(np.int8), "inside": sig["inside"].astype(np.int8),
                          "slow_thr": ""})
    frame.to_csv(path, index=False)


# --- 구간 처리 ---

def overlaps(start, stop, spans):
    return any(a < stop and start < b for a, b in spans)


def first_in(values, low, high):
    k = np.searchsorted(values, low)
    return int(values[k]) if k < len(values) and values[k] < high else None


def build(model_segments, crossings, window):
    """[(start, stop, kind)]. kind: model | model+out_start(시작을 라인 아웃 프레임으로 옮김)."""
    final = []
    for start, stop in model_segments:
        cross = first_in(crossings, start - window, min(start + window, stop))
        final.append((cross, stop, "model+out_start") if cross is not None and cross != start
                     else (start, stop, "model"))
    return sorted(final)


def force_line_outs(state, crossings, sig, score, period):
    """확실한 라인 아웃은 무조건 데드볼(축구 규칙: 공이 라인을 완전히 넘으면 경기 중단).

    라인 아웃 프레임 c가 인플레이로 판정돼 있으면 c부터 데드로 바꾸고, 그 뒤 '공이 경기장 안에서 빠르게 움직이고(ball_moving_mps) 모델 확률이
    인플레이(0.5 미만)'인 첫 프레임 직전까지 데드로 둔다. 공이 안으로 돌아오기 전에는 재개될 수 없고, 공만 안으로
    던져진 것(선수들은 멈춰 있음)은 재개가 아니다. 같은 하프 안에서만 늘린다. 새 기준값 없음.
    DFL 7경기: 인플레이 구간 F1 0.850 → 0.877, 7경기 모두 개선(src/eval_crossing.py).
    한계: 대회 트래킹은 2D이고 흔들려서, 라인 근처 공이 잠깐 밖으로 찍혔다가 사라지면 가짜 라인 아웃이 될 수 있다.
    이를 막는 안전장치(모델 확률이 0.5를 넘는 순간이 있어야 강제)는 DFL에서 0.867로 손해라 넣지 않았다.
    반환: (새 state, 새로 만든 구간 목록)."""
    out = state.copy()
    # 재개 = 공이 던져지거나 차여 경기장 안에서 움직임(축구 규칙 13·15·16·17조). 안에서 보이기만 하는 것은 재개가
    # 아니다: 스로인 선수가 라인 밖에서 머리 위로 든 공은 2D 투영으로 안쪽에 찍힐 수 있다.
    inside = sig["inside"] & sig["moving"]
    alive_model = score < 0.5
    made = []
    for c in crossings:
        if out[c]:
            continue
        t = c + 1
        while t < len(out) and period[t] == period[c] and not (inside[t] and alive_model[t]):
            t += 1
        out[c:t] = True
        made.append((c, t))
    return out, made


def apply_public_kickoffs(final, first_kickoff, second_start, second_kickoff):
    out = [(a, b, k) for a, b, k in final if not (a < second_kickoff and b > second_start) and a != first_kickoff]
    out.append((second_start, second_kickoff, "public_kickoff"))
    return sorted(out)


def set_piece_tags(blocks):
    """채점 구간 안의 스마트 태깅 SET PIECES 태그의 (시작, 끝) 프레임."""
    from apply_competition import set_piece_tags as tags_of
    return [(t["start"], t["end"]) for t in tags_of(blocks)]


def missed_tag_report(blocks, spans, model_segments, crossings, model_dir, out_dir, clock):
    """최종 구간과 겹치지 않는 SET PIECES 태그를 진단용으로 남긴다(디버깅용. 이걸로 상수를 맞추지 않는다).

    missed_tags.csv / viewer_missed.csv(뷰어: 태그 창 앞 10초 ~ 태그 끝). 열:
      nearest_final_s  가장 가까운 최종 데드 구간까지 거리(초, 태그 창 기준)
      max_score        태그 창 ±5초 안의 모델 프레임 확률 최댓값(평활화 전). 낮으면 모델이 전혀 못 본 것
      crossing         태그 창 앞 30초 ~ 태그 끝 안에 진짜 라인 아웃이 보였는지
    """
    from apply_competition import set_piece_tags as tags_of
    pad = int(CONFIG["tag_pad_s"] * FPS)
    frames = pd.read_csv(os.path.join(model_dir, "frames.csv"), usecols=["score"])["score"].to_numpy()
    rows = []
    for t in tags_of(blocks):
        a, b = t["start"] - pad, t["end"] + pad
        if overlaps(a, b, spans):
            continue
        gaps = [max(s - b, a - e, 0) / FPS for s, e in spans]
        rows.append({"tag_start": t["start"], "tag_end": t["end"], "start_clock": clock(t["start"]), "type": t["type"],
                     "tag_s": round((t["end"] - t["start"]) / FPS, 1),
                     "nearest_final_s": round(min(gaps), 1) if gaps else None,
                     "max_score": round(float(frames[max(a, 0):b].max()), 2),
                     "crossing": first_in(crossings, t["start"] - 30 * FPS, t["end"]) is not None})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "missed_tags.csv"), index=False)
    viewer = pd.DataFrame([{"dead_id": i + 1, "restart_type": r["type"] or "?", "half": 1 if r["tag_start"] < blocks[1][0] else 2,
                            "source": "missed_tag", "t_stop": max(r["tag_start"] - 10 * FPS, 0),
                            "t_recover": r["tag_start"], "t_setup": r["tag_start"], "t_kick": r["tag_end"],
                            "t_receive": r["tag_end"], "dead_len_s": r["tag_s"], "stop_method": "missed_tag",
                            "restart_method": "-", "reception_method": "-", "recover_method": "-", "setup_method": "-",
                            "t_line_out": -1, "t_vanish": -1, "t_slowdown": -1, "tag_start": r["tag_start"],
                            "tag_end": r["tag_end"], "review": "missed_tag"} for i, r in enumerate(rows)])
    viewer.to_csv(os.path.join(out_dir, "viewer_missed.csv"), index=False)
    lines = ["", "== SET PIECES tags with no final dead segment: {} (missed_tags.csv, viewer_missed.csv) ==".format(len(rows))]
    if rows:
        lines.append(table[["type", "tag_s", "nearest_final_s", "max_score", "crossing"]].to_string(index=False))
    return lines


OUT_DIR = os.path.join(DEAD_DIR, "final_model")


def main():
    import competition_data
    from apply_competition import OUT_DIR as model_dir, clock

    out_dir = OUT_DIR
    match, blocks = competition_data.load_competition_match()
    n = len(match.period)
    graded = match.period > 0
    sig = ball_signals(match)
    crossings = line_crossings(sig)
    pad = int(CONFIG["tag_pad_s"] * FPS)
    tags = set_piece_tags(blocks)
    tag_spans = [(a - pad, b + pad) for a, b in tags]

    # 1. 모델 확률로 비터비를 다시 돌린다. 진짜 라인 아웃 프레임에서는 데드로 들어가는 전환 벌점이 없다.
    score = pd.read_csv(os.path.join(model_dir, "frames.csv"), usecols=["score"])["score"].to_numpy(np.float64)
    free = np.zeros(n, dtype=bool)
    free[crossings[crossings < n]] = True
    # 1b. 느린 빌드업 보정(slow_buildup.py, JSH ⑤): 패스·수신·공 동행 구간은 데드로 들어가기 어렵게 한다.
    import slow_buildup
    cross_in = crossings[(crossings > 0) & (crossings < n)]
    cross_in = cross_in[match.period[cross_in] == match.period[cross_in - 1]]
    state = slow_buildup.decode(score, match.players, match.ball, match.pitch, match.period, cross_in)
    # 2. 확실한 라인 아웃은 무조건 데드볼
    state, forced = force_line_outs(state, cross_in, sig, score, match.period)
    state &= graded
    model_segments = [(a0 + a, a0 + b) for a0, b0 in blocks for a, b in runs_of(state[a0:b0])]
    baseline_count = len(pd.read_csv(os.path.join(model_dir, "segments.csv")))
    # 3. 시작을 라인 아웃 프레임에 맞춤, 4. 공개 킥오프
    final = build(model_segments, crossings, int(F["out_start_window_s"] * FPS))
    before = len(final)
    first, second_start, second_kick = F["public_kickoffs"]
    final = apply_public_kickoffs(final, first, second_start, second_kick)

    mask = np.zeros(n, dtype=bool)
    for a, b, _ in final:
        mask[a:b] = True
    mask &= graded
    spans = [(a, b) for a, b, _ in final]
    shifts = [(a - s) / FPS for a, b, k in final if k == "model+out_start" for s, e in model_segments if e == b]
    lines = ["== final dead balls: model ({}) + line-outs + public kickoffs ==".format(CONFIG["default_feature_set"]),
             "model: {} segments ({} before free dead entry at line crossings; crossings seen {})".format(
                 len(model_segments), baseline_count, int(free[graded].sum())),
             "clear line-outs the model had as in play, forced to dead: {}".format(len(forced)),
             "start moved to a visible line crossing: {} (shift s p10/p50/p90 {})".format(
                 len(shifts), " / ".join("{:.1f}".format(v) for v in np.percentile(shifts, [10, 50, 90]))
                 if shifts else "-"),
             "public kickoffs: segments replaced or dropped {}".format(before - len(final) + 1),
             "clear line-outs in graded halves still outside a final dead segment: {}".format(
                 int(sum(1 for c in crossings if c < n and graded[c] and not mask[c]))),
             "final segments shorter than 5 s (diagnostic only): {} (from forced line-outs {})".format(
                 sum(1 for a, b, _ in final if b - a < 5 * FPS),
                 sum(1 for a, b, _ in final if b - a < 5 * FPS and any(fa < b and a < fb for fa, fb in forced))),
             "final: {} segments, dead share {:.1%}".format(
                 sum(len(runs_of(mask[a:b])) for a, b in blocks), mask[graded].mean()),
             "SET PIECES tags in graded halves with a final dead segment (reference only, tags are not ground "
             "truth): {}/{}".format(sum(overlaps(a, b, spans) for a, b in tag_spans), len(tag_spans))]

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"frame": np.arange(n), "dead": mask.astype(np.int8)}).to_csv(
        os.path.join(out_dir, "frames.csv"), index=False)
    table = pd.DataFrame([{"start": a, "end": b - 1, "start_clock": clock(a), "seconds": round((b - a) / FPS, 1),
                           "kind": "forced_line_out" if any(fa < b and a < fb for fa, fb in forced) else k}
                          for a, b, k in final]).sort_values("start")
    table.to_csv(os.path.join(out_dir, "segments.csv"), index=False)
    viewer_rows(table, blocks).to_csv(os.path.join(out_dir, "viewer.csv"), index=False)
    write_viewer_signals(os.path.join(out_dir, "frame_signals.csv"), match, sig)
    lines += missed_tag_report(blocks, spans, model_segments, crossings, model_dir, out_dir, clock)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote {}".format(out_dir))


if __name__ == '__main__':
    main()
