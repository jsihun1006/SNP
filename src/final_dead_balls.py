"""최종 데드볼(파이프라인 2/2): 모델 확률에 라인 아웃 처리와 공개 킥오프를 더한다.

    python src/final_dead_balls.py      # 직접 연 터미널에서 실행할 것(대회 트래킹을 읽는다)

입력: src/apply_competition.py의 결과(data/dead_balls/model_<특징 세트>/frames.csv, segments.csv)와 대회 ATD
출력: data/dead_balls/final_model/ (커밋 안 함)
    frames.csv         frame, dead(최종 판정, 채점 구간만)
    segments.csv       최종 데드볼 구간: start, end(포함), start_clock, seconds,
                       kind(model | model+out_start | forced_line_out | public_kickoff)
    frame_signals.csv  frame, eps(선수 평균 속도), ball_speed, seen, inside — 공 보임 여부 등(Phase 1 내보내기와 로컬 뷰어용)
    viewer.csv         로컬 확인용 뷰어(dead_ball_viewer.html, 저장소에 없음)가 읽는 구간 목록
    keeper_holds.csv, viewer_keeper.csv   인플레이로 둔 골키퍼 쥠 구간(공이 사라짐 ~ 다시 나타남)
    missed_tags.csv, viewer_missed.csv   최종 구간과 겹치지 않는 SET PIECES 태그(진단용)
    penalty_kicks.csv, viewer_penalty.csv   페널티킥 감지(실험, 보기용. 최종 구간은 바꾸지 않는다)
    summary.txt        화면 출력과 같은 집계

순서:
  1. 모델 확률의 2상태 비터비(벌점 40). 진짜 라인 아웃이 보인 프레임에서는 인플레이 → 데드 전환 벌점이 0이다.
     느린 빌드업 보정(slow_buildup.py, JSH ⑤): 패스·수신이나 공과 선수의 동행이 보이는 구간은 데드 진입 비용을
     높이고 데드 쪽 점수를 낮춘다. Metrica 3경기 0.809 → 0.820, SkillCorner 20경기 0.729 → 0.748.
  2. 확실한 라인 아웃은 무조건 데드볼: 모델이 인플레이로 본 라인 아웃 프레임부터, 공이 경기장 안에서 다시 보이고
     모델이 인플레이로 볼 때까지(force_line_outs).
  2b. 페널티킥 순간 데드볼 끊기(split_at_penalty_kicks): 감지된 차는 순간부터 인플레이, 공이 보이면 3초 안의
     다음 라인 아웃까지, 아니면 0.6초. 공개 30경기(src/eval_penalty_split.py, 차는 순간 = 배치 깨짐):
     DFL 3경기 +0.0016~+0.0059, Metrica game2 +0.0039(공 없으면 −0.0003), SkillCorner 1874553 −0.0035
     (그 라벨은 킥 6.4초 전부터 인플레이로 봐서 정의가 다름), 나머지 변화 없음.
  2c. 페널티킥 뒤 92.7초 안에 킥오프(선수 배치, src/kickoff.py)가 감지되면 골로 보고, 킥 직후 인플레이(최대 3초)
     뒤부터 킥오프까지 데드(kickoff.hold_after_penalty_goals). 공을 쓰지 않는다. 공개 30경기(src/eval_penalty_goal_hold.py):
     나빠진 경기 0(공 있음·없음), SkillCorner 1874553 +0.0019, 나머지 변화 없음(모델이 세리머니를 이미 데드로 봄).
  2d. 골키퍼 쥠은 인플레이(keeper_hold.keep_holds_alive): 1초 전에 인플레이였는데 공이 박스 안 골키퍼 2 m 옆에서 사라지고
     (그 전 라인 아웃 없음), 22.8초 안에 재개 자리(골 에어리어·페널티 마크·코너·터치라인·센터)가 아닌 곳에서 다시 나타나면
     그 사이를 인플레이로. 공개 데이터(src/eval_keeper_hold_rule.py): DFL 쥠 구간 공 삭제 +0.0026(4/2),
     SkillCorner +0.0031(8/6), DFL 원본·Metrica 0.
     + 공이 보이는 채로 골키퍼가 굴려 몰고 나가면(박스 안 골키퍼 2 m 안, 굴러가는 속도로 1초 이상, 그 전 22.8초 라인 아웃 없음)
     그 데드 구간 시작부터 몰고 나간 끝까지 인플레이(keeper_carries): DFL 원본 +0.0016, Metrica +0.0014, SkillCorner +0.0004,
     나빠진 경기 0.
     + 위 두 경로로 인정된 쥠이 끝난 뒤 공이 골키퍼 2 m 안에 보이며 멈춰 있으면(내려놓고 둔 공), 공이 떠날 때까지
     이어 붙임(continue_holds, 잡은 순간부터 22.8초 안): DFL 원본 +0.0018(2/0), SkillCorner +0.0011(3/0), 나빠진 경기 0.
     + 마크 근처에서 다시 나타나도 골키퍼가 공 2 m 안이면 인정, v8(공이 골키퍼에게 도착한 순간 기준 경기 중 확인, 도착부터
     인플레이, 바꾼 구간 뒤 5초 안 데드도 인플레이), v9(날아가던 경로가 골키퍼 2 m 안을 지나면 인정, 경기장 밖에서
     다시 보인 공 제외). 근거와 예외는 keeper_hold.V4_ADOPTED의 주석.
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


# --- 페널티킥 순간 데드볼 끊기(2026-11-15) ---
# 페널티킥은 재개다(규칙 14조). 감지된 차는 순간부터 PENALTY_ALIVE_S초는 인플레이, 그 뒤는 원래 판정.
# 공개 실제 페널티킥 7개(DFL 3, SkillCorner 3, Metrica 1)의 킥 → 다음 데드볼 시작: 0.48~1.50초, 중앙값 0.60초.
# 공이 보여서 킥 뒤 PENALTY_OUT_MAX_S초 안에 라인 아웃(골 포함)이 보이면 그 순간까지 인플레이(라인 아웃 = 데드 시작).
# 아니면(공이 안 보이는 등) 중앙값 PENALTY_ALIVE_S초. 고정 0.6초만 쓰면 킥 → 골이 1.5초인 SkillCorner 1874553에서
# 인플레이 구간이 짧아 손해였다(src/eval_penalty_split.py). PENALTY_OUT_MAX_S는 결과 보기 전에 정한 상한:
# 마크에서 찬 공이 골라인에 닿기에 충분하고, 막혀서 경기가 이어지면 그 뒤는 원래 판정에 맡긴다.
PENALTY_ALIVE_S, PENALTY_OUT_MAX_S = 0.6, 3.0


def penalty_kicks(match):
    """감지된 페널티킥 순간 [(frame, method)] (eval_penalty_combo의 v2 두 방식, 5초 안 중복은 앞의 것)."""
    import eval_penalty_combo as pk
    # run_up_v2의 차는 순간 = 배치가 처음 깨진 순간(src/eval_penalty_moment.py: 공개 실제 페널티킥 5개 오차 절댓값
    # 중앙값 0.08초, 마크 도착 기준은 0.20초로 일찍 잡힘. 가짜 감지는 둘 다 0).
    found = sorted([(k, "run_up_v2") for k in pk.run_up_v2(match, "break")]
                   + [(k, "setup_then_kick_v2") for k in pk.setup_then_kick_v2(match)])
    out = []
    for k, method in found:
        if not out or k - out[-1][0] > pk.MERGE_S * FPS:
            out.append((k, method))
    return out


def split_at_penalty_kicks(state, kicks, period, crossings=()):
    """킥 순간이 데드로 판정돼 있으면 킥부터 인플레이로(같은 하프 안). 끝: 킥 뒤 PENALTY_OUT_MAX_S초 안의 첫 라인 아웃,
    없으면 킥 + PENALTY_ALIVE_S. 반환 (새 state, 바꾼 킥)."""
    out = state.copy()
    period = np.asarray(period)
    crossings = np.sort(np.asarray(crossings, dtype=np.int64))
    m, cap = int(round(PENALTY_ALIVE_S * FPS)), int(round(PENALTY_OUT_MAX_S * FPS))
    changed = []
    for k in kicks:
        if 0 <= k < len(out) and out[k] and period[k] > 0:
            c = crossings[(crossings > k) & (crossings <= k + cap)]
            e = min(int(c[0]) if len(c) else k + m, len(out))
            e = k + int(np.argmax(period[k:e] != period[k])) if (period[k:e] != period[k]).any() else e
            out[k:e] = False
            changed.append(int(k))
    return out, changed


# --- 골 뒤 재개까지 데드(실험, 2026-11-15) ---
# 규칙 8조: 골 뒤에는 센터 마크 킥오프로 재개한다. 2D라 크로스바 위로 넘은 공도 골대 사이로 보이므로 골킥(16조)과
# 코너킥(17조) 재개도 끝으로 인정한다. 재개 = 그 자리에 STILL_S초 이상 멈춰 놓였던 공이 빠르게 출발.
# 놓인 공을 요구하는 이유: 세리머니 중 골키퍼가 그물의 공을 차 내거나 선수가 공을 차서 보내는 것은 재개가 아니다.
# 상수는 결과를 보기 전에 정했다. RESTART_TOL_M은 페널티 마크 감지와 같은 좌표 흔들림 허용치.
POST_HALF_M, GOAL_AREA_DEPTH_M, GOAL_AREA_HALF_M, CORNER_ARC_M = 3.66, 5.5, 9.16, 1.0   # 경기 규칙 1조
RESTART_TOL_M, STILL_S = 2.0, 0.5


def goal_crossings(crossings, ball, pitch):
    """라인 아웃 중 골대 사이로 골라인을 넘은 것(넘기 전후 프레임 y 평균). 반환 (프레임, 골 쪽 0: x=0 / 1: x=length)."""
    length, width = pitch
    c = np.asarray(crossings, dtype=np.int64)
    c = c[c > 0]
    x = ball[c, 0]
    y_mid = (ball[c - 1, 1] + ball[c, 1]) / 2
    with np.errstate(invalid="ignore"):
        keep = ((x < 0) | (x > length)) & (np.abs(y_mid - width / 2) <= POST_HALF_M)
    return c[keep], (x[keep] > length).astype(np.int64)


def restart_spots(ball, pitch, side):
    """프레임별로 공이 이 골 뒤의 재개 자리(센터 마크, 그 골의 골 에어리어, 그 골 쪽 코너)에 있는지."""
    length, width = pitch
    bx, by = ball[:, 0].astype(np.float64), ball[:, 1].astype(np.float64)
    depth = bx if side == 0 else length - bx
    t = RESTART_TOL_M
    with np.errstate(invalid="ignore"):
        center = np.hypot(bx - length / 2, by - width / 2) <= t
        goal_area = (depth <= GOAL_AREA_DEPTH_M + t) & (depth >= -t) & (np.abs(by - width / 2) <= GOAL_AREA_HALF_M + t)
        corner = np.hypot(depth, np.abs(by - width / 2) - width / 2) <= CORNER_ARC_M + t
    return [center, goal_area, corner]


def hold_goal_dead(state, crossings, sig, ball, pitch, period):
    """골대 사이 골라인 아웃 c부터 첫 재개(놓인 공이 재개 자리에서 출발) 직전까지 데드. 같은 하프 안에서 재개를
    못 찾으면 바꾸지 않는다. 반환 (새 state, [(c, t)] 바꾼 구간, 재개 못 찾은 수)."""
    out = state.copy()
    n = len(out)
    still = sig["seen"] & (np.nan_to_num(sig["speed"], nan=np.inf) <= F["line_out_moving_mps"])
    go = sig["inside"] & sig["moving"]
    k = int(STILL_S * FPS)
    period = np.asarray(period)
    goals, sides = goal_crossings(crossings, ball, pitch)
    made, missing = [], 0
    for c, side in zip(goals, sides):
        end = c + 1
        while end < n and period[end] == period[c]:
            end += 1
        restart = None
        for spot in restart_spots(ball[c:end], pitch, side):
            placed = np.concatenate(([0], np.cumsum(still[c:end] & spot)))
            idx = np.flatnonzero(go[c:end] & spot)
            idx = idx[idx >= k]
            ok = idx[(placed[idx] - placed[idx - k]) == k]
            if len(ok):
                restart = c + int(ok[0]) if restart is None else min(restart, c + int(ok[0]))
        if restart is None:
            missing += 1
            continue
        out[c:restart] = True
        made.append((int(c), int(restart)))
    return out, made, missing


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


def penalty_kick_report(pen_kicks, blocks, final, out_dir, clock):
    """페널티킥 감지 결과(main 2b에서 최종 구간을 차는 순간에 끊는다. 이 함수는 기록만 한다).

    penalty_kicks.csv / viewer_penalty.csv(뷰어: 차기 15초 전 ~ 차는 순간). 방식은 src/eval_penalty_combo.py:
    배치(v2, formation): 골라인 위 골키퍼 + 마크 8.15 m 안 키커 1명 외에는 박스 깊숙이·마크 근처에 아무도 없음.
    보이는 선수 수 조건은 없다(겹친 선수가 한 명으로 잡히는 대회 트래킹 대비, src/eval_penalty_visible.py).
      run_up_v2           서 있던 키커가 마크에 도착하고, 그 뒤 배치가 깨지는 순간(공 안 씀)
      setup_then_kick_v2  배치가 1초 이상 이어진 뒤 3초 안에 마크 근처에서 공이 골 쪽으로 빠르게 출발
    공개 30경기(실제 페널티킥 DFL 3, SkillCorner 3, Metrica 1): 가짜 감지 둘 다 0. 공이 안 보여도 run_up_v2는 그대로다.
      run_up_v2           DFL 3/3, 시뮬레이션 2/3, Metrica 1/1, SkillCorner 0/3(검출 선수만 1/3)
      setup_then_kick_v2  DFL 3/3, 시뮬레이션 2/3, Metrica 0/1, SkillCorner 1/3(검출 선수만 1/3)
    상수는 대회 장면에 맞추지 않는다.
    열: in_dead(최종 데드 구간 안인지), dead_end_after_s(그 구간이 차는 순간 몇 초 뒤에 끝나는지, 음수면 먼저 끝남)."""
    rows = []
    for k, method in pen_kicks:
        inside = [(a, b) for a, b, _ in final if a <= k < b]
        near = min(final, key=lambda seg: min(abs(seg[0] - k), abs(seg[1] - k)), default=None)
        seg = inside[0] if inside else (near[0], near[1]) if near else None
        rows.append({"kick": int(k), "clock": clock(int(k)), "method": method, "in_dead": bool(inside),
                     "dead_end_after_s": round((seg[1] - k) / FPS, 1) if seg else None})
    rows.sort(key=lambda r: r["kick"])
    table = pd.DataFrame(rows, columns=["kick", "clock", "method", "in_dead", "dead_end_after_s"])
    table.to_csv(os.path.join(out_dir, "penalty_kicks.csv"), index=False)
    lead = 15 * FPS
    pd.DataFrame([{"dead_id": i + 1, "restart_type": "penalty?", "half": 1 if r["kick"] < blocks[1][0] else 2,
                   "source": r["method"], "t_stop": max(r["kick"] - lead, 0), "t_recover": max(r["kick"] - lead, 0),
                   "t_setup": max(r["kick"] - lead, 0), "t_kick": r["kick"], "t_receive": r["kick"],
                   "dead_len_s": lead / FPS, "stop_method": r["method"], "restart_method": "-",
                   "reception_method": "-", "recover_method": "-", "setup_method": "-", "t_line_out": -1,
                   "t_vanish": -1, "t_slowdown": -1, "tag_start": "", "tag_end": "", "review": r["method"]}
                  for i, r in enumerate(rows)]).to_csv(os.path.join(out_dir, "viewer_penalty.csv"), index=False)
    lines = ["", "== penalty kicks detected (dead ball split at the kick): {} (penalty_kicks.csv, "
             "viewer_penalty.csv) ==".format(len(rows))]
    if rows:
        lines.append(table.to_string(index=False))
    return lines


def keeper_hold_report(holds, blocks, out_dir, clock):
    """인플레이로 둔 골키퍼 쥠 구간 기록: keeper_holds.csv, viewer_keeper.csv(뷰어: 공이 사라진 순간 ~ 다시 나타난 순간)."""
    table = pd.DataFrame([{"vanish": a, "reappear": b, "clock": clock(a), "seconds": round((b - a) / FPS, 1)}
                          for a, b in holds], columns=["vanish", "reappear", "clock", "seconds"])
    table.to_csv(os.path.join(out_dir, "keeper_holds.csv"), index=False)
    pd.DataFrame([{"dead_id": i + 1, "restart_type": "keeper_hold", "half": 1 if a < blocks[1][0] else 2,
                   "source": "keeper_hold", "t_stop": a, "t_recover": a, "t_setup": a, "t_kick": b, "t_receive": b,
                   "dead_len_s": (b - a) / FPS, "stop_method": "keeper_hold", "restart_method": "-",
                   "reception_method": "-", "recover_method": "-", "setup_method": "-", "t_line_out": -1,
                   "t_vanish": a, "t_slowdown": -1, "tag_start": "", "tag_end": "", "review": "keeper_hold"}
                  for i, (a, b) in enumerate(holds)]).to_csv(os.path.join(out_dir, "viewer_keeper.csv"), index=False)
    lines = ["", "== goalkeeper holds kept in play: {} (keeper_holds.csv, viewer_keeper.csv) ==".format(len(holds))]
    if len(table):
        lines.append(table.to_string(index=False))
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
    # 2b. 페널티킥 순간 데드볼 끊기(차는 순간은 재개). 공이 보이면 다음 라인 아웃까지, 아니면 0.6초 인플레이.
    pen_kicks = [(k, m) for k, m in penalty_kicks(match) if graded[k]]
    state, pen_split = split_at_penalty_kicks(state, [k for k, _ in pen_kicks], match.period, cross_in)
    # 2c. 페널티킥 뒤 킥오프가 감지되면 골: 킥 직후 인플레이 뒤부터 킥오프까지 데드(세리머니 중 끊김 방지, 공 안 씀)
    import kickoff
    state, pen_goals = kickoff.hold_after_penalty_goals(state, [k for k, _ in pen_kicks], kickoff.kickoffs(match),
                                                        match.period)
    state &= graded
    # 2d. 골키퍼 쥠은 인플레이: 인플레이 중 공이 박스 안 골키퍼 옆에서 사라졌다가 같은 박스의 재개 자리가 아닌 곳에서
    # 다시 나타나면 그 사이를 인플레이로(keeper_hold.py)
    import keeper_hold
    state, keeper_holds = keeper_hold.keep_holds_alive(state, match, sig, cross_in)
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
             "penalty kicks detected {}, dead ball split at the kick {}, held dead until the kickoff after a "
             "penalty goal {}".format(len(pen_kicks), len(pen_split), len(pen_goals)),
             "goalkeeper holds kept in play: {} ({:.1f} s)".format(
                 len(keeper_holds), sum(b - a for a, b in keeper_holds) / FPS),
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
    lines += penalty_kick_report(pen_kicks, blocks, final, out_dir, clock)
    lines += keeper_hold_report(keeper_holds, blocks, out_dir, clock)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote {}".format(out_dir))


if __name__ == '__main__':
    main()
