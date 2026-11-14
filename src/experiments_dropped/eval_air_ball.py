"""보류(파이프라인에 넣지 않음): 대회 데이터에서 짧은 라인 아웃 구간이 그대로 남아 되돌렸다.

뜬 공이 라인 밖으로 찍히는 착시를 DFL에서 재현하고, 라인 아웃 처리 방법들을 비교한다.

    python src/experiments_dropped/eval_air_ball.py

대회 트래킹은 2D다. 공 픽셀을 호모그래피(바닥 평면)로 되돌리면 뜬 공은 카메라 -> 공 직선이 땅에 닿는 곳,
즉 실제보다 카메라에서 먼 곳에 찍힌다. DFL 공에는 실제 높이(ball_z, m)가 있으므로 가상 카메라에서 이 투영을
그대로 계산할 수 있다:  지면점 = C + (B - C) * h / (h - z)   (C: 카메라 바닥 위치, h: 카메라 높이, z: 공 높이)

카메라는 하프라인 근처, 한쪽 터치라인 뒤에 둔다(중계·전술 카메라의 일반적인 위치, 추정). 위치는 하나로 정하지
않고 경기마다 범위에서 무작위로 뽑는다(CAMERA_RANGES).

비교하는 라인 아웃 처리(모두 free_entry + force_line_outs, final_dead_balls.py와 같은 함수):
  current        지금 방식
  far_confirm    카메라에서 멀어지는 쪽 라인(반대편 터치라인, 양 골라인)을 넘은 것은 공이 안에서 다시 보이기 전에
                 모델도 데드(확률 0.5 이상)라고 할 때만 인정. 카메라 쪽 터치라인은 높이 착시로 생길 수 없으므로
                 그대로 인정(기하학: 투영은 공을 카메라에서 먼 쪽으로만 민다).
  all_confirm    모든 라인 넘음에 같은 확인(예전에 DFL 진짜 공에서 0.877 -> 0.867로 손해였던 방식, 참고용)
각 처리를 진짜 공 좌표(true)와 투영한 공 좌표(projected)에 모두 적용한다. 새 기준값은 없다(0.5는 모델의 판정 경계).
대회 데이터는 쓰지 않는다.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(HERE)  # src/ — 이 파일은 src/experiments_dropped/ 안에 있다
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "starter")]
from config import CONFIG  # noqa: E402
import compare_smoothing as cs  # noqa: E402
import dfl_labels  # noqa: E402
import dfl_load  # noqa: E402
import features as fx  # noqa: E402
import final_dead_balls as fd  # noqa: E402
import metrics  # noqa: E402
import smoothing  # noqa: E402
import simulate  # noqa: E402
import train_dfl  # noqa: E402

FPS = CONFIG["fps"]
CAMERAS_PER_MATCH = 3
# true: 진짜 공, projected: 높이 투영, projected_dropout: 투영 + 공 누락(simulate.ball_dropout, config 범위)
SOURCES = ("true", "projected", "projected_dropout")
# 카메라: 하프라인에서 x 방향 어긋남(m), 가까운 터치라인 뒤 거리(m), 높이(m). 넓게 잡은 범위(추정).
CAMERA_RANGES = {"dx": (-10.0, 10.0), "back": (10.0, 40.0), "height": (10.0, 30.0)}


def draw_camera(rng, pitch):
    length, width = pitch
    side = rng.integers(2)  # 0: y=0 쪽 터치라인 뒤, 1: y=width 쪽
    back = rng.uniform(*CAMERA_RANGES["back"])
    return {"x": length / 2 + rng.uniform(*CAMERA_RANGES["dx"]),
            "y": -back if side == 0 else width + back,
            "h": rng.uniform(*CAMERA_RANGES["height"])}


def project(ball_xy, z, cam):
    """뜬 공을 카메라 직선을 따라 바닥에 투영한 위치."""
    c = np.array([cam["x"], cam["y"]])
    with np.errstate(invalid="ignore", divide="ignore"):
        k = cam["h"] / (cam["h"] - np.clip(z, 0, None))
    out = c + (ball_xy - c) * k[:, None]
    out[~(z < cam["h"])] = np.nan
    return out


def away_from_camera(c, ball, cam, pitch):
    """라인 넘음 c에서 밖으로 나간 방향이 카메라에서 멀어지는 방향인지(그 라인은 높이 착시가 생길 수 있는 쪽)."""
    length, width = pitch
    x, y = ball[c]
    normal = np.array([int(x > length) - int(x < 0), int(y > width) - int(y < 0)], dtype=np.float64)
    return float(normal @ (ball[c] - np.array([cam["x"], cam["y"]]))) > 0


def confirm(crossings, sig, score, period, keep_without_check):
    """공이 경기장 안에서 다시 보이고 모델도 인플레이라고 하기 전에 모델이 데드라고 한 적이 있는 넘음만 남긴다.
    keep_without_check[i]가 참인 넘음은 확인 없이 남긴다."""
    inside = sig["inside"]
    out = []
    for c, keep in zip(crossings, keep_without_check):
        if keep:
            out.append(c)
            continue
        t = c + 1
        while t < len(score) and period[t] == period[c] and not (inside[t] and score[t] < 0.5):
            t += 1
        if score[c:t].max() >= 0.5:
            out.append(c)
    return np.asarray(out, dtype=np.int64)


def flew_back(crossings, sig, period):
    """넘은 뒤 공이 경기장 안에서 다시 보일 때까지 매 프레임 보였고 계속 굴러가는 속도(line_out_moving_mps)
    이상이었는지. 진짜 아웃이면 재개 전에 공이 멈추거나 손에 들린다(축구 규칙 13·15·16·17조)."""
    seen, inside, rolling = sig["seen"], sig["inside"], sig["rolling"]
    out = np.zeros(len(crossings), dtype=bool)
    for i, c in enumerate(crossings):
        t = c
        while t < len(seen) and period[t] == period[c] and not inside[t]:
            if not (seen[t] and rolling[t]):
                break
            t += 1
        out[i] = t < len(seen) and period[t] == period[c] and inside[t]
    return out


def run(score, period, sig, crossings):
    choice = CONFIG["smooth_choice"]
    free = np.zeros(len(score), dtype=bool)
    free[crossings] = True
    state = smoothing.viterbi(score, period, choice["penalty"], choice["bias"], free_entry=free).astype(bool)
    state, _ = fd.force_line_outs(state, crossings, sig, score, period)
    return state.astype(np.int8)


def main():
    pd.set_option("display.width", 200)
    rng = np.random.default_rng(CONFIG["sim_seed"])
    rows, cross_rows = [], []
    for m in dfl_load.MATCHES:
        tracking, events, meta = dfl_load.load_match(m)
        match = fx.match_from_dfl(tracking, meta)
        z = tracking["ball_z"].to_numpy(np.float64)
        oof = pd.read_parquet(os.path.join(train_dfl.oof_dir(), "{}_sim.parquet".format(m)))
        score, period, label = oof["score"].to_numpy(np.float64), oof["period"].to_numpy(), oof["dead"].to_numpy()
        truth = pd.DataFrame({"period": period, "dead": label})
        spells = dfl_labels.dead_spells(tracking, events)
        for k in range(CAMERAS_PER_MATCH):
            cam = draw_camera(rng, match.pitch)
            drop = (rng.uniform(*CONFIG["sim_ball_missing_share"]), rng.uniform(*CONFIG["sim_ball_gap_mean_s"]))
            for source in SOURCES:
                ball = match.ball if source == "true" else project(match.ball, z, cam)
                if source == "projected_dropout":
                    ball = simulate.ball_dropout(ball, rng, *drop)
                sig = fd.ball_signals(fx.Match(players=match.players, team=match.team, ball=ball,
                                               period=match.period, pitch=match.pitch, frame=match.frame))
                crossings = fd.line_crossings(sig)
                far = np.array([away_from_camera(c, ball, cam, match.pitch) for c in crossings], dtype=bool)
                for c, f in zip(crossings, far):
                    cross_rows.append({"source": source, "far_side": f,
                                       "real": label[min(c + FPS, len(label) - 1)] == 1})
                variants = {"current": crossings,
                            "far_confirm": confirm(crossings, sig, score, period, ~far),
                            "all_confirm": confirm(crossings, sig, score, period, np.zeros(len(crossings), bool)),
                            "far_motion": crossings[~(far & flew_back(crossings, sig, period))],
                            "all_motion": crossings[~flew_back(crossings, sig, period)]}
                for name, cr in variants.items():
                    state = run(score, period, sig, cr)
                    r = cs.score_one(state, truth)
                    seg = metrics.segment_metrics(spells, state, period)
                    rows.append({"match": m, "camera": k, "source": source, "variant": name,
                                 "alive_spell_f1": r["alive_spell_f1"], "crossings": len(cr),
                                 "spell_recall": seg["spell_recall"], "segment_precision": seg["segment_precision"]})
        print("done", m, flush=True)

    cross = pd.DataFrame(cross_rows)
    print("\n== line crossings (7 matches x {} cameras): count and share followed by a real dead ball 1 s later =="
          .format(CAMERAS_PER_MATCH))
    print(cross.groupby(["source", "far_side"])["real"].agg(crossings="size", real_share="mean").round(3)
          .to_string())
    table = pd.DataFrame(rows)
    print("\n== mean over matches and cameras ==")
    print(table.groupby(["source", "variant"])[["alive_spell_f1", "spell_recall", "segment_precision",
                                                "crossings"]].mean().round(4).to_string())
    per = table.groupby(["match", "source", "variant"])["alive_spell_f1"].mean().unstack(["source", "variant"])
    print("\n== per match (alive-spell F1, mean over cameras) ==")
    print(per.round(3).to_string())
    for source, v in [(a, b) for a in SOURCES for b in ("far_confirm", "far_motion", "all_motion")]:
        d = per[(source, v)] - per[(source, "current")]
        print("{}: {} - current: mean {:+.4f}, matches better {} / worse {}".format(
            source, v, d.mean(), int((d > 1e-9).sum()), int((d < -1e-9).sum())))


if __name__ == '__main__':
    main()
