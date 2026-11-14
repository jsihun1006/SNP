"""데드볼 모델의 오류 분석과 후보 특징 점검(DFL 7경기, 경기 하나씩 빼고 학습한 확률 사용).

    python src/analyze_dead_errors.py

1. 오류: 비터비 평활화 뒤 가짜 데드볼 구간(정답은 인플레이)과 놓친 데드볼 구간의 개수·길이·원인.
2. 후보 특징: 모델이 헷갈리는 두 프레임 묶음을 얼마나 가르는지 AUC로 본다.
     가짜 데드 프레임(정답 인플레이, 예측 데드)  대  진짜 데드 프레임(정답 데드, 예측 데드)
   AUC가 0.5에서 멀수록 그 특징이 두 경우를 가른다. 기존 특징도 같은 방식으로 재서 비교한다.
   후보는 모두 기준값 없이 위치에서 바로 계산한다(팀 이동 방향 일치도, 팀 중심 속력, 선수 가속도 분위수).
대회 데이터는 쓰지 않는다. 평가 경기는 기본 노이즈 버전(지금 DFL 평가와 같음).
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import CONFIG  # noqa: E402
import dfl_labels  # noqa: E402
import dfl_load  # noqa: E402
import features as fx  # noqa: E402
import metrics  # noqa: E402
import simulate  # noqa: E402
import smoothing  # noqa: E402
import train_dfl  # noqa: E402

FPS = CONFIG["fps"]
BINS_S = [0, 2, 5, 10, 20, 40, np.inf]


def bin_name(seconds):
    for low, high in zip(BINS_S[:-1], BINS_S[1:]):
        if low <= seconds < high:
            return "{}-{}s".format(low, "inf" if np.isinf(high) else high)


def candidate_features(match):
    """기준값 없는 후보 특징(프레임별)."""
    players = match.players.astype(np.float64)
    period = np.asarray(match.period)
    half = CONFIG["speed_half_window"]
    vel = np.full(players.shape, np.nan)
    step = players[2 * half:] - players[:-2 * half]
    same = (period[2 * half:] == period[:-2 * half])[:, None, None]
    vel[half:-half] = np.where(same, step * FPS / (2.0 * half), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.hypot(vel[..., 0], vel[..., 1])
        vel[speed > CONFIG["max_player_speed"]] = np.nan
        speed = np.hypot(vel[..., 0], vel[..., 1])
        unit = vel / speed[..., None]
    out = {}
    for code, name in ((0, "home"), (1, "away")):
        mask = match.team == code
        # 이동 방향 일치도: 단위 방향 벡터 평균의 길이(1 = 모두 같은 방향, 0 = 제각각)
        u = unit[:, mask]
        out[name + "_dir_coherence"] = np.hypot(fx._nan_stat(np.nanmean, u[..., 0], 1),
                                                fx._nan_stat(np.nanmean, u[..., 1], 1))
        # 팀 중심(평균 위치)의 속력 = 선수 속도 벡터 평균의 크기
        v = vel[:, mask]
        out[name + "_centroid_speed"] = np.hypot(fx._nan_stat(np.nanmean, v[..., 0], 1),
                                                 fx._nan_stat(np.nanmean, v[..., 1], 1))
    # 선수 가속도(속력 변화율)의 분포
    acc = np.full(speed.shape, np.nan)
    acc[half:-half] = (speed[2 * half:] - speed[:-2 * half]) * FPS / (2.0 * half)
    acc[np.r_[np.zeros(half, bool), ~(period[2 * half:] == period[:-2 * half]), np.zeros(half, bool)]] = np.nan
    for q in (10, 50, 90):
        out["all_acc_p{}".format(q)] = fx.nan_percentile(acc, q)
    out["all_dir_coherence"] = np.hypot(fx._nan_stat(np.nanmean, unit[..., 0], 1),
                                        fx._nan_stat(np.nanmean, unit[..., 1], 1))
    return pd.DataFrame(out)


def auc(pos, neg):
    """pos가 neg보다 클 확률(동점 0.5). NaN 제외."""
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    values = np.concatenate([pos, neg])
    ranks = pd.Series(values).rank().to_numpy()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def main():
    pd.set_option("display.width", 200)
    train_dfl.set_feature_set(CONFIG["default_feature_set"])
    choice = CONFIG["smooth_choice"]
    false_segs, missed, fp_frames, tp_frames = [], [], [], []
    for i, m in enumerate(dfl_load.MATCHES):
        tracking, events, meta = dfl_load.load_match(m)
        oof = pd.read_parquet(os.path.join(train_dfl.oof_dir(), "{}_sim.parquet".format(m)))
        score, period, label = oof["score"].to_numpy(np.float64), oof["period"].to_numpy(), oof["dead"].to_numpy()
        state = smoothing.viterbi(score, period, choice["penalty"], choice["bias"]).astype(bool)
        spells = dfl_labels.dead_spells(tracking, events)
        seg = metrics.segment_metrics(spells, state, period)
        for found, s in zip(seg["found"], spells.itertuples()):
            if not found:
                missed.append({"match": m, "length": bin_name(s.seconds), "cause": s.cause})
        for p in np.unique(period):
            rows = np.flatnonzero(period == p)
            for a, b in fx.runs_of(state[rows]):
                s0, s1 = rows[a], rows[b - 1] + 1
                if label[s0:s1].mean() < 0.5:
                    false_segs.append({"match": m, "length": bin_name((s1 - s0) / FPS),
                                       "seconds": (s1 - s0) / FPS})
        # 후보 특징과 기존 특징을 프레임 묶음별로 모은다(5프레임마다)
        match = simulate.simulate(fx.match_from_dfl(tracking, meta), seed=CONFIG["sim_seed"])
        cand = candidate_features(match)
        base = train_dfl.match_features(m, "sim", CONFIG["sim_seed"])
        keep = [c for c in base.columns if "@" not in c and c not in ("period", "dead")]
        both = pd.concat([cand.reset_index(drop=True), base[keep].reset_index(drop=True)], axis=1)
        fp = state & (label == 0)
        tp = state & (label == 1)
        fp_frames.append(both[fp].iloc[::5])
        tp_frames.append(both[tp].iloc[::5])
        print("done", m, flush=True)

    fs = pd.DataFrame(false_segs)
    ms = pd.DataFrame(missed)
    print("\n== false dead segments (prediction dead, truth mostly in play): {} in 7 matches ==".format(len(fs)))
    print(fs["length"].value_counts().reindex([bin_name(x + 1e-9) for x in BINS_S[:-1]]).fillna(0).astype(int)
          .to_string())
    print("\n== missed true dead spells: {} ==".format(len(ms)))
    print(ms.groupby(["cause", "length"]).size().unstack(fill_value=0).to_string())

    fp_all, tp_all = pd.concat(fp_frames), pd.concat(tp_frames)
    print("\n== frames predicted dead: false {} vs true {} (every 5th frame) ==".format(len(fp_all), len(tp_all)))
    rows = []
    for c in fp_all.columns:
        a = auc(fp_all[c].to_numpy(np.float64), tp_all[c].to_numpy(np.float64))
        rows.append({"feature": c, "auc_false_vs_true_dead": a, "separation": abs(a - 0.5),
                     "median_false": np.nanmedian(fp_all[c]), "median_true": np.nanmedian(tp_all[c]),
                     "candidate": c in set(cand_cols())})
    table = pd.DataFrame(rows).sort_values("separation", ascending=False)
    print(table.round(3).to_string(index=False))


def cand_cols():
    return ["home_dir_coherence", "away_dir_coherence", "home_centroid_speed", "away_centroid_speed",
            "all_acc_p10", "all_acc_p50", "all_acc_p90", "all_dir_coherence"]


if __name__ == '__main__':
    main()
