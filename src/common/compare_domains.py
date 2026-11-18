"""두 데이터셋의 선수 특징 분포를 같은 형식으로 요약한다. 시뮬레이션이 대회 데이터에 가까워졌는지 비교하는 용도다.

    python src/compare_domains.py dfl J03WPY          # DFL 원본과 시뮬레이션 (Claude가 실행해도 된다)
    python src/compare_domains.py competition         # 대회 데이터. 직접 연 터미널에서 실행할 것

출력은 경기 전체를 요약한 집계값뿐이다(분위수, 평균). 프레임 단위 값은 내보내지 않는다.
대회 데이터는 채점 구간(period 1, 2)만 요약한다.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
from config import CONFIG  # noqa: E402
import features as fx  # noqa: E402

FPS = CONFIG["fps"]
QUANTILES = [0.1, 0.5, 0.9]


def presence_runs(match):
    """선수 슬롯이 연속으로 보인 구간과 안 보인 구간의 길이(초). 하프 안에서만 센다."""
    shown, hidden = [], []
    valid = match.period > 0
    for slot in range(match.players.shape[1]):
        present = np.isfinite(match.players[:, slot, 0]) & valid
        for a, b in fx.runs_of(present):
            shown.append((b - a) / FPS)
        # 처음 나타나기 전, 마지막으로 사라진 뒤는 교체·트랙 시작 전후라 빼고, 중간의 빈 구간만 센다
        idx = np.flatnonzero(present)
        if len(idx):
            for a, b in fx.runs_of(~present[idx[0]:idx[-1] + 1]):
                hidden.append((b - a) / FPS)
    return np.array(shown), np.array(hidden)


def summary(match, label):
    base = fx.base_features(match)
    base = base[base["period"] > 0]
    rows = {}
    for column in CONFIG["player_features"] + CONFIG["feature_sets"]["v2"]:
        values = base[column].dropna()
        rows[column] = [values.mean()] + list(values.quantile(QUANTILES)) + [base[column].isna().mean()]
    table = pd.DataFrame(rows, index=["mean", "p10", "p50", "p90", "nan"]).T.round(2)
    shown, hidden = presence_runs(match)
    count = np.isfinite(match.players[match.period > 0, :, 0]).sum(axis=1)
    lines = ["== {} ==".format(label),
             "players per frame: mean {:.1f}, p10 {:.0f}, p50 {:.0f}, p90 {:.0f}".format(
                 count.mean(), *np.percentile(count, [10, 50, 90])),
             "track pieces: {} visible runs, length s p10 {:.1f} p50 {:.1f} p90 {:.1f}".format(
                 len(shown), *(np.percentile(shown, [10, 50, 90]) if len(shown) else [np.nan] * 3)),
             "gaps inside a track: {} , length s p10 {:.1f} p50 {:.1f} p90 {:.1f}".format(
                 len(hidden), *(np.percentile(hidden, [10, 50, 90]) if len(hidden) else [np.nan] * 3)),
             table.to_string()]
    return "\n".join(lines)


if __name__ == '__main__':
    pd.set_option("display.width", 200)
    if sys.argv[1:2] == ["competition"]:
        import competition_data
        match, _ = competition_data.load_competition_match()
        print(summary(match, "competition (graded frames)"))
    else:
        import dfl_load
        import simulate
        match_id = sys.argv[2] if len(sys.argv) > 2 else "J03WPY"
        tracking, _, meta = dfl_load.load_match(match_id)
        raw = fx.match_from_dfl(tracking, meta)
        print(summary(raw, "DFL {} raw".format(match_id)))
        print()
        print(summary(simulate.simulate(raw), "DFL {} simulated".format(match_id)))
