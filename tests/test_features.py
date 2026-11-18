"""features.py를 합성 데이터로 검증한다. 대회 데이터는 읽지 않는다.

    python tests/test_features.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import features as fx  # noqa: E402

FPS = fx.FPS


def toy(n=1000):
    """홈 2명, 원정 2명. 홈 0번은 공과 함께 2 m/s로 움직이고, 나머지는 서 있다."""
    t = np.arange(n)
    ball = np.column_stack((20 + t * 2.0 / FPS, np.full(n, 30.0)))
    players = np.zeros((n, 4, 2), dtype=np.float32)
    players[:, 0] = ball + (0.5, 0.0)
    players[:, 1] = (10.0, 10.0)
    players[:, 2] = ball + (0.0, 3.0)
    players[:, 3] = (90.0, 50.0)
    period = np.where(t < n // 2, 1, 2)
    return fx.Match(players=players, team=np.array([0, 0, 1, 1], dtype=np.int8), ball=ball,
                    period=period, pitch=(105.0, 68.0), frame=t)


def test_speeds_and_team_summaries():
    base = fx.base_features(toy())
    row = base.iloc[200]
    tol = 1e-4  # 선수 좌표는 float32로 저장한다
    assert abs(row.home_speed_max - 2.0) < tol and abs(row.home_speed_mean - 1.0) < tol
    assert abs(row.away_speed_max - 2.0) < tol and row.home_count == 2
    assert abs(row.ball_speed - 2.0) < tol
    assert abs(row.ball_nearest_player - 0.5) < tol and abs(row.ball_nearest_opponent - 3.0) < tol
    assert abs(row.home_width - 16.0) < tol  # 홈 두 선수의 y는 30과 10, 10~90% 범위라 0.8 x 20
    assert np.isnan(base.home_speed_max.iloc[0])  # 앞뒤 차분이 없는 첫 프레임


def test_speed_does_not_cross_halves():
    base = fx.base_features(toy())
    n = len(base)
    assert np.isnan(base.home_speed_max.iloc[n // 2]) and np.isnan(base.home_speed_max.iloc[n // 2 - 1])


def test_short_ball_gaps_are_filled_and_long_ones_are_not():
    match = toy()
    match.ball[100:110] = np.nan  # 0.4초
    match.ball[300:400] = np.nan  # 4초
    base = fx.base_features(match)
    assert base.ball_missing.iloc[105] == 0 and abs(base.ball_speed.iloc[105] - 2.0) < 1e-6
    assert base.ball_missing.iloc[350] == 1 and np.isnan(base.ball_speed.iloc[350])
    assert abs(base.ball_missing_s.iloc[350] - (350 - 299) / FPS) < 1e-9 and base.ball_missing_s.iloc[299] == 0


def test_outside_and_boundary_distance():
    match = toy()
    match.ball[600:650] = (50.0, -2.0)  # 터치라인 밖 2 m
    base = fx.base_features(match)
    assert base.ball_outside.iloc[620] == 1 and abs(base.ball_boundary_distance.iloc[620] + 2.0) < 1e-6
    assert base.ball_outside.iloc[200] == 0 and abs(base.ball_boundary_distance.iloc[200] - (30.0)) < 1e-6


def test_context_shifts_within_halves_only():
    frame = fx.compute_features(toy(), offsets_s=[-1, 1])
    n = len(frame)
    assert "ball_speed@+1s" in frame.columns and "ball_speed@-1s" in frame.columns
    assert frame["home_count@+1s"].iloc[100] == frame["home_count"].iloc[125]
    assert np.isnan(frame["home_count@+1s"].iloc[n // 2 - 5])  # 다음 하프 값을 당겨 오지 않는다
    assert np.isnan(frame["home_count@-1s"].iloc[n // 2 + 5])
    assert len(fx.feature_columns(frame)) == 3 * len([c for c in fx.base_features(toy()).columns
                                                       if c not in ("frame", "period")])


if __name__ == '__main__':
    tests = [(name, value) for name, value in sorted(globals().items()) if name.startswith("test_")]
    for name, test in tests:
        test()
        print("ok  {}".format(name))
    print("\n{} tests passed".format(len(tests)))
