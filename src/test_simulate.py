"""simulate.py를 합성 데이터로 검증한다.

    python src/test_simulate.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as fx  # noqa: E402
import simulate as sm  # noqa: E402


def toy(n=20000, slots=22):
    rng = np.random.default_rng(0)
    players = (rng.uniform(0, 60, size=(1, slots, 2)) + np.zeros((n, 1, 1))).astype(np.float32)
    ball = np.full((n, 2), 50.0)
    team = np.array([0] * (slots // 2) + [1] * (slots - slots // 2), dtype=np.int8)
    return fx.Match(players=players, team=team, ball=ball, period=np.ones(n, dtype=int), pitch=(105.0, 68.0))


def test_dropout_reaches_target_in_runs():
    out = sm.simulate(toy(), target_players=20.5, dropout_mean_s=4.0, jitter_m=0.0)
    count = np.isfinite(out.players[..., 0]).sum(axis=1)
    assert abs(count.mean() - 20.5) < 0.4, count.mean()
    gaps = [b - a for slot in range(22) for a, b in fx.runs_of(~np.isfinite(out.players[:, slot, 0]))]
    assert np.median(gaps) > 10  # 프레임 단위가 아니라 구간 단위로 지운다


def test_jitter_is_small_and_smooth():
    match = toy()
    out = sm.simulate(match, target_players=22, jitter_m=0.15, smooth_s=1.0)
    diff = out.players.astype(np.float64) - match.players
    assert abs(diff.std() - 0.15) < 0.02
    # 특징 계산과 같은 0.16초 차분 속도. 서 있는 선수에게 흔들림이 큰 가짜 속도를 만들지 않는다.
    speed = fx.central_speed(out.players[:, :1].astype(np.float64), out.period, 2, 12.0)[:, 0]
    assert np.nanmedian(speed) < 0.3, np.nanmedian(speed)


def test_same_seed_same_result_and_input_untouched():
    match = toy(2000)
    before = match.players.copy()
    a, b = sm.simulate(match, seed=3), sm.simulate(match, seed=3)
    assert np.array_equal(np.isnan(a.players), np.isnan(b.players))
    assert np.array_equal(match.players, before)


if __name__ == '__main__':
    tests = [(name, value) for name, value in sorted(globals().items()) if name.startswith("test_")]
    for name, test in tests:
        test()
        print("ok  {}".format(name))
    print("\n{} tests passed".format(len(tests)))
