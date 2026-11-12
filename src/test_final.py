"""final_dead_balls.py를 합성 데이터로 검증한다.

    python src/test_final.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as fx  # noqa: E402
import final_dead_balls as fd  # noqa: E402


def toy_match(n=1000):
    players = np.full((n, 2, 2), np.nan, dtype=np.float32)
    players[:, 0] = (50.0, 34.0)
    players[:, 1] = (60.0, 30.0)
    ball = np.tile([52.0, 34.0], (n, 1))
    return fx.Match(players=players, team=np.array([0, 1], dtype=np.int8), ball=ball,
                    period=np.ones(n, dtype=int), pitch=(105.0, 68.0))


def test_line_crossing_real_out_vs_high_ball():
    match = toy_match()
    t = np.arange(1000)
    ball = np.tile([52.0, 34.0], (1000, 1)).astype(float)
    ball[100:200, 1] = 34.0 + (t[100:200] - 100) * 0.4  # 10 m/s 로 터치라인 쪽으로 굴러감 → 185쯤 넘음
    ball[200:400, 1] = 75.0                              # 계속 밖: 진짜 아웃
    ball[500:600, 1] = 34.0 + (t[500:600] - 500) * 0.4  # 다시 넘음
    ball[600:610, 1] = 72.0
    ball[610:700, 1] = 60.0 - (t[610:700] - 610) * 0.4  # 곧 안에서 빠르게 움직임: 높이 뜬 공
    ball[700:, 1] = 30.0
    match.ball = ball
    cross = fd.line_crossings(fd.ball_signals(match))
    assert len(cross) == 1 and 180 <= cross[0] <= 190, cross


def test_build_moves_start_to_line_crossing():
    final = fd.build([(100, 300), (1000, 1200)], np.array([150]), 125)
    assert final == [(150, 300, "model+out_start"), (1000, 1200, "model")]


def test_force_line_out_until_ball_back_in_and_model_alive():
    n = 400
    state = np.zeros(n, dtype=bool)
    score = np.full(n, 0.1)
    inside = np.ones(n, dtype=bool); inside[100:180] = False
    moving = np.ones(n, dtype=bool); moving[180:200] = False  # 안에 들어왔지만 멈춰 있음(재개 아님) → 200에서 재개
    out, made = fd.force_line_outs(state, np.array([100]), {"inside": inside, "moving": moving}, score,
                                   np.ones(n, dtype=int))
    assert made == [(100, 200)] and out[100:200].all() and not out[200:].any()


def test_public_kickoffs():
    out = fd.apply_public_kickoffs([(100, 150, "model"), (980, 1300, "model"), (1250, 1400, "model")], 100, 1000, 1200)
    assert out == [(1000, 1200, "public_kickoff"), (1250, 1400, "model")]


if __name__ == '__main__':
    tests = [(name, value) for name, value in sorted(globals().items()) if name.startswith("test_")]
    for name, test in tests:
        test()
        print("ok  {}".format(name))
    print("\n{} tests passed".format(len(tests)))
