"""broadcast(중계) 트래킹에서 공이 안 보이는 구간을 집계한다. tactical과 비교한다(집계값만 출력).

    python src/broadcast_ball_gaps.py      # 직접 연 터미널에서 실행할 것(대회 데이터를 읽는다)

broadcast는 tactical과 시간이 맞지 않고(DATA.md) 리플레이·클로즈업이 섞여 있어서, 호모그래피가 없는(valid = 0)
프레임에서는 공 위치가 나올 수 없다. 그래서 세 가지로 나눠 센다.
  1. 전체 프레임 중 공이 안 보이는 비율
  2. 경기장이 화면에 나오는 프레임(호모그래피 valid = 1) 중 공이 안 보이는 비율
  3. 2의 프레임 안에서 공이 연속으로 안 보이는 구간의 길이 분포
tactical은 채점 구간(전·후반)만, broadcast는 채점 구간을 알 수 없어 전체 프레임을 센다(하프타임 등 포함).
프레임 번호는 출력하지 않는다.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]
from config import CONFIG  # noqa: E402
from competition_data import read_tracking  # noqa: E402
from features import runs_of  # noqa: E402

FPS = CONFIG["fps"]
BUCKETS_S = [0, 0.5, 1, 2, 3, 5, 10, 30, np.inf]


def ball_seen(source):
    from games import GAME, GAMES, require, source_frames
    _, raw_path = require(["atd_meta", "atd_raw"], source=source)
    length, width = GAMES[GAME]["pitch"]
    n = source_frames(source)
    _, ball = read_tracking(raw_path, 0, n, length, width)
    return np.isfinite(ball).all(axis=1)


def homography_valid(source, n):
    from load_homography import load
    _, _, valid = load(source=source)
    valid = np.asarray(valid).astype(bool)
    out = np.zeros(n, dtype=bool)
    out[:min(n, len(valid))] = valid[:n]
    return out


def gap_lines(seen, mask):
    """mask 안에서 공이 연속으로 안 보이는 구간. mask가 끊기면 구간도 끊는다."""
    lengths = []
    for a, b in runs_of(mask):
        lengths += [(y - x) / FPS for x, y in runs_of(~seen[a:b])]
    lengths = np.array(lengths)
    if not len(lengths):
        return ["  no gaps"]
    total = lengths.sum()
    lines = ["  gaps {}, length s p50 {:.1f} p90 {:.1f} p99 {:.1f} max {:.0f}".format(
        len(lengths), *np.percentile(lengths, [50, 90, 99]), lengths.max()),
        "  by gap length: gaps / share of all missing time"]
    for low, high in zip(BUCKETS_S[:-1], BUCKETS_S[1:]):
        k = (lengths >= low) & (lengths < high)
        lines.append("    {:>4}-{:<4}s  {:6d}  {:5.1%}".format(low, "inf" if np.isinf(high) else high, int(k.sum()),
                                                            lengths[k].sum() / total))
    return lines


def main():
    from games import GAME, half_video_blocks
    lines = []
    # tactical(기준): 채점 구간만
    seen = ball_seen("tactical")
    graded = np.zeros(len(seen), dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    lines += ["== tactical (graded halves, {} frames) ==".format(int(graded.sum())),
              "ball missing {:.1%} ({:.1f} min)".format((~seen[graded]).mean(), (~seen[graded]).sum() / FPS / 60)]
    lines += gap_lines(seen, graded)

    # broadcast: 전체 프레임, 그리고 경기장이 보이는 프레임만
    seen = ball_seen("broadcast")
    valid = homography_valid("broadcast", len(seen))
    lines += ["", "== broadcast (all {} frames, not aligned to tactical) ==".format(len(seen)),
              "ball missing, all frames: {:.1%} ({:.1f} min)".format((~seen).mean(), (~seen).sum() / FPS / 60),
              "frames with no pitch view (homography invalid): {:.1%}; ball missing there {:.1%}".format(
                  (~valid).mean(), (~seen[~valid]).mean() if (~valid).any() else float("nan")),
              "ball missing while the pitch is on screen (homography valid): {:.1%} of {} frames ({:.1f} min)".format(
                  (~seen[valid]).mean(), int(valid.sum()), (~seen[valid]).sum() / FPS / 60),
              "gaps while the pitch is on screen:"]
    lines += gap_lines(seen, valid)
    print("\n".join(lines))


if __name__ == '__main__':
    main()
