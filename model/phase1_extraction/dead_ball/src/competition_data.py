"""대회 데이터(Metrica ATD)를 features.py의 공통 포맷으로 읽는다.

직접 연 터미널에서 실행할 것. 대회 데이터와 거기서 나온 파일은 커밋하거나 호스팅 LLM에 올리지 않는다.
읽기는 아래 read_tracking이 한다.
좌표는 0~1 정규화 값에 피치 크기를 곱해 미터로,
트랙 슬롯(46개, 사람이 아니라 조각)의 팀은 메타데이터에서, 하프는 games.half_video_blocks의 채점 구간에서.
채점 구간 밖 프레임은 period 0으로 둔다.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = __import__("repo_root").ROOT
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]
from features import Match  # noqa: E402


def read_tracking(raw_path, n_slots, n_frames, length, width):
    """(players, ball): (n, slots, 2) m, (n, 2) m. 원본 한 줄 = 'frame_id:선수들;...:공'. video_frame = frame_id - 1."""
    from baseline import FIELD_SEP, GROUP_SEP
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            fields = parts[2].strip().split(",")
            if len(fields) >= 2 and fields[0] and fields[0][0] != "N":
                ball[frame] = float(fields[0]) * length, float(fields[1]) * width
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                fields = group.strip().split(",")
                if slot >= n_slots or len(fields) < 2 or not fields[0] or fields[0][0] == "N":
                    continue
                players[frame, slot] = float(fields[0]) * length, float(fields[1]) * width
    return players, ball


def load_competition_match():
    """(Match, half_blocks). half_blocks는 [(start, stop)] 채점 구간."""
    from baseline import team_of_each_track
    from games import GAME, GAMES, half_video_blocks, require, source_frames

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = team_of_each_track(meta_path)
    n = source_frames("tactical")
    length, width = GAMES[GAME]["pitch"]
    players, ball = read_tracking(raw_path, len(team), n, length, width)
    blocks = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]
    period = np.zeros(n, dtype=np.int64)
    for number, (start, stop) in enumerate(blocks, start=1):
        period[start:stop] = number
    return Match(players=players, team=team, ball=ball, period=period,
                 pitch=tuple(GAMES[GAME]["pitch"]), frame=np.arange(n)), blocks
