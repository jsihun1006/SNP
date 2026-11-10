"""Metrica Sports 공개 샘플(github.com/metrica-sports/sample-data) 1·2번 경기를 공통 포맷으로 읽는다.

원본 CSV는 data/external/metrica/raw/에 둔다(커밋 금지). 읽은 결과는 같은 폴더 위에 parquet로 캐시한다.
대회 데이터와 같은 회사의 형식이고, 공이 안 보이는 비율(약 39%)도 대회(40.7%)와 비슷하다.

데드볼 라벨은 이벤트로 만든다(사람이 단 이벤트라 프레임이 조금 부정확할 수 있다):
    재개 = SET PIECE(스로인, 프리킥, 코너킥, 킥오프)와 GOAL KICK 이벤트의 시작 프레임
    시작 = 그 재개 직전 이벤트의 끝 프레임(CARD와 재개와 같은 프레임의 이벤트 제외). BALL OUT이면 공이 라인을 넘은 프레임, 반칙이면 반칙 프레임.
    데드볼 = 시작 <= 프레임 < 재개
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import CONFIG  # noqa: E402
from features import Match  # noqa: E402

GAMES = [1, 2]
RAW = os.path.join(CONFIG["dfl_cache"], "..", "metrica", "raw")
CACHE = os.path.join(CONFIG["dfl_cache"], "..", "metrica")
PITCH = (105.0, 68.0)  # 샘플 데이터에 크기가 없다. 표준 경기장으로 둔다(추정).
SKIP_BEFORE_RESTART = {"CARD"}


def _tracking(game, side):
    path = os.path.join(RAW, "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(game, side))
    head = pd.read_csv(path, nrows=3, header=None)
    names = head.iloc[2].tolist()
    frame = pd.read_csv(path, skiprows=3, header=None)
    columns, k = ["period", "frame", "time"], 3
    while k < len(names):
        name = names[k]
        if isinstance(name, str) and name.startswith("Player"):
            columns += [name + "_x", name + "_y"]
        elif name == "Ball":
            columns += ["ball_x", "ball_y"]
        k += 2
    frame.columns = columns[:frame.shape[1]]
    return frame


def load_game(game, refresh=False):
    """(tracking, events). tracking: period, frame, ball_x/y, <Player>_x/y(0~1 정규화), team 정보는 열 이름 앞의 h_/a_."""
    path = os.path.join(CACHE, "game{}_tracking.parquet".format(game))
    events = pd.read_csv(os.path.join(RAW, "Sample_Game_{}_RawEventsData.csv".format(game)))
    if os.path.exists(path) and not refresh:
        return pd.read_parquet(path), events
    home, away = _tracking(game, "Home"), _tracking(game, "Away")
    keep = home[["period", "frame", "ball_x", "ball_y"]].copy()
    for prefix, side in (("h_", home), ("a_", away)):
        for c in side.columns:
            if c.startswith("Player"):
                keep[prefix + c] = side[c].to_numpy()
    os.makedirs(CACHE, exist_ok=True)
    keep.to_parquet(path)
    return keep, events


def match_from_metrica(tracking):
    length, width = PITCH
    ids = [c[:-2] for c in tracking.columns if c.endswith("_x") and c[:2] in ("h_", "a_")]
    players = np.stack([np.stack([tracking[i + "_x"].to_numpy(np.float64) * length,
                                  tracking[i + "_y"].to_numpy(np.float64) * width], axis=1) for i in ids], axis=1)
    team = np.array([0 if i.startswith("h_") else 1 for i in ids], dtype=np.int8)
    ball = np.stack([tracking["ball_x"].to_numpy(np.float64) * length,
                     tracking["ball_y"].to_numpy(np.float64) * width], axis=1)
    return Match(players=players.astype(np.float32), team=team, ball=ball, period=tracking["period"].to_numpy(),
                 pitch=PITCH, frame=tracking["frame"].to_numpy())


def dead_spells(tracking, events):
    """DataFrame[start, stop(half-open 행 번호), period, seconds, cause, stop_event]."""
    row_of = {int(f): i for i, f in enumerate(tracking["frame"].to_numpy())}
    # 골킥은 SET PIECE가 아니라 PASS/GOAL KICK 또는 BALL LOST/GOAL KICK-INTERCEPTION으로 기록된다.
    restart = events["Type"].eq("SET PIECE") | events["Subtype"].fillna("").str.startswith("GOAL KICK")
    rows = []
    for i in np.flatnonzero(restart.to_numpy()):
        event = events.iloc[i]
        before = events.iloc[:i]
        # 재개와 같은 프레임의 이벤트(골킥 직전 RECOVERY 등)는 재개의 일부다.
        before = before[~before["Type"].isin(SKIP_BEFORE_RESTART) & before["Period"].eq(event["Period"])
                        & ~(before["Type"].eq("SET PIECE")) & (before["Start Frame"] < event["Start Frame"])]
        if not len(before):
            continue  # 하프 첫 킥오프
        start_frame, stop_frame = int(before["End Frame"].iloc[-1]), int(event["Start Frame"])
        if stop_frame - start_frame < CONFIG["fps"] // 2 or start_frame not in row_of or stop_frame not in row_of:
            continue
        cause = str(event["Subtype"]).lower().split("-")[0].replace(" ", "_")
        rows.append((row_of[start_frame], row_of[stop_frame], int(event["Period"]),
                     (stop_frame - start_frame) / float(CONFIG["fps"]), cause, str(before["Type"].iloc[-1])))
    return pd.DataFrame(rows, columns=["start", "stop", "period", "seconds", "cause", "stop_event"])


def dead_label(tracking, spells):
    label = np.zeros(len(tracking), dtype=np.int8)
    for a, b in zip(spells["start"], spells["stop"]):
        label[a:b] = 1
    return label


if __name__ == '__main__':
    for g in GAMES:
        tracking, events = load_game(g)
        spells = dead_spells(tracking, events)
        label = dead_label(tracking, spells)
        print("game {}: {} frames, {} player slots, dead spells {}, dead share {:.1%}, ball missing {:.1%}".format(
            g, len(tracking), sum(c.endswith("_x") and c[:2] in ("h_", "a_") for c in tracking.columns),
            len(spells), label.mean(), tracking["ball_x"].isna().mean()))
        print("  spell length s p10/p50/p90", np.percentile(spells["seconds"], [10, 50, 90]).round(1),
              "| causes", spells["cause"].value_counts().to_dict())
        print("  stop events", spells["stop_event"].value_counts().to_dict())
