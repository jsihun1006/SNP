"""Sportec Open DFL(IDSSE) 경기를 불러와 parquet로 캐시한다.

    python src/dfl_load.py J03WPY          # 한 경기
    python src/dfl_load.py all             # 7경기 전부

출처: Bassek et al. (2025), Scientific Data 12, 195 (doi 10.1038/s41597-025-04505-y). CC-BY 4.0, DFL 표기.
원본: figshare doi 10.6084/m9.figshare.28196177. 경기마다 경기 정보, 이벤트, 트래킹 XML을 제공한다.
config의 dfl_raw 폴더(기본 data/external/dfl/raw, 하위 폴더 허용)에 있는 .xml.gz 또는 .xml을 읽는다.
로컬 파일이 없으면 kloppy가 공개 데이터 다운로드를 시도한다.
변환한 parquet는 config의 dfl_cache 폴더(기본 data/external/dfl, 커밋 안 함)에 둔다. 대회 데이터와 섞지 않는다.

트래킹은 only_alive=False로 읽어 데드볼 프레임도 포함한다. 프레임별 ball_state(ALIVE/DEAD)는
TRACAB의 BallStatus(1/0)에서 온다. 경기장 관찰자가 실시간으로 기록한 값이라 경계에 몇 프레임
지연이 있을 수 있다.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
from config import CONFIG  # noqa: E402

MATCHES = ["J03WPY", "J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WQQ", "J03WR9"]


def cache_paths(match_id):
    folder = CONFIG["dfl_cache"]
    return (os.path.join(folder, "{}_tracking.parquet".format(match_id)),
            os.path.join(folder, "{}_events.parquet".format(match_id)),
            os.path.join(folder, "{}_meta.parquet".format(match_id)))


def raw_files(match_id):
    """경기 정보, 이벤트, 트래킹 XML의 로컬 경로. 둘 다 있으면 비압축 XML을 우선한다."""
    folder = CONFIG["dfl_raw"]
    if not os.path.isdir(folder):
        return None
    prefixes = {"meta": "DFL_02_01_matchinformation", "event": "DFL_03_02_events_raw",
                "tracking": "DFL_04_03_positions_raw_observed"}
    found = {}
    for root, _, names in os.walk(folder):
        for name in names:
            if not name.endswith((".xml", ".xml.gz")) or f"DFL-MAT-{match_id}" not in name:
                continue
            for key, prefix in prefixes.items():
                if name.startswith(prefix) and (key not in found or name.endswith(".xml")):
                    found[key] = os.path.join(root, name)
    return found if len(found) == 3 else None


def load_match(match_id, refresh=False):
    """(tracking, events, meta) DataFrame. 캐시가 있으면 읽고, 없으면 내려받아 만든다.

    tracking 열: frame_id, period, timestamp_s, ball_state(1 인플레이 / 0 데드볼), ball_owning_team,
    ball_x, ball_y, ball_z, 그리고 선수마다 <player_id>_x, <player_id>_y.
    meta: player_id, team(home/away), pitch_length, pitch_width.
    좌표는 kloppy 기본 좌표계로 0~1 정규화돼 있다(확인: 첫 프레임 공 x 0.56, y 0.50). 대회 데이터와 같은 방식이라
    features.py에서 피치 크기(meta의 pitch_length × pitch_width, 105 × 68 m)를 곱해 미터로 바꾼다.
    """
    tracking_path, events_path, meta_path = cache_paths(match_id)
    if not refresh and all(os.path.exists(p) for p in (tracking_path, events_path, meta_path)):
        return pd.read_parquet(tracking_path), pd.read_parquet(events_path), pd.read_parquet(meta_path)

    from kloppy import sportec
    raw = raw_files(match_id)
    if raw:
        dataset = sportec.load_tracking(raw_data=raw["tracking"], meta_data=raw["meta"], only_alive=False)
    else:  # figshare가 막지 않으면 kloppy가 직접 받는다
        dataset = sportec.load_open_tracking_data(match_id=match_id, only_alive=False)
    frame = dataset.to_df(engine="pandas")
    home, away = dataset.metadata.teams
    length = dataset.metadata.pitch_dimensions.pitch_length
    width = dataset.metadata.pitch_dimensions.pitch_width
    rows = [{"player_id": player.player_id, "team": side, "jersey": player.jersey_no,
             "position": str(player.starting_position) if player.starting_position else "",
             "pitch_length": length, "pitch_width": width}
            for side, team in (("home", home), ("away", away)) for player in team.players]
    meta = pd.DataFrame(rows)

    keep = ["frame_id", "period_id", "timestamp", "ball_state", "ball_owning_team_id", "ball_x", "ball_y", "ball_z"]
    players = [c for c in frame.columns if c.endswith("_x") or c.endswith("_y")]
    players = [c for c in players if not c.startswith("ball_")]
    tracking = frame[[c for c in keep if c in frame.columns] + players].copy()
    tracking = tracking.rename(columns={"period_id": "period", "ball_owning_team_id": "ball_owning_team"})
    tracking["timestamp_s"] = tracking.pop("timestamp").dt.total_seconds()
    tracking["ball_state"] = (tracking["ball_state"].astype(str).str.lower() == "alive").astype(np.int8)
    tracking["ball_owning_team"] = tracking["ball_owning_team"].map(
        {home.team_id: "home", away.team_id: "away"}).fillna("")

    if raw:
        events_dataset = sportec.load_event(event_data=raw["event"], meta_data=raw["meta"])
    else:
        events_dataset = sportec.load_open_event_data(match_id=match_id)
    events = events_dataset.to_df(engine="pandas")
    for column in events.columns:
        if events[column].dtype == object:
            events[column] = events[column].astype(str)
    if "timestamp" in events and hasattr(events["timestamp"], "dt"):
        events["timestamp_s"] = events.pop("timestamp").dt.total_seconds()

    os.makedirs(CONFIG["dfl_cache"], exist_ok=True)
    tracking.to_parquet(tracking_path, index=False)
    events.to_parquet(events_path, index=False)
    meta.to_parquet(meta_path, index=False)
    return tracking, events, meta


if __name__ == '__main__':
    wanted = MATCHES if sys.argv[1:] == ["all"] else sys.argv[1:] or ["J03WPY"]
    for match in wanted:
        tracking_frame, events_frame, meta_frame = load_match(match)
        print("{}: {} frames, {} player columns, {} events".format(
            match, len(tracking_frame), (tracking_frame.shape[1] - 8) // 2, len(events_frame)))
