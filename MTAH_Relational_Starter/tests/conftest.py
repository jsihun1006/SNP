"""Synthetic fixtures only: no competition files or GT are loaded by tests."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from mtah_relational.data_quality import PlayerFrame


@pytest.fixture
def make_frame():
    def make(xy=((10, 10), (20, 20), (12, 10), (22, 20)),
             teams=(0, 0, 1, 1), frame=0, period=1):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        missing = np.isnan(xy).all(axis=1)
        partial = np.isnan(xy).any(axis=1) & ~missing
        invalid = np.isinf(xy).any(axis=1)
        return PlayerFrame(frame, period, xy, np.asarray(teams, dtype=np.int8), missing, partial, invalid)
    return make


@pytest.fixture
def synthetic_atd(tmp_path):
    def make(n_frames=60, fps=25, n_home=2, n_away=2, ball="NaN,NaN,NaN"):
        folder = tmp_path/str(len(list(tmp_path.iterdir())))
        folder.mkdir()
        root = ET.Element("Root")
        ET.SubElement(root, "FrameRate").text = str(fps)
        field = ET.SubElement(root, "FieldSize")
        ET.SubElement(field, "Width").text = "100"
        ET.SubElement(field, "Height").text = "60"
        # Teams deliberately declared AWAY FIRST, so iteration order is unsafe.
        ET.SubElement(root, "Team", id="away")
        ET.SubElement(root, "Team", id="home")
        ET.SubElement(root, "Score", idLocalTeam="home", idVisitingTeam="away")
        for channel, name, unit in (("x", "position_x", "normalized"),
                                    ("y", "position_y", "normalized"),
                                    ("s", "speed_kph", "kph")):
            node = ET.SubElement(root, "Channel", id=channel)
            ET.SubElement(node, "Name").text = name
            ET.SubElement(node, "Unit").text = unit
        n_slots = n_home+n_away
        for i in range(n_slots):
            ET.SubElement(root, "Player", id=f"slot{i}", teamId="home" if i < n_home else "away")
            for c in ("x", "y", "s"):
                ET.SubElement(root, "PlayerChannel", id=f"{i}{c}", playerId=f"slot{i}", channelId=c)
                ET.SubElement(root, "PlayerChannelRef", playerChannelId=f"{i}{c}")
        meta, raw = folder/"synthetic_atd.xml", folder/"synthetic_atd.txt"
        ET.ElementTree(root).write(meta, encoding="unicode")
        h = [(0.2+i*0.08, 0.2+i*0.08) for i in range(n_home)]
        a = [(0.22+i*0.08, 0.2+i*0.08) for i in range(n_away)]
        groups = ";".join(f"{x},{y},unused_speed" for x, y in h+a)
        raw.write_text("".join(f"{i}:{groups}:{ball}\n" for i in range(1, n_frames+1)))
        return meta, raw
    return make
