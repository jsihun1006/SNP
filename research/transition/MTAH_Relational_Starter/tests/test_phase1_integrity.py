import xml.etree.ElementTree as ET

import numpy as np
import pytest

from mtah_relational.data_quality import (ObservationPolicy, assess_observations,
    iter_player_frames, parse_players, read_metadata, validate_blocks)


def test_explicit_home_mapping_and_channel_slot_order(synthetic_atd):
    meta, _ = synthetic_atd()
    root = ET.parse(meta).getroot()
    refs = list(root.iter("PlayerChannelRef"))
    for r in refs:
        root.remove(r)
    for r in refs[6:]+refs[:6]:
        root.append(r)
    ET.ElementTree(root).write(meta)
    m = read_metadata(meta, expected_slots=4)
    assert m.team_codes.tolist() == [1, 1, 0, 0]
    assert m.mapping_source == "Score.Local/Visiting"
    assert m.coordinate_units == ("normalized",)
    assert m.fps == 25


@pytest.mark.parametrize("fps", [24, 30, "nan"])
def test_reject_fps_mismatch(synthetic_atd, fps):
    meta, _ = synthetic_atd(fps=fps)
    with pytest.raises(ValueError, match="frame rate"):
        read_metadata(meta, expected_slots=4)


@pytest.mark.parametrize("mutation", ["same_teams", "score", "units", "channel_order", "duplicate_refs", "slot_count"])
def test_metadata_rejects_ambiguous_contract(synthetic_atd, mutation):
    meta, _ = synthetic_atd()
    root = ET.parse(meta).getroot()
    slots = 4
    if mutation == "same_teams":
        list(root.iter("Team"))[1].set("id", "away")
    elif mutation == "score":
        root.remove(root.find("Score"))
    elif mutation == "units":
        root.find("Channel/Unit").text = "metres"
    elif mutation == "channel_order":
        root.find("PlayerChannel").set("channelId", "y")
    elif mutation == "duplicate_refs":
        list(root.iter("PlayerChannelRef"))[3].set("playerChannelId", "0x")
    else:
        slots = 5
    ET.ElementTree(root).write(meta)
    with pytest.raises(ValueError):
        read_metadata(meta, expected_slots=slots)


@pytest.mark.parametrize("ids", [[1, 2, 4, 4], [1, 3, 2, 4], [0, 1, 2, 3], [1, 2, 3], [1, 2, 3, 4, 5]])
def test_strict_ids_not_just_count_and_endpoints(synthetic_atd, ids):
    meta, raw = synthetic_atd(n_frames=4)
    m = read_metadata(meta, expected_slots=4)
    template = raw.read_text().splitlines()[0].split(":", 1)[1]
    raw.write_text("".join(f"{i}:{template}\n" for i in ids))
    with pytest.raises(ValueError):
        list(iter_player_frames(raw, m, (100, 60), 4, [(1, 0, 3)]))


def test_alignment_seconds_halves_and_ignored_ball(synthetic_atd):
    meta, raw = synthetic_atd(ball="NOT_A_BALL:IGNORED")
    m = read_metadata(meta, expected_slots=4)
    frames = list(iter_player_frames(raw, m, (100, 60), 60, [(1, 0, 29), (2, 40, 59)]))
    assert [f.frame for f in frames] == list(range(60))
    assert frames[25].frame/25 == 1
    assert [frames[i].period for i in (29, 30, 39, 40)] == [1, 0, 0, 2]
    np.testing.assert_allclose(frames[0].xy_m[0], [20, 12])
    assert not hasattr(frames[0], "ball")


@pytest.mark.parametrize("blocks", [[(1, 0, 5), (2, 5, 9)], [(0, 0, 9)], [(1, 0, 10)], [(1, 5, 4)]])
def test_bad_half_blocks(blocks):
    with pytest.raises(ValueError):
        validate_blocks(blocks, 10)


def test_missing_partial_infinite_malformed_and_unused_speed():
    xy, missing, partial, invalid = parse_players(
        "NaN,NaN,0;NaN,0.2,0;inf,0.1,0;bad,0.1,0;0.5,0.25,not_read;0.2,0.1", 6, (100, 60))
    assert missing.tolist() == [True, False, False, False, False, False]
    assert partial.tolist() == [False, True, False, False, False, False]
    assert invalid.tolist() == [False, False, True, True, False, True]
    np.testing.assert_allclose(xy[4], [50, 15])
    assert np.isnan(xy[[0, 1, 2, 3, 5]]).all()


def test_duplicate_suspicions_preserve_players_and_are_team_symmetric(make_frame):
    frame = make_frame(xy=[[10, 10], [10.1, 10], [30, 10], [30, 10]])
    policy = ObservationPolicy(min_team_players=2)
    xy, q = assess_observations(frame, (100, 60), policy)
    assert np.isfinite(xy).all()
    assert q["duplicate_suspect_pairs"] == 2
    assert q["duplicate_suspect_players"] == 4
    assert not q["quality_valid"]
    other = make_frame(frame.xy_m, 1-frame.team_codes)
    assert assess_observations(other, (100, 60), policy)[1] == q


def test_opponent_coincidence_is_not_same_team_duplicate(make_frame):
    f = make_frame([[10, 10], [20, 20], [10, 10], [22, 20]])
    _, q = assess_observations(f, (100, 60), ObservationPolicy(min_team_players=2))
    assert q["duplicate_suspect_pairs"] == 0
    assert q["quality_valid"]


def test_bounds_flag_without_clipping_or_manual_correction(make_frame):
    f = make_frame([[-1, 10], [20, 20], [103, 10], [22, 20]])
    xy, q = assess_observations(f, (100, 60), ObservationPolicy(min_team_players=1))
    assert xy[0, 0] == -1  # within margin, retained but index quality invalid
    assert np.isnan(xy[2]).all()  # beyond 2m margin
    assert q["outside_pitch_slots"] == 2
    assert q["excluded_bounds_slots"] == 1
    assert not q["quality_valid"]


@pytest.mark.parametrize("kwargs", [{"bounds_margin_m": -1}, {"duplicate_radius_m": float("nan")},
                                   {"min_team_players": 12}, {"duplicate_radius_m": -1}])
def test_policy_validation(kwargs):
    with pytest.raises(ValueError):
        ObservationPolicy(**kwargs)
