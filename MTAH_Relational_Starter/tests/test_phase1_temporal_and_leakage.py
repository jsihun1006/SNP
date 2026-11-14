from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mtah_relational.candidate_baseline import CausalContestedIndex
from mtah_relational.data_quality import ObservationPolicy
from mtah_relational.features import add_temporal_features, geometry_features
from mtah_relational.preprocessing import spatial_input, spatial_inputs


def feature_table(frames, periods):
    df = pd.DataFrame(dict(frame=frames, period=periods))
    for key in ("cross_pairs_3m", "max_cross_cluster_3m", "centroid_sep_m", "home_radius",
                "away_radius", "home_cx", "away_cx"):
        df[key] = np.arange(len(df), dtype=float)
    return df


def test_temporal_deltas_exclude_period_zero_and_internal_gaps():
    zero = add_temporal_features(feature_table(np.arange(60), [0]*60))
    assert zero.filter(like="delta").isna().all().all()
    frames = np.r_[np.arange(30), np.arange(31, 61)]
    gap = add_temporal_features(feature_table(frames, [1]*60))
    assert np.isnan(gap.loc[40, "cross_pairs_3m_delta_prev_1s"])
    assert gap.loc[58, "cross_pairs_3m_delta_prev_1s"] == 25


@pytest.mark.parametrize("frames", [[0, 1, 1], [2, 1, 0], [0, .5, 1], [0, np.nan, 2]])
def test_temporal_deltas_reject_bad_ids(frames):
    with pytest.raises(ValueError):
        add_temporal_features(feature_table(frames, [1]*len(frames)))


def test_index_is_causal_and_team_swap_invariant():
    h = np.array([[10., 10.], [20., 20.]])
    seq = [geometry_features(h, h+[8-.1*i, 0]) for i in range(60)]
    def run(fs):
        index = CausalContestedIndex()
        return [index.step(i, 1, f, True) for i, f in enumerate(fs)]
    original = run(seq)
    changed = run(seq[:35] + [geometry_features(h, h+[50, 0])]*25)
    for i in range(35):
        for key in original[i]:
            np.testing.assert_allclose(original[i][key], changed[i][key], equal_nan=True)
    swapped = run([geometry_features(h+[8-.1*i, 0], h) for i in range(60)])
    for a, b in zip(original, swapped):
        np.testing.assert_allclose(a["contested_index"], b["contested_index"])
    assert not original[24]["temporal_available"]
    assert original[25]["temporal_available"]
    assert 0 <= original[-1]["contested_index"] <= 1


def test_index_history_never_crosses_half_gap_or_bad_quality():
    f = geometry_features([[10, 10], [20, 20]], [[12, 10], [22, 20]])
    index = CausalContestedIndex()
    for i in range(30):
        out = index.step(i, 1, f, i != 10)
    assert not out["temporal_available"]
    for i in range(30, 37):
        out = index.step(i, 1, f, True)
    assert out["temporal_available"]
    assert not index.step(37, 2, f, True)["temporal_available"]
    assert not index.step(100, 2, f, True)["temporal_available"]
    outside = index.step(101, 0, f, True)
    assert not outside["index_valid"] and np.isnan(outside["contested_index"])
    with pytest.raises(ValueError):
        index.step(101, 0, f, True)


def test_invalid_index_is_unavailable_not_zero():
    f = geometry_features([], [[1, 1]])
    out = CausalContestedIndex().step(0, 1, f, False)
    assert not out["index_valid"]
    assert np.isnan(out["contested_index"])


def test_hidden_raw_sentinel_changes_cannot_change_any_input(make_frame):
    policy = ObservationPolicy(min_team_players=2)
    clean = [make_frame(frame=i) for i in range(60)]
    changed = clean.copy()
    # Change positions, cardinality, team balance and duplicate/bounds quality.
    for i in range(25, 30):
        changed[i] = make_frame([[999, -5]]*7+[[1, 1]], [0]*7+[1], frame=i)
    a = spatial_inputs(clean, (100, 60), range(25, 30), policy)
    b = spatial_inputs(changed, (100, 60), range(25, 30), policy)
    for left, right in zip(a, b):
        for name in ("xy_m", "team", "values", "feature_valid", "quality", "team_structure", "team_structure_valid"):
            np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
        assert left.hidden == right.hidden
    assert a[25].xy_m.shape == (0, 2)
    assert not a[25].quality.any()
    assert not a[25].feature_valid.any()
    assert not a[24].hidden and not a[30].hidden


def test_spatial_view_preserves_team_structure_and_natural_missing_mask(make_frame):
    f = make_frame()
    p = ObservationPolicy(min_team_players=2)
    a = spatial_input(f, (100, 60), p)
    b = spatial_input(replace(f, team_codes=1-f.team_codes), (100, 60), p)
    np.testing.assert_allclose(a.values, b.values)
    np.testing.assert_allclose(a.quality, b.quality)
    np.testing.assert_allclose(a.team_structure, b.team_structure[::-1])
    natural = spatial_input(make_frame([[np.nan, np.nan]]*4), (100, 60), p)
    assert not natural.hidden
    assert not natural.feature_valid.any()
    assert np.isfinite(natural.values).all()
