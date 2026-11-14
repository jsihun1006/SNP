import numpy as np
import pytest

from mtah_relational.features import geometry_features
from mtah_relational.spatial_schema import FEATURE_NAMES, symmetric_summary, team_structures
from mtah_relational.zones import zone_metrics


def test_exact_metric_distances_thresholds_and_nearest_mean():
    f = geometry_features([[0, 0]], [[3, 0], [0, 5], [0, 10]])
    assert [f[f"cross_pairs_{r}m"] for r in (3, 5, 10)] == [1, 2, 3]
    assert f["cross_min_m"] == 3
    assert f["cross_mean_m"] == 6
    assert f["cross_nearest_mean_m"] == 5.25  # each player's nearest opponent
    assert f["max_cross_cluster_3m"] == 2


def test_team_geometry_preserves_both_teams():
    h = [[0, 0], [4, 0], [0, 2], [4, 2]]
    a = np.array(h) + [10, 20]
    f = geometry_features(h, a)
    assert (f["home_cx"], f["home_cy"]) == (2, 1)
    assert (f["away_cx"], f["away_cy"]) == (12, 21)
    assert (f["home_width"], f["home_length"], f["home_bbox_area"]) == (2, 4, 8)
    assert f["home_compactness_m"] == pytest.approx(np.sqrt(5))
    assert f["home_density_per_m2"] == .5
    assert f["home_nearest_teammate_m"] == 2
    np.testing.assert_allclose(team_structures(geometry_features(a, h)), team_structures(f)[::-1])


def test_cross_team_component_chain_is_not_clique():
    f = geometry_features([[0, 0], [4, 0]], [[2, 0], [6, 0]])
    assert f["max_cross_cluster_3m"] == 4
    assert f["cross_pairs_3m"] == 3


@pytest.mark.parametrize("seed", range(10))
def test_permutation_and_team_swap_invariant_summaries(seed):
    rng = np.random.default_rng(seed)
    h, a = rng.uniform(0, 60, (8, 2)), rng.uniform(0, 60, (11, 2))
    f = geometry_features(h, a)
    shuffled = geometry_features(rng.permutation(h), rng.permutation(a))
    for key in f:
        np.testing.assert_allclose(f[key], shuffled[key], rtol=1e-12, atol=1e-12)
    values, mask = symmetric_summary(f, (100, 60))
    swapped, swapped_mask = symmetric_summary(geometry_features(a, h), (100, 60))
    assert len(values) == len(FEATURE_NAMES) == 24
    np.testing.assert_allclose(values, swapped, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(mask, swapped_mask)


def test_missing_team_zero_is_not_valid_non_contest_target():
    f = geometry_features([[1, 1], [np.nan, 1]], [])
    assert f["cross_pairs_3m"] == 0
    assert not f["cross_valid"]
    values, valid = symmetric_summary(f, (100, 60))
    assert not valid.any()
    assert np.isnan(values).all()
    single = geometry_features([[1, 1]], [[2, 2]])
    _, valid = symmetric_summary(single, (100, 60))
    assert valid[:7].all()
    assert not valid[7:13].any()
    assert np.isnan(single["home_density_per_m2"])


def test_global_distance_crosses_zone_boundary():
    pitch = (105.3, 68)
    boundary = pitch[0]/8
    h, a = np.array([[boundary-.1, 30]]), np.array([[boundary+.1, 30]])
    f = geometry_features(h, a)
    zones = zone_metrics(np.concatenate([h, a]), np.array([0, 1]), pitch)
    assert f["cross_pairs_3m"] == 1
    assert sum(z["within_zone_pairs_3m"] for z in zones) == 0
    assert sum(z["n_home"] for z in zones) == 1
    assert sum(z["n_away"] for z in zones) == 1


def test_zone_mixing_area_boundaries_and_empty():
    zones = zone_metrics(np.array([[0, 0], [1, 1], [100, 60], [101, 60]]),
                         np.array([0, 1, 0, 1]), (100, 60))
    assert len(zones) == 48
    assert zones[0]["mixed_index"] == 1
    assert zones[0]["n_players"] == 2
    assert zones[-1]["n_players"] == 1
    assert sum(z["n_players"] for z in zones) == 3
    assert zones[0]["density_per_m2"] == pytest.approx(2/(100*60/48))
    assert np.isnan(zones[1]["mixed_index"])
