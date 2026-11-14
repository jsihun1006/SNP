import numpy as np
import pandas as pd
from mtah_relational.features import geometry_features, max_bipartite_cluster_size, add_temporal_features


def test_basic_cross_pairs_and_permutation_invariance():
    h = np.array([[0.,0.], [1.,1.], [60.,30.]])
    a = np.array([[2.,0.], [2.,2.], [100.,50.]])
    d1 = geometry_features(h, a)
    d2 = geometry_features(h[::-1], a[[1, 2, 0]])
    assert d1["cross_pairs_3m"] == 4
    assert d1["max_cross_cluster_3m"] == 4
    for key in d1:
        assert np.isclose(d1[key], d2[key], equal_nan=True), key


def test_missing_points_do_not_break_stats():
    d = geometry_features([[1.,1.], [np.nan,np.nan]], [[2.,2.]])
    assert d["cross_pairs_3m"] == 1
    assert d["n_home"] == 1
    empty = geometry_features(np.empty((0,2)), np.empty((0,2)))
    assert empty["cross_pairs_3m"] == 0
    assert np.isnan(empty["centroid_sep_m"])


def test_bipartite_cluster():
    mask = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
    assert max_bipartite_cluster_size(mask) == 4


def test_temporal_delta_never_crosses_periods():
    n = 60
    df = pd.DataFrame({"frame": np.arange(n), "period": [1]*30+[2]*30})
    for name in ["cross_pairs_3m", "max_cross_cluster_3m", "centroid_sep_m",
                 "home_radius", "away_radius", "home_cx", "away_cx"]:
        df[name] = np.arange(n, dtype=float)
    result = add_temporal_features(df)
    assert np.isnan(result.loc[32, "cross_pairs_3m_delta_prev_1s"])
    assert np.isnan(result.loc[28, "cross_pairs_3m_delta_next_1s"])
    assert result.loc[26, "cross_pairs_3m_delta_prev_1s"] == 25

