import numpy as np
from mtah_relational.cli import _parse_player_groups


def test_groups_and_nan():
    xy = _parse_player_groups("0.5,0.25,2.0;NaN,NaN,NaN;0.1,0.2,3.4",3,(100,60))
    assert xy.shape == (3,2)
    assert np.allclose(xy[0],[50,15])
    assert np.isnan(xy[1]).all()
    assert np.allclose(xy[2],[10,12])
