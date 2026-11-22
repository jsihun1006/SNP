from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from mtah_relational import cli


def test_local_synthetic_atd_end_to_end(tmp_path, monkeypatch):
    meta = tmp_path / 'test_atd.xml'
    raw = tmp_path / 'test_atd.txt'
    meta.write_text('<Root><Team id="h"/><Team id="a"/>'
                    '<Player teamId="h"/><Player teamId="a"/></Root>', encoding='utf8')
    raw.write_text(''.join(f'{i}:0.5,0.5,0.0;0.52,0.5,0.0:NaN,NaN,NaN\n'
                           for i in range(1,31)), encoding='utf8')
    fake_games = SimpleNamespace(
        asset_paths=lambda source='tactical': {'atd_meta': str(meta), 'atd_raw': str(raw)},
        source_frames=lambda source='tactical': 30,
        GAMES={'SYN': {'pitch': (100.0, 60.0)}},
        GAME='SYN',
        half_video_blocks=lambda game: [(1,0,29)],
    )
    monkeypatch.setattr(cli, '_official', lambda: (fake_games, None))
    dest = tmp_path / 'derived' / 'features.csv'
    cli.run_extract(0, 30, str(dest), True)
    df = pd.read_csv(dest)
    assert len(df) == 30
    assert df.frame.iloc[0] == 0
    assert df.frame.iloc[-1] == 29
    assert df['cross_pairs_3m'].eq(1).all()
    nodes = np.load(dest.with_name('raw_nodes.npz'))
    assert nodes['xy_m'].shape == (30,2,2)
    assert nodes['observed'].all()
