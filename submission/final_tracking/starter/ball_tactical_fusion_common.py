"""Shared immutable XML, ATD and live-time mapping readers for ball fusion."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
MAPFILE=ROOT/'data/derived/sync/final_tracking_alignment.csv'
N=164487

def read(path):
    return pd.read_csv(path, float_precision='round_trip', low_memory=False)

def metadata(path):
    root = ET.parse(path).getroot()
    players = list(root.iter('Player'))
    keepers = []
    for i, player in enumerate(players):
        for parameter in player.iter('ProviderParameter'):
            if parameter.findtext('Name') == 'position_type' and (parameter.findtext('Value') or '').lower() == 'goalkeeper':
                keepers.append(i)
    speed = next(c for c in root.iter('Channel') if c.get('id') == 's')
    directions = {}
    for team in root.iter('Team'):
        for parameter in team.iter('ProviderParameter'):
            if parameter.findtext('Name') == 'attack_direction_first_half':
                directions[team.get('id')] = parameter.findtext('Value')
    return dict(players=len(players), goalkeeper_slots=keepers,
                fps=float(root.findtext('.//FrameRate')),
                pitch=[float(root.findtext('.//FieldSize/Width')), float(root.findtext('.//FieldSize/Height'))],
                speed_unit=speed.findtext('Unit'), team_first_half_directions=directions)

def scan_atd(path, meta, presence=False):
    balls = []; tokens = []; player_stats = []
    keeper_mask = np.zeros(meta['players'], bool)
    keeper_mask[meta['goalkeeper_slots']] = True
    with path.open('rb') as stream:
        for frame, line in enumerate(stream):
            parts = line.rstrip(b'\r\n').split(b':')
            assert len(parts) == 3 and int(parts[0]) == frame + 1
            ball_tokens = parts[2].split(b',')
            assert len(ball_tokens) == 3
            balls.append([float(v) for v in ball_tokens]); tokens.append(ball_tokens)
            if presence:
                values = np.fromstring(parts[1].replace(b';', b',').decode('ascii'), sep=',')
                assert len(values) == meta['players'] * 3
                people = values.reshape(-1, 3)
                finite = np.isfinite(people[:, :2]).all(axis=1)
                keeper_speed = finite & keeper_mask & np.isfinite(people[:, 2])
                player_stats.append([int(finite.sum()), int((finite & ~keeper_mask).sum()),
                                     int((finite & keeper_mask).sum()), int(keeper_speed.sum())])
    return np.asarray(balls), tokens, np.asarray(player_stats, dtype=int) if presence else None

def no_observed_players(outfield_count, keeper_with_finite_speed_count):
    """NaN-speed XML GK-only reference positions are not measured player tracks."""
    return (np.asarray(outfield_count) == 0) & (np.asarray(keeper_with_finite_speed_count) == 0)

def alignment(nt):
    source = read(MAPFILE)
    assert len(source) == nt and np.array_equal(source.tactical_frame, np.arange(nt))
    pairs = source[source.broadcast_atd_frame0.notna()]
    b = pairs.broadcast_atd_frame0.to_numpy(int); t = pairs.tactical_frame.to_numpy(int)
    assert len(np.unique(b)) == len(b) and np.all(np.diff(b) > 0) and np.all(np.diff(t) > 0)
    assert (b >= 0).all() and (b < N).all() and (t >= 0).all() and (t < nt).all()
    assert pairs.xy_reflection_index.eq(0).all()
    mapped = np.full(N, -1, int); half = np.zeros(N, int)
    mapped[b] = t; half[b] = pairs.half.to_numpy(int)
    specs = []
    for k, expected in [(1, 4014), (2, 26919)]:
        select = pairs.half.eq(k).to_numpy()
        bk, tk = b[select], t[select]
        assert np.all(tk - bk == expected) and np.array_equal(bk, np.arange(bk[0], bk[-1] + 1))
        specs.append(dict(half=k, broadcast_first=int(bk[0]), broadcast_last=int(bk[-1]),
                          tactical_first=int(tk[0]), tactical_last=int(tk[-1]), offset_frames=expected,
                          mapped_frames=len(bk), xy_reflection='NONE'))
    assert len(b) == 155923 and not (mapped[86638:86823] >= 0).any()
    return mapped, half, specs
