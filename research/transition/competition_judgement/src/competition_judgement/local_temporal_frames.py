"""Source-preserving existence aggregation; sequences are formed before grouping."""
import numpy as np
import pandas as pd
from .local_temporal_features import SIGNALS, SEQUENCES

FRAME_SIGNALS = SIGNALS + ('local_rise_then_fall_short', 'local_rise_then_fall_near')


class LocalTemporalFrames:
    def __init__(self, frames):
        self.ids = frames.frame.to_numpy(int)
        n, k = len(frames), len(FRAME_SIGNALS)
        self.total = np.zeros(n, int)
        self.hits = np.zeros((n, k, 2), int)
        self.unknown = np.zeros((n, k), int)
        self.rank = np.full((n, k), 99, int)
        self.radius = np.full((n, k), np.nan)
        self.cid = self.radius.copy()
        self.sequence_bits = np.zeros((n, 2), int)

    def add(self, base, local):
        f = np.searchsorted(self.ids, base.frame.to_numpy(int))
        if (f >= len(self.ids)).any() or not np.array_equal(self.ids[f], base.frame):
            raise ValueError('Frame axis mismatch')
        radius = base.radius_m.to_numpy(int); cid = base.cluster_id.to_numpy(int); ri = (radius == 5).astype(int)
        np.add.at(self.total, f, 1)
        states = {key: local[key].to_numpy() for key in SIGNALS}
        for j, scale in enumerate(('short', 'near')):
            seq = local['local_sequence_' + scale].to_numpy()
            code = np.array([SEQUENCES.index(x) for x in seq])
            np.bitwise_or.at(self.sequence_bits[:, j], f, 1 << code)
            states['local_rise_then_fall_' + scale] = np.where(code == 0, 'OBSERVED', np.where(np.isin(code, [2, 5, 6, 7, 8]), 'UNKNOWN', 'NOT_OBSERVED'))
        for j, key in enumerate(FRAME_SIGNALS):
            state = states[key]; hit = state == 'OBSERVED'; unknown = state == 'UNKNOWN'
            np.add.at(self.hits[:, j], (f, ri), hit)
            np.add.at(self.unknown[:, j], f, unknown)
            rank = np.where(hit, 0, np.where(unknown, 1, 2))
            order = np.lexsort((cid, radius, rank, f))
            first = order[np.r_[True, f[order][1:] != f[order][:-1]]]
            better = (rank[first] < self.rank[f[first], j]) | ((rank[first] == self.rank[f[first], j]) &
                     ((radius[first] < self.radius[f[first], j]) | ((radius[first] == self.radius[f[first], j]) & (cid[first] < self.cid[f[first], j]))))
            use = first[better]
            self.rank[f[use], j] = rank[use]; self.radius[f[use], j] = radius[use]; self.cid[f[use], j] = cid[use]

    def table(self):
        result = dict(frame=self.ids, anchor_views_n=self.total)
        for j, key in enumerate(FRAME_SIGNALS):
            hit = self.hits[:, j].sum(1) > 0
            known = (self.total > 0) & (self.unknown[:, j] == 0)
            result[key] = np.where(hit, 'OBSERVED', np.where(known, 'NOT_OBSERVED', 'UNKNOWN'))
            result[key + '_source_radius_m'] = self.radius[:, j]
            result[key + '_source_cluster_id'] = self.cid[:, j]
            for rj, radius in enumerate((3, 5)):
                result[key + f'_observed_{radius}m_views'] = self.hits[:, j, rj]
        result['local_sequence_short_view_mask'] = self.sequence_bits[:, 0]
        result['local_sequence_near_view_mask'] = self.sequence_bits[:, 1]
        result['local_frame_reason'] = np.where(self.total > 0, 'ANCHOR_SPECIFIC_SOURCES', 'NO_CURRENT_ANCHOR')
        return pd.DataFrame(result)
