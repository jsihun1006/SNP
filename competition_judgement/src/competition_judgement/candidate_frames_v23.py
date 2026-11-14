"""Aggregate completed anchor decisions; never fuse Evidence across anchors."""
import numpy as np
import pandas as pd
from .candidate_decision_v23 import STATES, CANDIDATES


class CandidateFrames:
    def __init__(self, frames, config):
        self.frames = frames.reset_index(drop=True); self.ids = frames.frame.to_numpy(int)
        if len(np.unique(self.ids)) != len(self.ids) or (np.diff(self.ids) <= 0).any(): raise ValueError('Ordered unique frame axis required')
        self.priority = config['frame_priority']; n = len(frames)
        self.counts = np.zeros((n, len(STATES), 2), int)
        # source slots 0=all,1=3m,2=5m. These are output views, not player IDs.
        self.rank = np.full((n, 3), 99, int); self.radius = np.full((n, 3), np.nan); self.cid = self.radius.copy()
        self.reason = np.full((n, 3), 'NO_CURRENT_ANCHOR', object)
        self.basis = np.zeros((n, 3), int); self.future = self.basis.copy()

    def add(self, base, decisions):
        f = np.searchsorted(self.ids, base.frame.to_numpy(int))
        if len(base) != len(decisions) or (f >= len(self.ids)).any() or not np.array_equal(self.ids[f], base.frame): raise ValueError('Frame/key axis mismatch')
        r = base.radius_m.to_numpy(int); cid = base.cluster_id.to_numpy(int); ri = (r == 5).astype(int)
        state = decisions.decision_state.to_numpy()
        for j, name in enumerate(STATES): np.add.at(self.counts[:, j], (f, ri), state == name)
        ranks = np.array([self.priority.index(x) for x in state])
        for source_slot in range(3):
            eligible = np.ones(len(base), bool) if source_slot == 0 else ri == source_slot - 1
            indices = np.flatnonzero(eligible)
            if not len(indices): continue
            order = indices[np.lexsort((cid[indices], r[indices], ranks[indices], f[indices]))]
            first = order[np.r_[True, f[order][1:] != f[order][:-1]]]
            old = self.rank[f[first], source_slot]
            better = (ranks[first] < old) | ((ranks[first] == old) & ((r[first] < self.radius[f[first], source_slot]) |
                     ((r[first] == self.radius[f[first], source_slot]) & (cid[first] < self.cid[f[first], source_slot]))))
            use = first[better]; idx = f[use]
            self.rank[idx, source_slot] = ranks[use]; self.radius[idx, source_slot] = r[use]; self.cid[idx, source_slot] = cid[use]
            self.reason[idx, source_slot] = decisions.decision_reason.to_numpy()[use]
            self.basis[idx, source_slot] = decisions.decision_basis_mask.to_numpy(int)[use]
            self.future[idx, source_slot] = decisions.decision_future_frames.to_numpy(int)[use]

    def table(self):
        frames = self.frames; n = len(frames)
        official = frames.period.isin([1, 2]).to_numpy()
        strict = frames.strict_quality_valid.to_numpy(bool)
        teams = (frames.n_home.to_numpy() > 0) & (frames.n_away.to_numpy() > 0)
        context = official & strict & teams
        totals = self.counts.sum(axis=(1, 2)); has = totals > 0
        if (self.counts[:, :2].sum(axis=(1, 2))[~context] > 0).any(): raise ValueError('Anchor candidate contradicts current frame QC')
        empty_known = ~has & context
        for radius in (3, 5):
            column = f'n_local_clusters_{radius}m'
            if column not in frames: empty_known[:] = False
            else: empty_known &= frames[column].to_numpy() == 0
        state = np.array(self.priority, object)[np.minimum(self.rank[:, 0], 4)]
        state[~has] = np.where(empty_known[~has], 'NO_DETECTED_SIGNAL', 'INSUFFICIENT')
        state[~context] = 'INSUFFICIENT'
        reason = self.reason[:, 0].copy(); reason[empty_known] = 'SPATIAL_NO_MIXED_CLUSTER'
        reason[~has & ~empty_known] = 'CURRENT_ANCHOR_UNAVAILABLE'
        reason[~teams] = 'CURRENT_TEAMS_UNOBSERVED'
        reason[~strict] = 'CURRENT_QC'; reason[~official] = 'OUTSIDE_RECORD'
        out = dict(frame=self.ids, decision_state=state, decision_reason=reason, current_strict=strict,
                   current_teams_observed=teams, current_record_valid=official, anchor_views_n=totals,
                   representative_radius_m=self.radius[:, 0], representative_cluster_id=self.cid[:, 0],
                   representative_basis_mask=self.basis[:, 0], decision_future_frames=np.where(empty_known | ~context, 0, self.future[:, 0]),
                   any_insufficient_anchor=self.counts[:, STATES.index('INSUFFICIENT')].sum(1) > 0,
                   frame_insufficient=state == 'INSUFFICIENT',
                   temporal_assessment=np.where(empty_known, 'NOT_APPLICABLE', np.where(~context | ~has, 'UNAVAILABLE', 'ANCHOR_SPECIFIC')))
        for j, name in enumerate(STATES):
            out[name.lower() + '_exists'] = self.counts[:, j].sum(1) > 0
            for k, radius in enumerate((3, 5)): out[f'{name.lower()}_{radius}m_anchor_views'] = self.counts[:, j, k]
        for k, radius in enumerate((3, 5)):
            slot = k + 1
            out[f'candidate_{radius}m_exists'] = self.counts[:, :2, k].sum(1) > 0
            out[f'candidate_{radius}m_anchor_views'] = self.counts[:, :2, k].sum(1)
            out[f'radius_{radius}m_state'] = np.where(self.rank[:, slot] < 99, np.array(self.priority, object)[np.minimum(self.rank[:, slot], 4)], 'NO_ANCHOR')
            out[f'radius_{radius}m_source_cluster_id'] = self.cid[:, slot]
            out[f'radius_{radius}m_reason'] = self.reason[:, slot]
        return pd.DataFrame(out)
