"""Fixed rule index of spatial contact/approach, NOT P(contest) or H/A/D."""
from collections import deque
import math

import numpy as np


class CausalContestedIndex:
    """No fit/scaler/GT/future frames. All intervening observations must be reliable."""

    def __init__(self, fps=25):
        if not isinstance(fps, int) or fps <= 0:
            raise ValueError("fps must be a positive integer")
        self.fps = fps
        self.history = deque(maxlen=fps + 1)
        self.previous_frame = None
        self.previous_period = None

    def step(self, frame, period, features, quality_valid):
        if self.previous_frame is not None and frame <= self.previous_frame:
            raise ValueError("Index input frames must be strictly increasing")
        if (self.previous_frame is None or frame != self.previous_frame + 1
                or period != self.previous_period or period not in (1, 2)):
            self.history.clear()
        self.previous_frame, self.previous_period = frame, period
        keys = ("cross_contact_fraction_3m", "cross_cluster_fraction_3m", "cross_min_m",
                "cross_nearest_mean_m")
        valid = bool(quality_valid and period in (1, 2) and features["cross_valid"]
                     and np.isfinite([features[k] for k in keys]).all())
        current = (valid, features[keys[0]], features[keys[1]], features[keys[2]], features[keys[3]])
        self.history.append(current)
        result = dict(contested_index=np.nan, spatial_contact_index=np.nan,
                      approach_change_index=np.nan, contact_fraction_delta_1s=np.nan,
                      nearest_approach_m_per_s=np.nan, temporal_available=False, index_valid=valid)
        if not valid:
            return result
        spatial = 0.5*current[1] + 0.3*current[2] + 0.2*math.exp(-current[3]/3.0)
        result["spatial_contact_index"] = spatial
        result["contested_index"] = spatial
        if len(self.history) == self.fps+1 and all(p[0] for p in self.history):
            old = self.history[0]
            delta = current[1] - old[1]
            # Exactly one second at the validated sampling rate, aggregate approach only.
            approach = old[4] - current[4]
            change = 0.5*np.clip(delta, 0, 1) + 0.5*np.clip(approach/5.0, 0, 1)
            result.update(contested_index=float(0.8*spatial+0.2*change),
                          approach_change_index=float(change), temporal_available=True,
                          contact_fraction_delta_1s=float(delta), nearest_approach_m_per_s=float(approach))
        return result
