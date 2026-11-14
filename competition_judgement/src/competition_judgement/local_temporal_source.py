"""B.2 read-only extension: ephemeral nodes of the frozen geometric corridor."""
import numpy as np
from .pattern_source import PatternSource


class LocalTemporalSource(PatternSource):
    def path_metadata(self, selected, centers, radii):
        result = super().path_metadata(selected, centers, radii)
        selected = np.asarray(selected, int)
        path = np.full((len(selected), 21), -1, int)
        path[:, 10] = selected
        for sign, links in ((-1, self.prev), (1, self.next)):
            prior = selected.copy()
            for step in range(1, 11):
                prior = np.where(prior >= 0, links[np.maximum(prior, 0)], -1)
                path[:, 10 + sign * step] = prior
        safe = np.maximum(path, 0)
        xy = self.flat_xy[safe]
        inside = (path >= 0) & (np.sum((xy - centers[:, None]) ** 2, axis=-1) <= radii[:, None] ** 2)
        center_rows = self.cluster_frame[np.maximum(selected, 0)]
        usable = result['path_support'] & inside & self.strict[center_rows, None]
        usable &= (self.periods[center_rows] > 0)[:, None]
        result.update(local_path_support=usable,
                      local_path_xy=np.where(usable[..., None], xy, np.nan),
                      local_path_radius=np.where(usable, self.flat_R[safe], np.nan),
                      local_path_diameter=np.where(usable, self.flat_D[safe], np.nan),
                      local_path_density=np.where(usable, self.flat_density[safe], np.nan),
                      local_path_inside=inside)
        return result


def load_local_source(root, legacy):
    # Constructor uses existing read-only caches; it does not parse ATD or rebuild pairs.
    from .run_pattern_b1 import load_source
    frames, base = load_source(root, legacy)
    source = object.__new__(LocalTemporalSource)
    source.__dict__.update(base.__dict__)
    return frames, source


def mask_local_series(series, deleted):
    """Artificial deletion invalidates support, never fills values across a hole."""
    from .pattern_diagnostics import mask_series
    changed = mask_series(series, deleted)
    support = series['local_path_support'] & ~deleted[:, 40:61]
    changed['local_path_support'] = support
    for key in ('local_path_xy', 'local_path_radius', 'local_path_diameter', 'local_path_density'):
        mask = support[..., None] if series[key].ndim == 3 else support
        changed[key] = np.where(mask, series[key], np.nan)
    return changed
