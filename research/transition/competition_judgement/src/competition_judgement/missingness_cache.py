"""Read-only reuse of player binary and pair caches; no ATD parsing."""
from pathlib import Path
import json
import numpy as np
from .quality import POLICY
from .temporal_source import TemporalSource
from .missingness_diagnostics import MaskContext


class ReadOnlyAnchorCache(MaskContext):
    support=TemporalSource.support
    anchor_series=TemporalSource.anchor_series

    def __init__(self,root,frames,manifest):
        super().__init__(frames,manifest)
        root=Path(root)
        raw=root/'data/derived/relational_temporal_v1_20261109/raw_cache'
        old=root/'data/derived/competition_judgement/temporal_v2_2_final_20261109'
        self.config=json.loads((old/'preregistered_config.json').read_text())
        if self.config['window_radius_frames']!=50 or self.config['fps']!=25:raise ValueError('Unexpected frozen source settings')
        if not np.array_equal(np.load(raw/'frame_ids.npy',mmap_mode='r'),self.ids):raise ValueError('Cache frame mismatch')
        if not np.array_equal(np.load(raw/'periods.npy',mmap_mode='r'),self.periods):raise ValueError('Cache period mismatch')
        xy=np.load(raw/'xy_m.npy',mmap_mode='r').copy()
        self.pitch=(105.,68.)
        # Same bounds policy as frozen TemporalSource; input units already metres.
        metadata=json.loads((raw/'cache.json').read_text())
        self.pitch=tuple(metadata['pitch']) if 'pitch' in metadata else self.pitch
        finite=np.isfinite(xy).all(-1)
        excluded=((xy < -POLICY.bounds_margin_m).any(-1)|(xy > np.asarray(self.pitch)+POLICY.bounds_margin_m).any(-1))&finite
        xy[~finite|excluded]=np.nan
        teams=np.load(raw/'team_codes.npy',mmap_mode='r')
        self.h=xy[:,teams==0];self.a=xy[:,teams==1]
        if not np.array_equal(np.isfinite(self.h).all(-1).sum(1),self.nh) or not np.array_equal(np.isfinite(self.a).all(-1).sum(1),self.na):raise ValueError('Player counts differ')
        self.kernel=np.load(old/'pair_cache/pair_kernel.npy',mmap_mode='r')
        self.hard3=np.load(old/'pair_cache/pair_3m.npy',mmap_mode='r');self.hard5=np.load(old/'pair_cache/pair_5m.npy',mmap_mode='r')
        shape=(len(self.ids),self.h.shape[1],self.a.shape[1])
        if any(x.shape!=shape or x.flags.writeable for x in (self.kernel,self.hard3,self.hard5)):raise ValueError('Invalid read-only cache')

    def reference_series(self,reference,batch=32):
        pieces=[]
        for start in range(0,len(reference),batch):
            ch=reference.iloc[start:start+batch];rows=np.searchsorted(self.ids,ch.frame.to_numpy())
            centers=ch[['cluster_centroid_x','cluster_centroid_y']].to_numpy()
            radii=np.maximum(self.config['anchor_min_radius_m'],ch.cluster_max_pairwise_distance_m.to_numpy()+self.config['anchor_diameter_margin_m'])
            s=self.anchor_series(rows,centers,radii)
            if not s['reliable'].all():raise ValueError('Reference support is not Full-reliable')
            pieces.append({k:s[k] for k in ('soft','nearest_m','edges_3m','edges_5m')})
        return {k:np.concatenate([x[k] for x in pieces]) for k in pieces[0]}
