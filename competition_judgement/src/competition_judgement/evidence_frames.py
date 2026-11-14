"""Independent score maxima with source keys; never one combined event."""
import numpy as np
import pandas as pd
from .evidence_features import NAMES

SCORES=(*NAMES.values(),'positive_evidence_score','negative_evidence_score')


class FrameEvidence:
    def __init__(self,frames):
        self.frames=frames;self.metrics=[f'{m}_{s}' for m in ('full','partial') for s in SCORES]
        shape=(len(frames),len(self.metrics));self.best=np.full(shape,np.nan)
        self.row=np.full(shape,-1,int);self.radius=np.full(shape,np.nan);self.cluster=self.radius.copy()
        self.full_support=np.zeros(shape,bool);self.count=np.zeros(shape,int)
        self.n_clusters=np.zeros(len(frames),int);self.n_full_source=self.n_clusters.copy()
        self.paired={m:np.full(len(frames),np.nan) for m in ('full','partial')}

    def add(self,base,features,global_rows):
        ids=base.frame.to_numpy(int);official=base.period.to_numpy()>0
        np.add.at(self.n_clusters,ids,1)
        np.add.at(self.n_full_source,ids[official],features.full_reliable_window.to_numpy()[official])
        for j,metric in enumerate(self.metrics):
            value=features[metric].to_numpy(float);good=np.isfinite(value)
            np.add.at(self.count[:,j],ids[good],1)
            if not good.any():continue
            table=pd.DataFrame(dict(frame=ids[good],value=value[good],local_row=np.flatnonzero(good)))
            take=table.loc[table.groupby('frame').value.idxmax(),'local_row'].to_numpy(int)
            fi=ids[take];better=~np.isfinite(self.best[fi,j])|(value[take]>self.best[fi,j])|((value[take]==self.best[fi,j])&(global_rows[take]<self.row[fi,j]))
            take=take[better];fi=ids[take]
            self.best[fi,j]=value[take];self.row[fi,j]=global_rows[take]
            self.radius[fi,j]=base.radius_m.to_numpy()[take];self.cluster[fi,j]=base.cluster_id.to_numpy()[take]
            self.full_support[fi,j]=features.full_reliable_window.to_numpy()[take]
            if metric.endswith('positive_evidence_score'):
                mode=metric.split('_')[0];self.paired[mode][fi]=features[f'{mode}_negative_evidence_score'].to_numpy()[take]

    def table(self):
        out={k:self.frames[k].to_numpy() for k in ('frame','period','quality_tier','strict_quality_valid')}
        out.update(n_current_clusters=self.n_clusters,n_full_reliable_anchors=self.n_full_source)
        for j,metric in enumerate(self.metrics):
            valid=np.isfinite(self.best[:,j]);out[metric+'_max']=self.best[:,j]
            out[metric+'_valid_n']=self.count[:,j];out[metric+'_source_radius_m']=self.radius[:,j]
            out[metric+'_source_cluster_id']=self.cluster[:,j]
            out[metric+'_support']=np.where(valid,np.where(self.full_support[:,j],'full_reliable','experimental_partial'),
                np.where(self.frames.period.to_numpy()==0,'outside_official',np.where(self.n_clusters>0,'spatial_only','no_current_cluster')))
        for mode in ('full','partial'):
            out[f'{mode}_negative_at_positive_representative']=self.paired[mode]
        out['aggregation_semantics']='independent_anchor_maxima_not_one_event'
        return pd.DataFrame(out)
