"""Existence aggregation, with one source per pattern; no Evidence maxima fusion."""
import numpy as np
import pandas as pd
from .pattern_profiles import PATTERNS


class PatternFrames:
    def __init__(self,frames):
        self.frames=frames;n=len(frames)
        self.total=np.zeros(n,int);self.unknown=np.zeros((n,4),int);self.known=self.unknown.copy()
        self.supported=np.zeros((n,4,2),int);self.available=self.supported.copy()
        self.rank=np.full((n,4),99,int);self.source_radius=np.full((n,4),np.nan);self.source_id=self.source_radius.copy()

    def add(self,base,profile):
        f=base.frame.to_numpy(int);r=base.radius_m.to_numpy(int);cid=base.cluster_id.to_numpy(int);ri=(r==5).astype(int)
        np.add.at(self.total,f,1)
        for j,p in enumerate(PATTERNS):
            unknown=profile[p+'_outcome'].to_numpy()=='UNKNOWN';hit=profile[p+'_outcome'].to_numpy()=='SUPPORTED'
            np.add.at(self.unknown[:,j],f,unknown);np.add.at(self.known[:,j],f,~unknown)
            np.add.at(self.supported[:,j],(f,ri),hit);np.add.at(self.available[:,j],(f,ri),profile[p+'_obs'].to_numpy()!='UNKNOWN')
            ranks=np.where(hit,0,np.where(unknown,1,2))
            # Within each rank, smaller radius then canonical cluster id. No score selection.
            order=np.lexsort((cid,r,ranks,f));first=order[np.r_[True,f[order][1:]!=f[order][:-1]]]
            oldrank=self.rank[f[first],j];oldr=self.source_radius[f[first],j];oldid=self.source_id[f[first],j]
            better=(ranks[first]<oldrank)|((ranks[first]==oldrank)&((r[first]<oldr)|((r[first]==oldr)&(cid[first]<oldid))))
            use=first[better];self.rank[f[use],j]=ranks[use];self.source_radius[f[use],j]=r[use];self.source_id[f[use],j]=cid[use]

    def table(self):
        out={k:self.frames[k].to_numpy() for k in ('frame','period','strict_quality_valid')}
        out['anchor_views_n']=self.total
        empty_known=(self.total==0)&self.frames.strict_quality_valid.to_numpy(bool)&(self.frames.period.to_numpy()>0)&(
            self.frames.n_home.to_numpy()>0)&(self.frames.n_away.to_numpy()>0)
        anyhit=np.zeros(len(self.frames),int)
        for j,p in enumerate(PATTERNS):
            hit=self.supported[:,j].sum(1)>0
            notseen=((self.total>0)&(self.unknown[:,j]==0))|empty_known
            outcome=np.where(hit,'SUPPORTED',np.where(notseen,'NOT_OBSERVED','UNKNOWN'))
            out[p+'_outcome']=outcome;out[p+'_obs']=np.where((hit|notseen),'COMPLETE','UNKNOWN')
            out[p+'_seq']=np.where(hit,'VERIFIED',np.where(empty_known,'NOT_APPLICABLE','UNVERIFIED'))
            out[p+'_source_radius_m']=self.source_radius[:,j];out[p+'_source_cluster_id']=self.source_id[:,j]
            out[p+'_reason']=np.where(hit,'ANCHOR_SUPPORTED',np.where(empty_known,'NO_OBSERVED_MIXED_ANCHOR',np.where(notseen,'ALL_ANCHORS_NOT_MET','ANCHORS_UNKNOWN')))
            for k,radius in enumerate((3,5)):
                out[f'{p}_available_{radius}m_views']=self.available[:,j,k]
                out[f'{p}_supported_{radius}m_views']=self.supported[:,j,k]
            anyhit+=hit
        out['supported_pattern_types_n']=anyhit
        return pd.DataFrame(out)
