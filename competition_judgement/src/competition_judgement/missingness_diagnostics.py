"""Stage A read-only mask audit. No inference policy or football labels."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .partial_window_diagnostics import run_statistics, policy_predicates
from .controlled_temporal_masking import FEATURES

CAUSES=('boundary_or_record_unavailable','anchor_team_absent_ambiguous',
        'current_strict_qc_failed','offset_strict_qc_failed','global_count_changed')
REGIONS={'far_past':slice(0,25),'near_past':slice(25,50),'present':slice(50,51),
         'near_future':slice(51,76),'far_future':slice(76,101)}


def unpack_masks(values):
    """Leading zeros are significant: CSV mask columns MUST be read as str."""
    raw=np.frombuffer(b''.join(bytes.fromhex(str(x)) for x in values),dtype=np.uint8).reshape(-1,13)
    return np.unpackbits(raw,axis=1,bitorder='little')[:,:101].astype(bool)


class MaskContext:
    def __init__(self,frames,manifest):
        self.ids=frames.frame.to_numpy();self.periods=frames.period.to_numpy()
        self.strict=frames.strict_quality_valid.to_numpy(bool)
        self.half_bounds={p:(self.ids[self.periods==p].min(),self.ids[self.periods==p].max()) for p in (1,2) if (self.periods==p).any()}
        self.nh=frames.n_home.to_numpy(int);self.na=frames.n_away.to_numpy(int)
        self.roles=np.full(len(frames),'outside',dtype='<U10')
        self.split_first=np.full(len(frames),-1,int);self.split_last=self.split_first.copy()
        for b in manifest['blocks']:
            sel=(self.ids>=b['first'])&(self.ids<=b['last'])
            self.roles[sel]=b['split'];self.split_first[sel]=b['first'];self.split_last[sel]=b['last']
        changes=np.r_[True,(np.diff(self.ids)!=1)|(np.diff(self.periods)!=0)|(self.roles[1:]!=self.roles[:-1])]
        starts=np.flatnonzero(changes);ends=np.r_[starts[1:]-1,len(frames)-1]
        self.first=np.empty(len(frames),int);self.last=self.first.copy()
        for a,b in zip(starts,ends):self.first[a:b+1]=a;self.last[a:b+1]=b
        outside=self.periods==0
        self.first[outside]=np.flatnonzero(outside);self.last[outside]=np.flatnonzero(outside)

    def support(self,rows):
        offsets=np.arange(-50,51);ideal=rows[:,None]+offsets
        record=(ideal>=self.first[rows,None])&(ideal<=self.last[rows,None])&(self.periods[rows,None]>0)
        return np.where(record,ideal,rows[:,None]),record

    def audit(self,chunk):
        rows=np.searchsorted(self.ids,chunk.frame.to_numpy())
        if not np.array_equal(self.ids[rows],chunk.frame):raise ValueError('Frame axis mismatch')
        masks={k:unpack_masks(chunk[f'{k}_mask_hex']) for k in ('record','observed','reliable')}
        idx,expected=self.support(rows)
        if not np.array_equal(expected,masks['record']):raise ValueError('Record mask mismatch')
        same=(self.nh[idx]==self.nh[rows,None])&(self.na[idx]==self.na[rows,None])
        obs=masks['observed'];rec=masks['record']
        expected=obs&self.strict[idx]&self.strict[rows,None]&same
        if not np.array_equal(expected,masks['reliable']):raise ValueError('Reliable mask mismatch')
        if (obs&~rec).any():raise ValueError('Observed without record')
        conditions=(~rec,rec&~obs,rec&~self.strict[rows,None],rec&~self.strict[idx],rec&~same)
        code=np.zeros(rec.shape,np.uint8)
        for bit,condition in enumerate(conditions):code|=condition.astype(np.uint8)<<bit
        if not np.array_equal(code>0,~masks['reliable']):raise ValueError('Failure decomposition mismatch')
        return rows,idx,masks,code,same


def strata(chunk,roles):
    radius=chunk.radius_m.to_numpy(int);strict=chunk.strict_quality_valid.to_numpy(bool)
    size=chunk.n_players.to_numpy(int)
    out={'all':np.ones(len(chunk),bool)}
    for r in (3,5):
        out[f'radius={r}']=radius==r
        for q,v in [('strict',strict),('low',~strict)]:out[f'radius={r};quality={q}']=(radius==r)&v
    for q,v in [('strict',strict),('low',~strict)]:out[f'quality={q}']=v
    for label,v in [('2',size==2),('3',size==3),('4-5',(size>=4)&(size<=5)),('6+',size>=6)]:
        out[f'size={label}']=v
        for r in (3,5):out[f'radius={r};size={label}']=v&(radius==r)
    for role in ('train','validation','test'):out[f'role={role}']=roles==role
    return out


class Aggregate:
    def __init__(self):self.data={}
    def add(self,group,field,value):
        d=self.data.setdefault(group,{})
        if isinstance(value,np.ndarray):
            d[field]=d.get(field,np.zeros_like(value,dtype=np.int64))+value
        else:d[field]=d.get(field,0)+int(value)
    def failure(self,group,prefix,mask):
        n,t=mask.shape;lengths,maximum,counts=run_statistics(mask)
        self.add(group,prefix+'_windows',mask.any(1).sum());self.add(group,prefix+'_offsets',mask.sum())
        self.add(group,prefix+'_run_length_hist',np.bincount(lengths,minlength=102)[:102])
        self.add(group,prefix+'_max_run_hist',np.bincount(maximum,minlength=102)[:102])
        self.add(group,prefix+'_run_count_hist',np.bincount(counts,minlength=102)[:102])
        any_=mask.any(1);first=np.argmax(mask,1)
        self.add(group,prefix+'_first_failure_hist',np.bincount(first[any_],minlength=101))
        nearest=np.argmin(np.where(mask,np.abs(np.arange(t)-t//2),999),1)
        self.add(group,prefix+'_nearest_failure_hist',np.bincount(nearest[any_],minlength=t))
        self.add(group,prefix+'_offset_hist',mask.sum(0))


def analyze_masks(path,context,config,out):
    """Stream all original cluster masks; Test is descriptive ONLY.

    Policy comparisons exclude Test. Train-only donor masks are retained by
    frequency, not cherry-picked for low errors. No original outputs are edited.
    """
    agg=Aggregate();policies=Counter();denominators=Counter();donors={};references=[]
    frame_n=np.zeros(len(context.ids),int);frame_full=frame_n.copy()
    frame_obs=np.zeros(len(context.ids));frame_rel=frame_obs.copy()
    total=outside=0
    use=['frame','period','radius_m','cluster_id','n_players','strict_quality_valid',
         'record_mask_hex','observed_mask_hex','reliable_mask_hex','cluster_centroid_x','cluster_centroid_y',
         'cluster_max_pairwise_distance_m',*FEATURES]
    for ch in pd.read_csv(path,usecols=use,dtype={f'{k}_mask_hex':str for k in ('record','observed','reliable')},chunksize=config['chunk_windows']):
        total+=len(ch);official=ch.period.to_numpy()>0;outside+=int((~official).sum())
        # Audit every original row including out-of-play rows, without reading Test features.
        rows,idx,masks,code,same=context.audit(ch)
        roles=context.roles[rows];radii=ch.radius_m.to_numpy(int)
        for radius in (3,5):
            train=(roles=='train')&(radii==radius)
            for basis in ('record','observed','reliable'):
                packed=np.packbits(~masks[basis][train],axis=1,bitorder='little')
                counts=donors.setdefault((radius,basis),Counter())
                counts.update(bytes(v) for v in packed)
        full=masks['reliable'].all(1);val=(roles=='validation')&full
        # Disjoint 500-frame support blocks; a retained +/-50 support never crosses a block.
        block=(ch.frame.to_numpy()-context.split_first[rows])//config['bootstrap_block_frames']
        block_first=context.split_first[rows]+block*config['bootstrap_block_frames']
        block_last=np.minimum(block_first+config['bootstrap_block_frames']-1,context.split_last[rows])
        contained=(ch.frame.to_numpy()-50>=block_first)&(ch.frame.to_numpy()+50<=block_last)
        eligible=val&contained
        if eligible.any():
            ref=ch.loc[eligible,[x for x in use if not x.endswith('_hex')]].copy()
            ref['support_block']=context.periods[rows[eligible]]*1000+block[eligible]
            references.append(ref)
        np.add.at(frame_n,rows[official],1);np.add.at(frame_full,rows[official],full[official])
        np.add.at(frame_obs,rows[official],masks['observed'][official].mean(1))
        np.add.at(frame_rel,rows[official],masks['reliable'][official].mean(1))
        ch=ch.loc[official].reset_index(drop=True);rows=rows[official];idx=idx[official]
        roles=roles[official];code=code[official];same=same[official]
        masks={k:v[official] for k,v in masks.items()}
        for group,sel in strata(ch,roles).items():
            if not sel.any():continue
            c=code[sel];n=len(c);a='all';agg.add(group,'windows',n)
            agg.add(group,'window_failure_code_hist',np.bincount(np.bitwise_or.reduce(c,axis=1),minlength=32))
            agg.add(group,'offset_failure_code_hist',np.bincount(c.ravel(),minlength=32))
            for b,name in enumerate(CAUSES):
                bad=(c&(1<<b))>0;agg.failure(group,name,bad)
                agg.add(group,name+'_only_windows',((np.bitwise_or.reduce(c,axis=1)==(1<<b))).sum())
                agg.add(group,name+'_only_offsets',(c==(1<<b)).sum())
                agg.add(group,name+'_remove_factor_full',((c&(31^(1<<b)))==0).all(1).sum())
            agg.add(group,'remove_both_qc_full',((c&(31^(4|8)))==0).all(1).sum())
            for basis,mask in masks.items():
                m=mask[sel];agg.add(group,basis+'_full',m.all(1).sum())
                agg.failure(group,basis+'_missing',~m)
                # Region runs are clipped to region; whole-window runs above are not clipped.
                for region,sl in REGIONS.items():
                    region_valid=m[:,sl];agg.add(group,f'{basis}_{region}_valid_offsets',region_valid.sum())
                    agg.failure(group,f'{basis}_{region}_missing',~region_valid)
                agg.add(group,basis+'_future_minus_past_missing',((~m[:,51:]).sum(1)-(~m[:,:50]).sum(1)).sum())
                asymmetry=(~m[:,51:]).sum(1)-(~m[:,:50]).sum(1)
                agg.add(group,basis+'_future_minus_past_hist',np.bincount(asymmetry+50,minlength=101))
                agg.add(group,basis+'_center21_missing',(~m[:,40:61]).sum())
                agg.add(group,basis+'_outer50_missing',(~m[:,np.r_[0:25,76:101]]).sum())
            excluded=masks['observed'][sel]&~masks['reliable'][sel]
            agg.add(group,'observable_but_excluded_windows',excluded.any(1).sum())
            agg.add(group,'observable_but_excluded_offsets',excluded.sum())
            agg.add(group,'count_change_offsets',(~same[sel]&masks['record'][sel]).sum())
            for region,sl in REGIONS.items():agg.add(group,f'count_change_{region}_offsets',(~same[sel,sl]&masks['record'][sel,sl]).sum())
            ideal=context.ids[rows[sel,None]]+np.arange(-50,51)
            sf=context.split_first[rows[sel,None]];slast=context.split_last[rows[sel,None]]
            # Half limits derive from manifest, whereas split limits above are narrower.
            period=context.periods[rows[sel]]
            pf=np.array([context.half_bounds[p][0] for p in period])[:,None]
            pl=np.array([context.half_bounds[p][1] for p in period])[:,None]
            missing=~masks['record'][sel]
            half=missing&((ideal<pf)|(ideal>pl))
            split=missing&~half&((ideal<sf)|(ideal>slast))
            gap=missing&~half&~split
            for name,mask in [('half_boundary',half),('split_boundary',split),('record_gap',gap)]:
                agg.add(group,name+'_windows',mask.any(1).sum());agg.add(group,name+'_offsets',mask.sum())
            ri=idx[sel];rec=masks['record'][sel]
            adjacent=rec[:,1:]&rec[:,:-1]
            changes=(context.nh[ri[:,1:]]!=context.nh[ri[:,:-1]])|(context.na[ri[:,1:]]!=context.na[ri[:,:-1]])
            agg.add(group,'adjacent_count_change_offsets',(changes&adjacent).sum())
            agg.add(group,'adjacent_record_pairs',adjacent.sum())
        # Hypothetical coverage comparisons NEVER use Test.
        for role in ('train','validation'):
            for r in (3,5):
                for q in ('strict','low'):
                    sel=(roles==role)&(ch.radius_m.to_numpy()==r)&(ch.strict_quality_valid.to_numpy()==(q=='strict'))
                    if not sel.any():continue
                    changed=((~same)&masks['record']).any(1)[sel]
                    for basis in ('observed','reliable'):
                        group=(role,r,q,basis);denominators[group]+=int(sel.sum())
                        preds=policy_predicates(masks[basis][sel],config,current_strict=ch.strict_quality_valid.to_numpy()[sel],
                             count_changed=changed,center_relation=masks['observed'][sel,50])
                        for (p,g),v in preds.items():policies[(*group,p,g)]+=int(v.sum())
    output={'total_cluster_rows':total,'outside_official_rows':outside,'official_rows':total-outside,
            'raw_frame_gaps':int((np.diff(context.ids)!=1).sum()),'mask_audit_exact':True,
            'aggregates':agg.data}
    (out/'missingness_summary.json').write_text(json.dumps(output,default=lambda x:x.tolist(),indent=2))
    rows=[dict(split=k[0],radius=k[1],quality=k[2],basis=k[3],percent=k[4],gate=k[5],
               eligible=v,total=denominators[k[:4]],coverage=v/denominators[k[:4]]) for k,v in policies.items()]
    pd.DataFrame(rows).to_csv(out/'actual_policy_coverage.csv',index=False)
    pd.DataFrame(dict(frame=context.ids,n_clusters=frame_n,full_reliable_clusters=frame_full,
        observed_offset_mean=np.divide(frame_obs,frame_n,out=np.full(len(frame_n),np.nan),where=frame_n>0),
        reliable_offset_mean=np.divide(frame_rel,frame_n,out=np.full(len(frame_n),np.nan),where=frame_n>0))).to_csv(out/'frame_missingness_summary.csv',index=False)
    references=pd.concat(references,ignore_index=True)
    return output,donors,references
