"""Conservative IDATR-inspired pilot refinement, explicitly not a paper replica."""
from collections import Counter,defaultdict
import numpy as np
from .io import csv_write,json_write


def vote(rows,key):
    values=[str(r[key]) for r in rows if r.get(key) not in ('',None,'100')]
    return Counter(values).most_common(1)[0][0] if values else None


def incompatible(a,b):
    return any(vote(a,k) is not None and vote(b,k) is not None and vote(a,k)!=vote(b,k)
               for k in ('role','jersey','team_raw'))


def split(rows,gap):
    rows=sorted(rows,key=lambda r:int(r['frame_idx']))
    chunks=[];start=0
    for i in range(1,len(rows)):
        if int(rows[i]['frame_idx'])-int(rows[i-1]['frame_idx'])>gap:
            chunks.append(rows[start:i]);start=i
    chunks.append(rows[start:])
    result=[]
    for chunk in chunks:
        if result and not incompatible(result[-1],chunk):result[-1]+=chunk
        else:result.append(chunk)
    return result


def refine(rows,out,cfg,fps,fields):
    grouped=defaultdict(list)
    for r in rows:grouped[r['track_id_raw']].append(r)
    tracks=[chunk for group in grouped.values() for chunk in split(group,cfg.get('refine_gap_frames',25))]
    input_count=len(tracks)
    meta=__import__('json').loads((out/'features.json').read_text())
    if meta['count']:
        features=np.memmap(out/'features.float32',dtype='<f4',mode='r',shape=(meta['count'],meta['dimension']))
        def mean(t):return np.mean(features[[r['feature_index'] for r in t]],axis=0)
        while True:
            best=None
            means=[mean(t) for t in tracks]
            for i,a in enumerate(tracks):
                for j in range(i+1,len(tracks)):
                    b=tracks[j]
                    if a[0]['camera_segment_id']!=b[0]['camera_segment_id'] or incompatible(a,b):continue
                    af={r['frame_idx'] for r in a};bf={r['frame_idx'] for r in b}
                    if af&bf:continue
                    # Pilot adapter only merges disjoint chronological intervals.
                    first,second=(a,b) if a[-1]['frame_idx']<b[0]['frame_idx'] else (b,a)
                    gap=second[0]['frame_idx']-first[-1]['frame_idx']
                    if gap<=0 or gap>cfg.get('refine_merge_gap_frames',75):continue
                    l,r=first[-1],second[0]
                    if not l['homography_valid'] or not r['homography_valid']:continue
                    dist=np.hypot(l['pitch_x_m']-r['pitch_x_m'],l['pitch_y_m']-r['pitch_y_m'])
                    limit=cfg.get('refine_position_slack_m',1.)+cfg.get('refine_max_speed_mps',12.)*gap/fps
                    if dist>limit:continue
                    cosine=float(1-np.dot(means[i],means[j]))
                    if cosine<cfg.get('refine_cosine_threshold',.4) and (best is None or cosine<best[0]):best=(cosine,i,j)
            if best is None:break
            _,i,j=best
            tracks[i]=sorted(tracks[i]+tracks[j],key=lambda r:r['frame_idx']);tracks.pop(j)
        del features
    final=[];lineage={}
    for i,t in enumerate(tracks,1):
        key=f's{t[0]["camera_segment_id"]}:r{i}'
        lineage[key]=sorted({r['track_id_raw'] for r in t})
        for r in t:
            final.append({**r,'track_id':key})
    csv_write(out/'tracks_refined.csv',sorted(final,key=lambda r:(r['frame_idx'],r['track_id'])),fields)
    json_write(out/'refinement.json',dict(method='IDATR-inspired, bounded gap + metre speed constraint; not original IDATR',
        original_online_tracks=len(grouped),after_split=input_count,after_merge=len(tracks),lineage=lineage,
        attributes_not_overwritten=True,performance_validated=False))
