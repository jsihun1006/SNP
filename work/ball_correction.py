import json,math
from pathlib import Path
from collections import Counter
import numpy as np

WORK=Path(__file__).resolve().parent/'ball_pilot'

def evidence(candidate,prior,h_error):
    score=float(candidate['confidence']);box=candidate['bbox_xyxy']
    diag=math.hypot(box[2]-box[0],box[3]-box[1])
    d=candidate.get('atd_distance_px')
    sigma=max(25.,min(70.,diag*2))
    support=math.exp(-.5*(d/sigma)**2) if d is not None and h_error>=0 and h_error<=2 else 0.
    spot=candidate.get('penalty_spot_distance_m',1e9)
    # This is a soft penalty, overcome by observation/prior evidence, never a mask.
    mark_penalty=1.65*math.exp(-.5*(spot/.9)**2)*(1-support) if h_error>=0 and h_error<=2 else 0.
    green=candidate.get('green_fraction',.5)
    border=(candidate['point'][1]<190 or candidate['point'][1]>1040)
    pitch=candidate.get('pitch')
    off_field=bool(pitch is not None and (pitch[0]<-.025 or pitch[0]>1.025 or pitch[1]<-.025 or pitch[1]>1.025))
    background_penalty=.8*(1-green) if (border or off_field) else 0.
    unary=2.8*score-.9+1.1*support-mark_penalty-background_penalty
    return unary,dict(prior_support=support,soft_field_mark_penalty=mark_penalty,background_penalty=background_penalty)

def solve(records,context):
    n=len(records);prev_values=np.array([0.]);prev_candidates=[];backs=[];unaries=[];features=[]
    for i,record in enumerate(records):
        candidates=record['candidates']
        info=[evidence(c,context['atd_pixel'][i],context['h_error'][i]) for c in candidates]
        u=np.array([0.]+[v[0] for v in info]);unaries.append(u);features.append([v[1] for v in info])
        costs=np.zeros((len(prev_values),len(u)))
        if i and not context['cuts'][i]:
            costs[1:,0]=-.25;costs[0,1:]=-.35
            warp=context['warps'][i]
            for p,old in enumerate(prev_candidates):
                predicted=warp@np.r_[old['point'],1.]
                for q,new in enumerate(candidates):
                    b=new['bbox_xyxy'];diag=math.hypot(b[2]-b[0],b[3]-b[1])
                    dist=np.linalg.norm(predicted-new['point'])
                    scale=max(24.,min(90.,diag*1.5))
                    costs[p+1,q+1]=-.65*min(7.,(dist/scale)**2)
        values=prev_values[:,None]+costs
        pred=np.argmax(values,axis=0)
        current=values[pred,np.arange(len(u))]+u
        current-=np.max(current)
        backs.append(pred);prev_values=current;prev_candidates=candidates
    selected=np.empty(n,dtype=int);state=int(np.argmax(prev_values))
    for i in range(n-1,-1,-1):selected[i]=state-1;state=int(backs[i][state])
    out=[]
    for i,(record,index) in enumerate(zip(records,selected)):
        item=dict(source_frame=record['frame'],local_frame=i,candidate_index=int(index),status='unresolved',pixel_x=None,pixel_y=None,
                  candidate_confidence=None,evidence=None,manually_verified=False)
        if index>=0:
            c=record['candidates'][index];feat=features[i][index]
            unary=unaries[i][index+1]
            near_cut=any(context['cuts'][max(0,i-2):min(n,i+3)])
            temporal=(i>0 and selected[i-1]>=0) or (i+1<n and selected[i+1]>=0)
            strong=(c['confidence']>=.55 and feat['soft_field_mark_penalty']<.8) or (feat['prior_support']>=.55 and c['confidence']>=.18)
            if unary>=.2 and (strong or (temporal and c['confidence']>=.3 and feat['soft_field_mark_penalty']<.5)):
                item.update(status='selected_observation',pixel_x=c['point'][0],pixel_y=c['point'][1],candidate_confidence=c['confidence'],evidence=feat,
                            origin=c.get('origin','original_detector'),near_camera_cut=bool(near_cut),needs_review=bool(not strong or near_cut or (c.get('atd_distance_px') is not None and c['atd_distance_px']>120)))
            else:
                item.update(candidate_index=-1,needs_review=True)
        else:item['needs_review']=True
        if item['status']=='selected_observation' and item['needs_review']:
            item.update(status='review_candidate',provisional_pixel_x=item['pixel_x'],provisional_pixel_y=item['pixel_y'],pixel_x=None,pixel_y=None)
        out.append(item)
    return out

if __name__=='__main__':
    source=WORK/'candidates_enriched.json'
    if not source.exists():source=WORK/'candidates.json'
    doc=json.loads(source.read_text(encoding='utf-8'));context=np.load(WORK/'context.npz')
    result=solve(doc['records'],context)
    (WORK/'corrected.json').write_text(json.dumps(dict(start=doc['start'],end=doc['end'],records=result)),encoding='utf-8')
    print(json.dumps(dict(statuses=dict(Counter(x['status'] for x in result)),review=sum(x['needs_review'] for x in result),selected_recovery=sum(x.get('origin')=='targeted_redetection' for x in result))))
