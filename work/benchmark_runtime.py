import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / 'source/SNP-ball-tracking'
sys.path.insert(0,str(PROJECT))
from ball_only.run import environment
environment()
import cv2
import torch
from ball_only.detector import BallDetector
from ball_only.runtime_control import RuntimeController

frames=[]
cap=cv2.VideoCapture(r'H:\data\broadcast\BAR-ATH_broadcast.mp4')
for start in (7171,30000,50000,80000):
    assert cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    for _ in range(8):
        ok, frame=cap.read();assert ok
        frames.append(frame)
cap.release()
detector=BallDetector(PROJECT/'ball_only/weights/yolo26x.pt',RuntimeController(target_low=98,target_high=99.5))
kwargs=dict(full_batch=1,roi_batch=4,tile_batch=8,max_rois=2)

def compare(a,b):
    failures=[]
    max_deltas=dict(coordinates=0.,confidence=0.)
    def check(x,y,path=''):
        if isinstance(x,dict):
            if not isinstance(y,dict) or x.keys()!=y.keys():failures.append(path+':keys');return
            for k in x:
                if k!='timing':check(x[k],y[k],path+'/'+k)
        elif isinstance(x,list):
            if not isinstance(y,list) or len(x)!=len(y):failures.append(path+':count');return
            for i,(xx,yy) in enumerate(zip(x,y)):check(xx,yy,path+'/'+str(i))
        elif isinstance(x,float):
            delta=abs(x-y)
            key='confidence' if 'confidence' in path else 'coordinates'
            max_deltas[key]=max(max_deltas[key],delta)
            if delta>(.0001 if key=='confidence' else .01):failures.append(path+':delta='+str(delta))
        elif x!=y:failures.append(path+':value')
    check(a,b)
    return dict(passed=not failures, failures=failures[:20],failure_count=len(failures),max_deltas=max_deltas,
                exact_without_timing=strip(a)==strip(b))

def strip(x):
    if isinstance(x,dict):return {k:strip(v) for k,v in x.items() if k!='timing'}
    if isinstance(x,list):return [strip(v) for v in x]
    return x

baseline=None
report=dict(frames=32,frame_starts=[7171,30000,50000,80000],configurations=[])
for name,threads,autotune,tile in (('baseline',2,False,8),('cpu1',1,False,8),('cpu4',4,False,8),('tile16',2,False,16),('autotune',2,True,8)):
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark=autotune
    torch.cuda.empty_cache()
    config=dict(name=name,cpu_threads=threads,cudnn_benchmark=autotune,tile_batch=tile)
    print(json.dumps(dict(event='starting',**config)),flush=True)
    timings=[];comparisons=[]
    kwargs['tile_batch']=tile
    detector.run_chunk(frames,**kwargs)
    # Predictor initialization resets CPU threads; set the measured value again.
    torch.set_num_threads(threads)
    torch.cuda.reset_peak_memory_stats()
    for repeat in range(3):
        started=time.perf_counter()
        rows=detector.run_chunk(frames,**kwargs)
        torch.cuda.synchronize()
        timings.append(time.perf_counter()-started)
        if baseline is None:baseline=rows
        comparisons.append(compare(baseline,rows))
    row=dict(**config,seconds=timings,fps=32/(sum(timings)/len(timings)),comparisons=comparisons,
             peak_allocated_mib=torch.cuda.max_memory_allocated()/1048576,
             peak_reserved_mib=torch.cuda.max_memory_reserved()/1048576)
    report['configurations'].append(row)
    (ROOT/'runtime-benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(row),flush=True)
valid=[r for r in report['configurations'] if all(c['passed'] for c in r['comparisons'])]
report['selected']=max(valid,key=lambda r:r['fps'])['name']
report['settings']=detector.settings()
(ROOT/'runtime-benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(dict(event='selected',name=report['selected'])),flush=True)
