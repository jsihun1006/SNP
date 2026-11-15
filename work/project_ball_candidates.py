import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
OUT = BASE/'outputs'
DERIVED = Path(r'H:\data\ball-tracking-results\first_half\homography')
DERIVED.mkdir(exist_ok=True)
HFILE = Path(r'H:\data\broadcast\BAR-ATH_broadcast_homography.csv')
SOURCE = DERIVED.parent/'candidates.jsonl'
LENGTH, WIDTH = 105.3, 68.0
hs = {}
h_count = 0
bad_h = 0
with HFILE.open(newline='',encoding='utf-8-sig') as stream:
    for row in csv.DictReader(stream):
        frame = int(row['frame'])
        assert frame == h_count
        h_count += 1
        h = np.array([float(row[f'h{i}{j}']) for i in range(3) for j in range(3)]).reshape(3,3)
        error = float(row['reprojection_error'])
        valid = int(float(row['valid'])) == 1 and np.isfinite(h).all()
        inv = None
        if valid:
            try: inv = np.linalg.inv(h)
            except np.linalg.LinAlgError: bad_h += 1
        hs[frame] = (h, inv, error)

counts = Counter()
frame_counts = Counter()
histogram = np.zeros((9,27,42),dtype=np.int64)
samples = []
selected = {}
roundtrip_max = 0.
columns = ['source_frame','part_frame','video_seconds','candidate_index','pixel_x','pixel_y',
           'confidence','source','h_frame','h_valid','reprojection_error','h_quality_known',
           'pitch_x_normalized','pitch_y_normalized','pitch_x_m','pitch_y_m','inside_pitch','status']
target = DERIVED/'ball_candidates_pitch.csv'
sample_target_starts = [10000,30000,50000,70000]
with SOURCE.open(encoding='utf-8') as stream, target.open('w',newline='',encoding='utf-8') as result:
    writer=csv.writer(result)
    writer.writerow(columns)
    expected=7171
    for line in stream:
        record=json.loads(line)
        frame=record['source_frame']
        assert frame==expected
        expected+=1
        h, inv, error=hs[frame]
        hv=inv is not None
        known=hv and np.isfinite(error) and error>=0
        frame_counts['all']+=1
        frame_counts['valid_h' if hv else 'no_h']+=1
        if hv and not known:frame_counts['h_quality_unknown']+=1
        shown=[]
        outside=0
        raw=[]
        for index,candidate in enumerate(record['candidates']):
            px,py=map(float,candidate['point'])
            score=float(candidate['confidence'])
            status='no_valid_homography'
            normalized_x=normalized_y=mx=my=''
            inside=False
            if hv:
                ray=inv@np.array([px,py,1.])
                if np.isfinite(ray).all() and abs(ray[2])>1e-12:
                    normalized_x,normalized_y=map(float,ray[:2]/ray[2])
                    mx,my=normalized_x*LENGTH,normalized_y*WIDTH
                    inside=0<=normalized_x<=1 and 0<=normalized_y<=1
                    status='on_pitch' if inside else 'outside_pitch'
                    back=h@np.array([normalized_x,normalized_y,1.])
                    if abs(back[2])>1e-12:
                        roundtrip_max=max(roundtrip_max,float(np.linalg.norm(back[:2]/back[2]-[px,py])))
                    if inside and known:
                        for threshold_index in range(9):
                            threshold=(threshold_index+1)/10
                            if score>=threshold:
                                histogram[threshold_index,min(26,int(normalized_y*27)),min(41,int(normalized_x*42))]+=1
                        shown.append([round(mx,3),round(my,3),round(score,6)])
                    elif not inside:outside+=1
                else:status='projection_at_infinity'
            counts[status]+=1
            writer.writerow([frame,record['part_frame'],record['video_seconds'],index,px,py,score,
                             candidate['source'],frame,int(hv),error,int(known),normalized_x,normalized_y,
                             mx,my,int(inside),status])
            raw.append(dict(candidate,projected_x=mx,projected_y=my,inside_pitch=inside))
        if (frame-7171)%25==0:
            samples.append([frame,round(error,4) if known else None,int(hv),len(raw),outside,shown])
        for anchor in sample_target_starts:
            if anchor not in selected and anchor<=frame<anchor+1000 and known and error<1 and shown and any(p[2]>=.5 for p in shown):
                selected[anchor]=dict(frame=frame,h=h.tolist(),error=error,candidates=raw)
    assert expected==86638

assert sum(counts.values())==244334
with target.open('rb') as source,gzip.open(OUT/'ball-candidates-pitch.csv.gz','wb',compresslevel=6) as zipped:
    for chunk in iter(lambda:source.read(1048576),b''):zipped.write(chunk)
summary=dict(candidate_counts=dict(counts),frame_counts=dict(frame_counts),homography_rows=h_count,
             actual_video_frames=164487,unused_tail_homography_rows=1,offset_applied=0,
             direction='inverse of normalized-pitch-to-image H[source_frame]',pitch_size_m=[LENGTH,WIDTH],
             point='original-image bounding-box center',roundtrip_max_error_pixels=roundtrip_max,
             output_csv=str(target),csv_bytes=target.stat().st_size,
             visualization_sampling_fps=1,visualization_samples=len(samples),
             visualization_exclusions='No H, nonfinite/infinite projection, outside pitch, or unknown/negative homography error. No interpolation/clipping.',
             h_sha256=hashlib.sha256(HFILE.read_bytes()).hexdigest())
(OUT/'homography-conversion-report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(BASE/'work/pitch-viz-data.json').write_text(json.dumps(dict(samples=samples,histogram=histogram.tolist(),summary=summary),separators=(',',':')),encoding='utf-8')
(BASE/'work/pitch-overlay-samples.json').write_text(json.dumps(list(selected.values())),encoding='utf-8')

# Same-frame pitch lines and raw candidate boxes support a visual alignment check.
cap=cv2.VideoCapture(r'H:\data\broadcast\BAR-ATH_broadcast.mp4')
for i,record in enumerate(selected.values()):
    frame=record['frame'];h=np.array(record['h'])
    assert cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
    ok,image=cap.read();assert ok
    overlay=image.copy()
    def draw_pitch_line(points):
        pts=np.column_stack([np.asarray(points),np.ones(len(points))])@h.T
        xy=pts[:,:2]/pts[:,2:]
        if np.isfinite(xy).all() and np.max(np.abs(xy))<1e7:
            cv2.polylines(overlay,[np.int32(np.round(xy))],False,(255,220,0),2,cv2.LINE_AA)
    for x in (0,.5,1):draw_pitch_line([(x,0),(x,1)])
    for y in (0,1):draw_pitch_line([(0,y),(1,y)])
    for x0,x1 in ((0,16.5/LENGTH),(1-16.5/LENGTH,1)):
        draw_pitch_line([(x0,(WIDTH-40.32)/2/WIDTH),(x1,(WIDTH-40.32)/2/WIDTH),(x1,(WIDTH+40.32)/2/WIDTH),(x0,(WIDTH+40.32)/2/WIDTH),(x0,(WIDTH-40.32)/2/WIDTH)])
    theta=np.linspace(0,2*np.pi,100)
    draw_pitch_line(np.column_stack([.5+9.15/LENGTH*np.cos(theta),.5+9.15/WIDTH*np.sin(theta)]))
    image=cv2.addWeighted(overlay,.7,image,.3,0)
    for candidate in record['candidates']:
        px,py=map(lambda x:int(round(x)),candidate['point'])
        color=(0,180,255) if candidate['inside_pitch'] else (70,70,245)
        cv2.circle(image,(px,py),9,color,2,cv2.LINE_AA)
        label=f"{candidate['confidence']:.2f}"
        if candidate['inside_pitch']:label+=f" ({candidate['projected_x']:.1f},{candidate['projected_y']:.1f})m"
        cv2.putText(image,label,(min(px+12,1500),max(35,py-12)),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2,cv2.LINE_AA)
    cv2.putText(image,f"Source frame {frame} | H error {record['error']:.3f}",(25,45),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2,cv2.LINE_AA)
    cv2.imwrite(str(OUT/f'homography-check-{i+1}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,90])
cap.release()
print(json.dumps(summary))
