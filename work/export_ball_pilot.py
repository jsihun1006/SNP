import csv,json,hashlib,subprocess,math
from pathlib import Path
from collections import Counter
import cv2,numpy as np,imageio_ffmpeg

ROOT=Path(__file__).resolve().parent;WORK=ROOT/'ball_pilot';OUT=ROOT.parent/'outputs/ball-correction-3min';OUT.mkdir(exist_ok=True)
doc=json.loads((WORK/'candidates_enriched.json').read_text());track=json.loads((WORK/'corrected.json').read_text())['records'];ctx=np.load(WORK/'context.npz')
rank=json.loads((ROOT/'ranked_windows.json').read_text());recovery=json.loads((WORK/'recovery_report.json').read_text())
original=json.loads((WORK/'candidates.json').read_text())['records']
reviewed=[18919,19069,19369,19519,19669,20269,20419,20569,20719,21019,21319,21619,21769,21919,22069,22219,22369,23269]
audit=[]
for frame in reviewed:
    row=track[frame-doc['start']]
    assert row['status']=='selected_observation'
    row['visual_reviewed']=True
    audit.append(dict(source_frame=frame,assessment='ball-like selected observation in context and magnified crop',reviewer='assistant_visual',localization_ground_truth=False))
# This single exact frame was visibly the large patterned ball in the replay.
row=track[20119-doc['start']]
assert row['status']=='review_candidate'
row.update(status='visually_reviewed_observation',pixel_x=row.pop('provisional_pixel_x'),pixel_y=row.pop('provisional_pixel_y'),visual_reviewed=True,needs_review=False)
audit.append(dict(source_frame=20119,assessment='visible patterned ball; exact-frame provisional candidate accepted by visual inspection only',reviewer='assistant_visual',localization_ground_truth=False))
accepted={'selected_observation','visually_reviewed_observation'}
fields=['source_frame','local_frame','video_seconds','status','pixel_x','pixel_y','pitch_x_normalized','pitch_y_normalized','pitch_x_m','pitch_y_m','candidate_confidence','origin','candidate_index','visual_reviewed','needs_review','provisional_pixel_x','provisional_pixel_y']
with (OUT/'corrected_ball_coordinates.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for i,r in enumerate(track):
        r['visual_reviewed']=r.get('visual_reviewed',False)
        r.update(video_seconds=r['source_frame']/25,pitch_x_normalized=None,pitch_y_normalized=None,pitch_x_m=None,pitch_y_m=None)
        if r['pixel_x'] is not None and np.isfinite(ctx['H'][i]).all():
            ray=np.linalg.inv(ctx['H'][i])@np.r_[r['pixel_x'],r['pixel_y'],1]
            if abs(ray[2])>1e-12:
                x,y=map(float,ray[:2]/ray[2]);r.update(pitch_x_normalized=x,pitch_y_normalized=y,pitch_x_m=x*105.3,pitch_y_m=y*68)
        writer.writerow({key:r.get(key) for key in fields})
with (OUT/'frame_decisions.jsonl').open('w',encoding='utf-8') as f:
    for i,r in enumerate(track):f.write(json.dumps(dict(**r,original_candidate_count=len(original[i]['candidates']),all_candidates=doc['records'][i]['candidates']),allow_nan=False)+'\n')

states=Counter(r['status'] for r in track)
summary=dict(source_range_frames=[doc['start'],doc['end']],start_mp4=rank['selected']['start'],end_mp4_exclusive=rank['selected']['end_exclusive'],duration_seconds=180,fps=25,total_frames=4500,
             selection_basis=rank['method'],rank_metrics=rank['selected']['metrics'],original_candidates=sum(len(r['candidates']) for r in original),
             targeted_redetection=recovery,statuses=dict(states),accepted_observation_frames=sum(r['status'] in accepted for r in track),
             accepted_recovered_frames=sum(r['status'] in accepted and r.get('origin')=='targeted_redetection' for r in track),
             visually_reviewed_exact_frames=len(audit),remaining_review_or_unresolved=sum(r['status'] not in accepted for r in track),
             no_interpolated_or_atd_copied_coordinates=True,accuracy_measured_against_ground_truth=False,
             limitations=['Window chosen by suspicion score, not known error rate.','Automatic selections are not all verified.','ATD is a supplied automatic weak prior, not truth.','Video examples and 19 exact frames visually checked, not all4500.','H ground projection does not recover airborne3D positions.'])
(OUT/'report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(OUT/'visual_review.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
(WORK/'final_track.json').write_text(json.dumps(dict(start=doc['start'],end=doc['end'],records=track)),encoding='utf-8')

# Video: unmodified raw detections on the left; selected/review decisions right.
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
video=OUT/'ball_correction_before_after_3min.mp4'
command=[ffmpeg,'-hide_banner','-y','-loglevel','warning','-f','rawvideo','-pix_fmt','bgr24','-s','1920x720','-r','25','-i','pipe:0',
         '-ss',str(doc['start']/25),'-t','180','-i',r'H:\data\broadcast\BAR-ATH_broadcast.mp4',
         '-map','0:v:0','-map','1:a?','-c:v','h264_nvenc','-preset','p5','-cq','19','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k','-shortest','-movflags','+faststart',str(video)]
log=(WORK/'encode.log').open('w',encoding='utf-8')
process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
cap=cv2.VideoCapture(r'H:\data\broadcast\BAR-ATH_broadcast.mp4');assert cap.set(cv2.CAP_PROP_POS_FRAMES,doc['start'])
def put(image,text,p,color=(240,240,240),scale=.65):cv2.putText(image,text,p,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
for i,r in enumerate(track):
    ok,image=cap.read();assert ok
    left=cv2.resize(image,(960,540));right=left.copy()
    for c in original[i]['candidates']:
        p=tuple(np.int32(np.asarray(c['point'])/2));cv2.circle(left,p,6,(0,155,255),1,cv2.LINE_AA)
        put(left,f"{c['confidence']:.2f}",(min(903,p[0]+8),max(45,p[1]-8)),(0,180,255),.45)
    color=(20,235,80) if r['status'] in accepted else (0,185,255)
    point=None
    if r['status'] in accepted:point=(r['pixel_x'],r['pixel_y'])
    elif r['status']=='review_candidate':point=(r['provisional_pixel_x'],r['provisional_pixel_y'])
    if point is not None:
        p=tuple(np.int32(np.asarray(point)/2));cv2.circle(right,p,10,color,2,cv2.LINE_AA);cv2.line(right,(p[0],max(40,p[1]-40)),(p[0],p[1]-12),color,2)
    canvas=np.full((720,1920,3),20,np.uint8);canvas[:540,:960]=left;canvas[:540,960:]=right
    cv2.rectangle(canvas,(0,0),(1920,34),(20,20,20),-1)
    put(canvas,'BEFORE: all original ball candidates',(18,24));put(canvas,'AFTER: selected observation / amber = review only',(978,24))
    seconds=r['source_frame']/25
    put(canvas,f"Original video {int(seconds//60):02d}:{seconds%60:05.2f} | source frame {r['source_frame']} | pilot {i/25:06.2f}/180s",(20,574))
    put(canvas,f"Original candidates: {len(original[i]['candidates'])} | {summary['original_candidates']:,} across this clip",(20,611),(0,180,255))
    label='SELECTED OBSERVATION' if r['status'] in accepted else 'REVIEW - NO ACCEPTED COORDINATE' if r['status']=='review_candidate' else 'UNRESOLVED - NO COORDINATE'
    put(canvas,label,(980,574),color)
    if point is not None:
        put(canvas,f"x={point[0]:.1f}px y={point[1]:.1f}px score={r['candidate_confidence']:.3f}",(980,611),color)
        x,y=map(int,point);d=45;crop=image[max(0,y-d):min(1080,y+d),max(0,x-d):min(1920,x+d)]
        if crop.size:
            crop=cv2.resize(crop,(144,144));cv2.rectangle(crop,(0,0),(143,143),color,2);canvas[552:696,1756:1900]=crop
    put(canvas,'No interpolation. Missing observations stay unknown. ATD is a weak search hint, not ground truth.',(20,652),(180,180,180),.58)
    put(canvas,'Green is an automated selection unless specifically visually reviewed; it is not a 100% accuracy claim.',(20,686),(180,180,180),.58)
    process.stdin.write(canvas.tobytes())
    if i in (0,1500,3000):cv2.imwrite(str(OUT/f'comparison-{i//1500+1}.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,95])
    if i%500==0:print(json.dumps(dict(rendered=i,total=4500)),flush=True)
cap.release();process.stdin.close();returncode=process.wait();log.close()
if returncode:raise RuntimeError((WORK/'encode.log').read_text())
test=cv2.VideoCapture(str(video));n=int(test.get(cv2.CAP_PROP_FRAME_COUNT));fps=test.get(cv2.CAP_PROP_FPS);test.release()
assert n==4500 and abs(fps-25)<1e-6
summary.update(video_frames=n,video_fps=fps,video_bytes=video.stat().st_size,csv_sha256=hashlib.sha256((OUT/'corrected_ball_coordinates.csv').read_bytes()).hexdigest())
(OUT/'report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary),flush=True)
