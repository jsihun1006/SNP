"""Render synchronized original-candidate / final-QA ball comparison videos."""
import argparse,gzip,hashlib,json,math,subprocess,time
from pathlib import Path
import cv2,numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont

BASE=Path(__file__).resolve().parent.parent
OUT=BASE/'outputs/full-correction'
SOURCE=Path(r'H:\data\broadcast\BAR-ATH_broadcast.mp4')
DECISIONS=Path(r'H:\data\ball-tracking-results\first_half\correction_recall_v1\qa_adjusted\frame_decisions.jsonl.gz')
FIRST,LAST,FPS=7171,86638,25
ACCEPTED={'selected_observation','visually_reviewed_observation'}
CLIPS=[(9000,9100,9042),(30360,30449,30442),(33150,33200,33172),(33310,33370,33329),(45690,45790,45711)]
GREEN=(65,225,95);AMBER=(0,180,255);GRAY=(170,170,170);BACKGROUND=(24,20,18)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def korean_patch(text,width,height,size=23,color=(235,235,235)):
    image=Image.new('RGB',(width,height),(18,20,24))
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',size)
    ImageDraw.Draw(image).text((8,1),text,font=font,fill=color)
    return np.asarray(image)[:,:,::-1].copy()

def template():
    canvas=np.full((720,1920,3),BACKGROUND,np.uint8)
    canvas[0:36,:960]=korean_patch('보정 전 · 원본 후보 전체 (굵은 원: 최고 점수)',960,36)
    canvas[0:36,960:]=korean_patch('보정 후 · 선택 / 검토 상태 (자동 선택은 미검증)',960,36)
    canvas[674:704,:800]=korean_patch('원본 후보는 확정된 공이 아닌 검출 후보입니다.',800,30,20,(175,175,175))
    canvas[674:704,960:1750]=korean_patch('초록 ○ 선택 · 주황 ○ 추가 검토 · 주황 ◇ 검토 후보',790,30,19,(190,190,190))
    return canvas

def clock(seconds):
    return f'{int(seconds//3600):02d}:{int(seconds//60)%60:02d}:{seconds%60:05.2f}'

def text(image,value,point,color=GRAY,scale=.58):
    cv2.putText(image,value,point,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)

def crop_on_footer(canvas,image,point,x,color):
    if point is None:return
    px,py=map(int,point);r=48
    crop=image[max(0,py-r):min(1080,py+r),max(0,px-r):min(1920,px+r)]
    if crop.size:
        crop=cv2.resize(crop,(128,128),interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(crop,(0,0),(127,127),color,2)
        canvas[584:712,x:x+128]=crop

def draw_frame(image,row,base,status_patches):
    left=cv2.resize(image,(960,540));right=left.copy()
    originals=[c for c in row['candidates'] if c.get('origin')=='original_detector']
    assert len(originals)==row['original_candidate_count']
    best=max(originals,key=lambda c:c['confidence']) if originals else None
    for candidate in originals:
        point=tuple(np.int32(np.round(np.asarray(candidate['point'])/2)))
        cv2.circle(left,point,8 if candidate is best else 5,AMBER,2 if candidate is best else 1,cv2.LINE_AA)
    decision=row['selection'];status=decision['status'];point=None
    if status in ACCEPTED:
        point=[decision['pixel_x'],decision['pixel_y']]
        key='visual' if status=='visually_reviewed_observation' else 'selected_review' if decision.get('needs_review') else 'selected'
        color=AMBER if key=='selected_review' else GREEN
        p=tuple(np.int32(np.round(np.asarray(point)/2)))
        cv2.circle(right,p,10,color,2,cv2.LINE_AA)
        cv2.line(right,(p[0],max(0,p[1]-35)),(p[0],max(0,p[1]-12)),color,2,cv2.LINE_AA)
    elif status=='review_candidate':
        key='review';color=AMBER
        if decision.get('provisional_pixel_x') is not None:
            point=[decision['provisional_pixel_x'],decision['provisional_pixel_y']]
            p=tuple(np.int32(np.round(np.asarray(point)/2)))
            cv2.drawMarker(right,p,color,cv2.MARKER_DIAMOND,20,2,cv2.LINE_AA)
    else:key='unknown';color=GRAY
    canvas=base.copy();canvas[36:576,:960]=left;canvas[36:576,960:]=right
    cv2.line(canvas,(960,0),(960,720),(70,70,70),1)
    frame=row['source_frame'];elapsed=(frame-FIRST)/25
    text(canvas,f'MP4 {clock(frame/25)} | frame {frame} | half elapsed {clock(elapsed)}',(18,602),GRAY,.56)
    text(canvas,f'{len(originals)} original candidates',(18,635),AMBER,.62)
    if best:
        text(canvas,f'Top score {best["confidence"]:.3f}  x={best["point"][0]:.1f} y={best["point"][1]:.1f}px',(18,662),AMBER,.56)
        crop_on_footer(canvas,image,best['point'],816,AMBER)
    canvas[581:613,976:1746]=status_patches[key]
    if point is not None:
        score=decision.get('candidate_confidence')
        text(canvas,f'x={point[0]:.1f} y={point[1]:.1f}px | score {score:.3f}',(986,640),color,.61)
        origin=decision.get('origin','')
        text(canvas,origin or '',(986,665),GRAY,.53)
        crop_on_footer(canvas,image,point,1776,color)
    else:text(canvas,'No observed coordinate / no interpolation',(986,640),GRAY,.57)
    return canvas

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);p.add_argument('--test',action='store_true');args=p.parse_args()
    OUT.mkdir(exist_ok=True)
    summary=json.loads((OUT/'completion_report.json').read_text(encoding='utf-8'))
    assert digest(DECISIONS)==summary['qa_decisions_sha256']
    frames_expected=min(args.limit,LAST-FIRST) if args.limit else LAST-FIRST
    suffix='_test' if args.test else ''
    full=OUT/f'before_after_first_half_full{suffix}.mp4';highlight=OUT/f'before_after_key_examples{suffix}.mp4'
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();log=(BASE/f'outputs/full-comparison{suffix}.encoder.log').open('w',encoding='utf-8')
    common=[ffmpeg,'-hide_banner','-y','-loglevel','warning','-f','rawvideo','-pix_fmt','bgr24','-s','1920x720','-r','25','-i','pipe:0']
    full_args=common+['-ss',str(FIRST/25),'-t',str(frames_expected/25),'-i',str(SOURCE),'-map','0:v:0','-map','1:a?',
         '-c:v','h264_nvenc','-preset','p5','-b:v','4M','-maxrate','5M','-bufsize','10M','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k','-movflags','+faststart',str(full)]
    main_encoder=subprocess.Popen(full_args,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
    high_encoder=None
    if not args.test:
        high_args=common+['-an','-c:v','h264_nvenc','-preset','p5','-b:v','4M','-maxrate','5M','-bufsize','10M','-pix_fmt','yuv420p','-movflags','+faststart',str(highlight)]
        high_encoder=subprocess.Popen(high_args,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
    base=template()
    # User-facing labels are Korean; this dictionary is deliberately pre-rendered once.
    names={'selected':'선택한 관측','selected_review':'선택한 관측 · 추가 검토 표시 있음','visual':'해당 프레임 시각 검수 반영','review':'검토 후보 · 최종 선택 좌표 아님','unknown':'미확정 · 좌표 없음'}
    status_patches={key:korean_patch(label,770,32,23,tuple(reversed(AMBER if key in ('review','selected_review') else GRAY if key=='unknown' else GREEN))) for key,label in names.items()}
    cap=cv2.VideoCapture(str(SOURCE));assert cap.set(cv2.CAP_PROP_POS_FRAMES,FIRST)
    count=high_count=0;started=time.perf_counter();original_observations=0;keys={key for _,_,key in CLIPS};snapshot_files=[]
    try:
        with gzip.open(DECISIONS,'rt',encoding='utf-8') as src:
            for line in src:
                if count>=frames_expected:break
                row=json.loads(line);assert row['source_frame']==FIRST+count
                ok,image=cap.read();assert ok
                canvas=draw_frame(image,row,base,status_patches)
                main_encoder.stdin.write(canvas.tobytes())
                original_observations+=row['original_candidate_count']
                frame=row['source_frame']
                if high_encoder is not None and any(a<=frame<b for a,b,_ in CLIPS):
                    high_encoder.stdin.write(canvas.tobytes());high_count+=1
                    if frame in keys:
                        freeze=canvas.copy();freeze[:36]=korean_patch(f'검수 프레임 정지 · 원본 프레임 {frame} (이 프레임의 판정만 표시)',1920,36,23)
                        for _ in range(50):high_encoder.stdin.write(freeze.tobytes())
                        high_count+=50
                if count==0 or frame in keys:
                    filename=f'comparison-video-frame-{frame}.jpg';cv2.imwrite(str(OUT/filename),canvas,[cv2.IMWRITE_JPEG_QUALITY,92]);snapshot_files.append(filename)
                count+=1
                if count%1000==0:
                    elapsed=time.perf_counter()-started;rate=count/elapsed
                    progress=dict(status='running',frames=count,total_frames=frames_expected,percent=count/frames_expected*100,processing_fps=rate,eta_seconds=(frames_expected-count)/rate,updated_unix=time.time())
                    (OUT/f'video-progress{suffix}.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress),flush=True)
    finally:
        cap.release();main_encoder.stdin.close()
        if high_encoder is not None:high_encoder.stdin.close()
    result_main=main_encoder.wait();result_high=high_encoder.wait() if high_encoder is not None else 0;log.close()
    if result_main or result_high:raise RuntimeError('Encoding failed; see encoder log')
    assert count==frames_expected
    def verify(path,expected):
        c=cv2.VideoCapture(str(path));n=int(c.get(cv2.CAP_PROP_FRAME_COUNT));fps=c.get(cv2.CAP_PROP_FPS);w=int(c.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert n==expected and fps==25 and (w,h)==(1920,720)
        for at in (0,n//2,n-1):
            c.set(cv2.CAP_PROP_POS_FRAMES,at);ok,image=c.read();assert ok and image.shape[:2]==(720,1920)
        c.release();return dict(frames=n,fps=fps,duration_seconds=n/fps,width=w,height=h,bytes=path.stat().st_size,first_middle_last_decode='passed')
    result=dict(status='complete',source_frame_range=[FIRST,FIRST+count],source_decisions_sha256=summary['qa_decisions_sha256'],full_video=verify(full,count),highlight_video=verify(highlight,high_count) if high_encoder is not None else None,highlight_source_intervals=CLIPS,highlight_extra_pause_frames_per_key=50,highlight_audio=False,original_candidate_observations=original_observations,elapsed_seconds=time.perf_counter()-started,snapshots=snapshot_files,legend='Before original-only candidates, thick=max score; after green=selected, amber circle=selected but flagged, amber diamond=provisional, no marker=unresolved. Automated selection is not confirmed truth.')
    if not args.test:assert original_observations==244334 and high_count==sum(b-a for a,b,_ in CLIPS)+50*len(CLIPS)
    (OUT/f'video-verification{suffix}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (OUT/f'video-progress{suffix}.json').write_text(json.dumps(dict(status='complete',frames=count,total_frames=frames_expected,eta_seconds=0,updated_unix=time.time()),indent=2))
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
