"""Add an estimated football clock to the local comparison player, without re-encoding.

python starter/add_tracking_match_clock.py
Half anchors come from the existing player's frame mapping. No video OCR, GT,
Smart Tagging, source changes, server or external resources are used.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from starter.export_tracking_disagreement import sha

SOURCE=ROOT/'data/derived/tracking_disagreement_ball_20261113/tracking_disagreement_ball_player.html'
HALVES={'1st Half':('전반',0,45),'2nd Half':('후반',45*60,90)}


def starts_from_segments(segments):
    starts={}
    for s in segments:
        if s['half'] not in HALVES or int(s['count'])<1 or int(s['tactical_first'])<0:raise ValueError('Invalid clock mapping segment')
        starts[s['half']]=min(starts.get(s['half'],s['tactical_first']),s['tactical_first'])
    return starts


def mmss(seconds):return f'{seconds//60:02d}:{seconds%60:02d}'


def clock_info(frame,half,starts,fps=25):
    """Football convention: H1 starts0:00, H2 starts45:00, independent of H1 stoppage.

    Native tactical frame differences retain real gaps inside a half. Playback
    frame counts never define the clock. Display seconds are floored.
    """
    if half not in HALVES or half not in starts or not math.isfinite(fps) or fps<=0 or frame!=int(frame):raise ValueError('Invalid clock input')
    offset=int(frame)-int(starts[half])
    if offset<0:raise ValueError('Frame precedes this half anchor')
    elapsed=math.floor(offset/fps);label,base,stop=HALVES[half];extra=max(0,elapsed-45*60)
    return dict(half_label=label,half_elapsed_seconds=elapsed,half_elapsed_display=mmss(elapsed),
        match_clock_seconds=base+elapsed,match_clock_display=f'{stop}+{mmss(extra)}' if extra else mmss(base+elapsed),
        estimated_half_start_tactical_frame=int(starts[half]),stoppage_seconds=extra)


CLOCK_HELPERS_JS=r'''
function matchClockMMSS(seconds){return String(Math.floor(seconds/60)).padStart(2,'0')+':'+String(seconds%60).padStart(2,'0');}
function matchClockInfo(frame,half,starts,fps){
 const spec={'1st Half':{label:'전반',base:0,stop:45},'2nd Half':{label:'후반',base:2700,stop:90}};
 if(!spec[half]||!Object.prototype.hasOwnProperty.call(starts,half)||!Number.isInteger(frame)||!Number.isFinite(fps)||fps<=0)throw new Error('Invalid clock input');
 const offset=frame-starts[half];if(offset<0)throw new Error('Frame before half anchor');
 const elapsed=Math.floor(offset/fps),extra=Math.max(0,elapsed-2700),s=spec[half];
 return {half_label:s.label,half_elapsed_seconds:elapsed,half_elapsed_display:matchClockMMSS(elapsed),
  match_clock_seconds:s.base+elapsed,match_clock_display:extra?s.stop+'+'+matchClockMMSS(extra):matchClockMMSS(s.base+elapsed),
  estimated_half_start_tactical_frame:starts[half],stoppage_seconds:extra};
}
'''

CSS='''
#matchClockWrap{position:relative;width:100%}
#matchClockBar{position:sticky;top:0;z-index:10;display:flex;gap:14px;align-items:center;justify-content:space-between;flex-wrap:wrap;padding:12px 15px;background:#172b40;border:1px solid #59728c;box-sizing:border-box}
#matchClockMain{font-size:25px;font-weight:700;color:#ffe09a;font-variant-numeric:tabular-nums}
#matchClockDetail{font-size:13px;color:#c4d4e7;line-height:1.7}
#matchClockFullscreen{white-space:nowrap}
#matchClockWrap:fullscreen{background:#101820;display:flex;flex-direction:column;width:100vw;height:100vh}
#matchClockWrap:fullscreen #matchClockBar{position:static;width:100%;flex:0 0 auto}
#matchClockWrap:fullscreen video{flex:1;min-height:0;max-height:calc(100vh - 90px);object-fit:contain;width:100%}
'''

BAR='''<div id="matchClockWrap"><div id="matchClockBar" role="status" aria-live="off">
<div><div id="matchClockMain">경기시계 계산 중…</div><div id="matchClockDetail">기존 하프 시작점 기준 추정 · 영상 재생 시간과 구분</div></div>
<button id="matchClockFullscreen" type="button">시계와 함께 전체화면</button></div>
'''


def enhance_html(html):
    if 'id="matchClockWrap"' in html:raise ValueError('Player already has a clock; use the original source')
    mark='const meta=';where=html.find(mark)
    if where<0:raise ValueError('Player frame metadata missing')
    meta,_=json.JSONDecoder().raw_decode(html[where+len(mark):]);starts=starts_from_segments(meta['segments'])
    if meta['fps']<=0 or not meta['count']:raise ValueError('Invalid player metadata')
    videos=list(re.finditer(r'<video\b[^>]*id="video"[^>]*>.*?</video>',html,re.S))
    if len(videos)!=1 or '</style>' not in html:raise ValueError('Unsupported player layout')
    video=videos[0];body=video.group().replace('<video id="video"','<video id="video" controlsList="nofullscreen"',1)
    html=html[:video.start()]+BAR+body+'\n</div>'+html[video.end():]
    html=html.replace('</style>',CSS+'\n</style>',1)
    hook="el('seek').value=current;el('jump').value=s.tactical_first+step;"
    if html.count(hook)!=1 or html.count('function show(index)')!=1:raise ValueError('Player seek hook changed')
    ui=CLOCK_HELPERS_JS+'\nconst matchClockStarts='+json.dumps(starts,separators=(',',':'))+r''';
function updateMatchClock(frame,half){
 const c=matchClockInfo(frame,half,matchClockStarts,meta.fps);
 el('matchClockMain').textContent=c.half_label+' · 경기시계 '+c.match_clock_display;
 el('matchClockDetail').textContent=c.half_label+' 경과 '+c.half_elapsed_display+' · 하프 시작 Tactical '+c.estimated_half_start_tactical_frame+' 기준 추정';
}
el('matchClockFullscreen').onclick=async()=>{
 const wrap=el('matchClockWrap');
 try{if(document.fullscreenElement===wrap)await document.exitFullscreen();else if(wrap.requestFullscreen)await wrap.requestFullscreen();else if(wrap.webkitRequestFullscreen)wrap.webkitRequestFullscreen();else el('message').textContent='이 브라우저는 시계 포함 전체화면을 지원하지 않습니다.';}
 catch(error){el('message').textContent='전체화면을 열지 못했습니다. 일반 화면의 경기시계를 이용하세요.';}
};
'''
    html=html.replace('function show(index)',ui+'\nfunction show(index)',1)
    html=html.replace(hook,hook+"\n updateMatchClock(s.tactical_first+step,s.half);",1)
    note='''<p class="muted">경기시계는 각 하프의 기존 정렬 시작점을0:00/45:00으로 놓은 추정값입니다. 정확한 방송 전광판 시각이나 새 킥오프 판정이 아닙니다.
추가시간은45+MM:SS /90+MM:SS로 표시하며, 후반에는 전반 추가시간과 하프타임을 더하지 않습니다.
이 시계는 HTML 플레이어 표시입니다. 기존 MP4·선수 지표·공 지표는 그대로 유지합니다.</p>'''
    html=html.replace('<script>\n',note+'\n<script>\n',1)
    return html,meta,starts


def write_clock_timeline(source,output,meta,starts):
    segments=meta['segments'];seen=0;seg_index=0
    with Path(source).open(newline='') as f,Path(output).open('w',newline='') as out:
        reader=csv.DictReader(f);w=csv.writer(out)
        w.writerow(['video_frame','tactical_frame','broadcast_frame','half','half_elapsed_seconds','half_elapsed_display','match_clock_seconds','match_clock_display','estimated_half_start_tactical_frame'])
        for i,row in enumerate(reader):
            if int(row['video_frame'])!=i:raise ValueError('Player timeline order invalid')
            while seg_index<len(segments) and i>=segments[seg_index]['video_first']+segments[seg_index]['count']:seg_index+=1
            if seg_index>=len(segments):raise ValueError('Timeline longer than video')
            s=segments[seg_index];step=i-s['video_first'];t=int(row['tactical_frame']);b=int(row['broadcast_frame'])
            if step<0 or t!=s['tactical_first']+step or b!=s['broadcast_first']+step:raise ValueError('Source timeline differs from player segments')
            c=clock_info(t,s['half'],starts,meta['fps'])
            w.writerow([i,t,b,s['half'],c['half_elapsed_seconds'],c['half_elapsed_display'],c['match_clock_seconds'],c['match_clock_display'],c['estimated_half_start_tactical_frame']]);seen+=1
    if seen!=meta['count']:raise ValueError('Incomplete player timeline')
    return seen


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0]);p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--output',type=Path);args=p.parse_args()
    output=args.output or args.source.with_name('tracking_disagreement_ball_clock_player.html')
    if output.resolve()==args.source.resolve() or output.parent.resolve()!=args.source.parent.resolve():raise ValueError('Keep a separate clock HTML in the same folder as the original player')
    folder=args.source.parent;video=folder/'tracking_disagreement_with_ball.mp4';timeline=folder/'frame_timeline.csv'
    protected=[args.source,video,timeline,*[folder/name for name in ('ball_values.npy','ball_metrics.csv.gz','ball_summary.json') if (folder/name).exists()]]
    before={str(path):sha(path) for path in protected};html,meta,starts=enhance_html(args.source.read_text())
    count=write_clock_timeline(timeline,folder/'frame_timeline_with_clock.csv',meta,starts);output.write_text(html)
    after={str(path):sha(path) for path in protected}
    if before!=after:raise ValueError('Original comparison files changed')
    report=dict(rows=count,half_anchor_frames=starts,fps=meta['fps'],clock_basis='native tactical frame minus first aligned frame of each half; 0/45min base; estimated',
        mp4_reencoded=False,player_or_ball_metrics_modified=False,protected_inputs_unchanged=True,protected_sha256=before,local_only=True,
        HTML=str(output.relative_to(ROOT)),clock_timeline='frame_timeline_with_clock.csv')
    (folder/'match_clock_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(dict(rows=count,half_anchor_frames=starts,mp4_reencoded=False,protected_inputs_unchanged=True)),flush=True)


if __name__=='__main__':main()
