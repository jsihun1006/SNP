"""Synthetic football clock boundaries and real local JavaScript seek execution."""
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import pytest

from starter.add_tracking_match_clock import clock_info,starts_from_segments,enhance_html,write_clock_timeline,CLOCK_HELPERS_JS
from starter.view_all_movement_vectors import HTML

STARTS={'1st Half':1000,'2nd Half':20000}


@pytest.mark.parametrize('half,seconds,expected',[
    ('1st Half',0,'00:00'),('1st Half',12*60+34,'12:34'),('1st Half',45*60,'45:00'),
    ('1st Half',45*60+135,'45+02:15'),('2nd Half',0,'45:00'),
    ('2nd Half',8*60+20,'53:20'),('2nd Half',45*60,'90:00'),('2nd Half',45*60+190,'90+03:10'),
])
def test_football_clock_and_stoppage_time(half,seconds,expected):
    c=clock_info(STARTS[half]+seconds*25,half,STARTS)
    assert c['match_clock_display']==expected and c['half_elapsed_seconds']==seconds


def test_no_halftime_addition_and_subsecond_floor():
    assert clock_info(STARTS['2nd Half'],'2nd Half',STARTS)['match_clock_seconds']==2700
    assert clock_info(STARTS['1st Half']+24,'1st Half',STARTS)['match_clock_display']=='00:00'
    assert clock_info(STARTS['1st Half']+25,'1st Half',STARTS)['match_clock_display']=='00:01'
    with pytest.raises(ValueError):clock_info(999,'1st Half',STARTS)
    with pytest.raises(ValueError):clock_info(1000,'1st Half',STARTS,fps=0)


def test_clock_uses_native_frame_and_does_not_restart_after_mapping_gap():
    segments=[dict(video_first=0,count=10,tactical_first=100,broadcast_first=0,half='1st Half'),
        dict(video_first=10,count=10,tactical_first=125,broadcast_first=25,half='1st Half')]
    starts=starts_from_segments(segments)
    assert starts['1st Half']==100 and clock_info(125,'1st Half',starts)['half_elapsed_seconds']==1
    with tempfile.TemporaryDirectory() as folder:
        source=Path(folder)/'timeline.csv';out=Path(folder)/'clock.csv'
        with source.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['video_frame','tactical_frame','broadcast_frame'])
            for i in range(20):w.writerow([i,100+i if i<10 else 115+i,i if i<10 else i+15])
        assert write_clock_timeline(source,out,dict(count=20,fps=25,segments=segments),starts)==20
        rows=list(csv.DictReader(out.open()));assert rows[10]['half_elapsed_seconds']=='1'


def synthetic_html():
    meta=dict(count=20000,fps=25,segments=[dict(video_first=0,count=1000,tactical_first=1000,broadcast_first=50,half='1st Half'),
        dict(video_first=1000,count=19000,tactical_first=20000,broadcast_first=1100,half='2nd Half')])
    return HTML.replace('__META__',json.dumps(meta)),meta


def test_html_clock_preserves_video_and_seek_controls_without_external_resources():
    source,meta=synthetic_html();html,got,starts=enhance_html(source)
    assert got==meta and starts==STARTS
    assert html.count('id="matchClockMain"')==1 and html.count('id="video"')==1
    assert 'full_movement_vectors.mp4' in html and 'type="range"' in html and '시계와 함께 전체화면' in html
    assert 'updateMatchClock(s.tactical_first+step,s.half)' in html
    assert 'https://' not in html and 'fetch(' not in html and '추정' in html
    with pytest.raises(ValueError):enhance_html(html)


def jxa(script):
    return json.loads(subprocess.run(['osascript','-l','JavaScript','-e',script],check=True,capture_output=True,text=True).stdout)


@pytest.mark.skipif(shutil.which('osascript') is None,reason='Local macOS JavaScript runtime unavailable')
def test_browser_clock_math_matches_python_in_local_javascript_runtime():
    cases=[]
    for half in STARTS:
        for elapsed in (0,24,25,67500,67501,67525,70000,12500):cases.append([STARTS[half]+elapsed,half])
    script=CLOCK_HELPERS_JS+'\nvar starts='+json.dumps(STARTS)+';var cases='+json.dumps(cases)+';JSON.stringify(cases.map(function(x){return matchClockInfo(x[0],x[1],starts,25);}));'
    result=jxa(script)
    assert result==[clock_info(frame,half,STARTS) for frame,half in cases]


@pytest.mark.skipif(shutil.which('osascript') is None,reason='Local macOS JavaScript runtime unavailable')
def test_full_player_javascript_clock_updates_on_seeks_and_half_reset():
    source,_=synthetic_html();html,_,_=enhance_html(source);script=re.search(r'<script>(.*?)</script>',html,re.S).group(1)
    mock=r'''
var nodes={};var document={fullscreenElement:null,getElementById:function(id){
 if(!nodes[id])nodes[id]={textContent:'',value:0,tagName:'DIV',paused:true,currentTime:0,
  pause:function(){this.paused=true;},play:function(){this.paused=false;return Promise.resolve();},addEventListener:function(){}};
 return nodes[id];},addEventListener:function(){}};
'''
    end=r'''
var start=nodes.matchClockMain.textContent;
seek(13500);var played=nodes.matchClockMain.textContent;var detail=nodes.matchClockDetail.textContent;
seek(1000);var reset=nodes.matchClockMain.textContent;
seek(0);JSON.stringify({start:start,played:played,detail:detail,reset:reset,returned:nodes.matchClockMain.textContent});
'''
    result=jxa(mock+script+end)
    assert result['start']=='전반 · 경기시계 00:00' and result['played']=='후반 · 경기시계 53:20'
    assert '후반 경과 08:20' in result['detail'] and result['reset']=='후반 · 경기시계 45:00' and result['returned']==result['start']
