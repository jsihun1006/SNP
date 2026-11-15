"""Check audio timing against source samples at start, middle, and end."""
import json, subprocess
from pathlib import Path
import imageio_ffmpeg
import numpy as np

BASE=Path(__file__).resolve().parent.parent
OUT=BASE/'outputs/full-correction'
SOURCE=Path(r'H:\data\broadcast\BAR-ATH_broadcast.mp4')
VIDEO=OUT/'before_after_first_half_full.mp4'
FFMPEG=imageio_ffmpeg.get_ffmpeg_exe()
RATE=8000

def samples(path,start):
    result=subprocess.run([FFMPEG,'-hide_banner','-loglevel','error','-ss',str(start),'-i',str(path),'-t','3','-map','0:a:0','-vn','-ac','1','-ar',str(RATE),'-f','f32le','pipe:1'],capture_output=True,check=True)
    data=np.frombuffer(result.stdout,dtype='<f4').astype(np.float64)
    assert len(data)>=2.9*RATE, (path,start,len(data),result.stderr)
    return data-data.mean()

results=[]
for offset in (2.0,1589.0,3173.0):
    a=samples(SOURCE,7171/25+offset)
    b=samples(VIDEO,offset)
    n=min(len(a),len(b));a=a[:n];b=b[:n]
    assert np.linalg.norm(a)>0 and np.linalg.norm(b)>0
    size=1<<((2*n-1).bit_length())
    correlation=np.fft.irfft(np.fft.rfft(a,size)*np.conj(np.fft.rfft(b,size)),size)
    lags=np.arange(-int(.25*RATE),int(.25*RATE)+1)
    lag=int(lags[np.argmax(correlation[lags%size])])
    aa,bb=(a[lag:],b[:n-lag]) if lag>=0 else (a[:n+lag],b[-lag:])
    coefficient=float(np.dot(aa,bb)/(np.linalg.norm(aa)*np.linalg.norm(bb)))
    result=dict(output_start_seconds=offset,source_start_seconds=7171/25+offset,sample_rate=RATE,compared_samples=len(aa),lag_seconds=lag/RATE,normalized_correlation=coefficient)
    assert abs(lag/RATE)<=.1 and coefficient>=.8,result
    results.append(result)

report=dict(status='passed',method='Three 3-second mono audio windows matched to source; correlation over +/-250ms; <=100ms lag and >=0.8 normalized correlation required.',source_offset_seconds=7171/25,checks=results)
(OUT/'video-audio-verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
