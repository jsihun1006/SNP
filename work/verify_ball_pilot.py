import csv,json,subprocess,hashlib
from pathlib import Path
import numpy as np,imageio_ffmpeg
ROOT=Path(__file__).resolve().parent;WORK=ROOT/'ball_pilot';OUT=ROOT.parent/'outputs/ball-correction-3min'
doc=json.loads((WORK/'candidates_enriched.json').read_text());track=json.loads((WORK/'final_track.json').read_text())['records'];report=json.loads((OUT/'report.json').read_text())
accepted={'selected_observation','visually_reviewed_observation'}
rows=list(csv.DictReader((OUT/'corrected_ball_coordinates.csv').open(encoding='utf-8-sig')))
assert len(rows)==len(track)==4500
for i,(csvrow,row) in enumerate(zip(rows,track)):
    assert int(csvrow['source_frame'])==row['source_frame']==doc['start']+i
    assert int(csvrow['local_frame'])==i
    if row['status'] in accepted:
        assert row['candidate_index']>=0
        candidate=doc['records'][i]['candidates'][row['candidate_index']]
        assert np.allclose([row['pixel_x'],row['pixel_y']],candidate['point'],rtol=0,atol=1e-10)
        assert csvrow['pixel_x'] and csvrow['pixel_y']
    else:assert csvrow['pixel_x']==csvrow['pixel_y']==''
for frame in (19294,19419,19544):
    r=track[frame-doc['start']]
    assert r['status']=='review_candidate' and r['pixel_x'] is None
assert sum(r['status'] in accepted for r in track)==3130
assert sum(r['status'] not in accepted for r in track)==1370
assert sum(c.get('origin')=='targeted_redetection' for r in doc['records'] for c in r['candidates'])==739
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
master=OUT/'ball_correction_before_after_3min.mp4';preview=OUT/'ball_correction_preview_3min.mp4'
subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-i',str(master),'-vf','scale=1280:480','-c:v','h264_nvenc','-preset','p5','-b:v','3M','-maxrate','4M','-bufsize','8M','-c:a','copy','-movflags','+faststart',str(preview)],check=True)
result=dict(passed=True,rows=4500,frame_range=[doc['start'],doc['end']],observed_coordinate_source_matches=True,unresolved_coordinates_blank=True,
            known_advertising_conflicts_not_accepted=True,csv_sha256=hashlib.sha256((OUT/'corrected_ball_coordinates.csv').read_bytes()).hexdigest(),
            preview_bytes=preview.stat().st_size)
(OUT/'verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
