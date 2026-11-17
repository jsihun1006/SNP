"""Merge corrected broadcast ball exports and audit their same-frame homography.

Run with the existing mtah2027 Python environment. Inputs are preserved under
data/derived/ball_tracking_homography_20261116/sources. No tracking is reselected.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import GAMES

PITCH = np.array(GAMES['BAR-ATH']['pitch'])
FPS = GAMES['BAR-ATH']['fps']
DEFAULT = ROOT / 'data/derived/ball_tracking_homography_20261116'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def transform(matrices, xy):
    ray = np.einsum('nij,nj->ni', matrices, np.column_stack([xy, np.ones(len(xy))]))
    good = np.isfinite(ray).all(axis=1) & (np.abs(ray[:, 2]) > 1e-12)
    result = np.full((len(xy), 2), np.nan)
    result[good] = ray[good, :2] / ray[good, 2, None]
    return result


def distribution(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return dict(n=len(v), median=float(np.median(v)) if len(v) else None,
                p95=float(np.percentile(v, 95)) if len(v) else None,
                max=float(v.max()) if len(v) else None)


def prepare(folder, hfile, video):
    first = pd.read_csv(folder/'sources/first_half_corrected_ball_coordinates.csv', low_memory=False)
    second = pd.read_csv(folder/'sources/second_half_ball_object_audited.csv', low_memory=False)
    assert np.array_equal(first.source_frame, np.arange(7171, 86638))
    assert np.array_equal(second.source_frame, np.arange(86823, 163279))
    assert np.allclose(first.video_seconds, first.source_frame/FPS)
    assert np.allclose(second.video_seconds, second.source_frame/FPS)
    cap = cv2.VideoCapture(str(video))
    meta = dict(frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps=cap.get(cv2.CAP_PROP_FPS),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    config = json.loads((folder/'sources/second_half_run_config.json').read_text())
    assert all(meta[k] == config['video'][k] for k in meta)
    assert video.stat().st_size == config['video']['bytes']
    # Full content hash verifies that these are the same source video, not just the same name.
    meta['sha256'] = sha(video)
    assert meta['sha256'] == config['video']['sha256']
    h = pd.read_csv(hfile)
    reference_h = json.loads((folder/'sources/first_half_homography_conversion_report.json').read_text())
    assert sha(hfile) == reference_h['h_sha256']
    assert np.array_equal(h.frame, np.arange(len(h)))
    matrices = h[[f'h{i}{j}' for i in range(3) for j in range(3)]].to_numpy().reshape(-1, 3, 3)
    valid = (h.valid.to_numpy() == 1) & np.isfinite(matrices).all(axis=(1, 2))
    inv = np.full_like(matrices, np.nan)
    for i in np.flatnonzero(valid):
        try:
            inv[i] = np.linalg.inv(matrices[i])
        except np.linalg.LinAlgError:
            valid[i] = False

    pieces = []
    for half, source in [(1, first), (2, second)]:
        d = pd.DataFrame(dict(source_frame=source.source_frame, half=half,
                              video_seconds=source.video_seconds, source_status=source.status))
        if half == 1:
            d['pixel_x'], d['pixel_y'] = source.pixel_x, source.pixel_y
            d['observation_kind'] = source.status.map({'selected_observation':'observed',
                'visually_reviewed_observation':'observed', 'review_candidate':'review_candidate', 'unresolved':'missing'})
            d['confidence'] = source.confidence
            d['needs_source_review'] = source.needs_review
            d['source_scene_id'] = np.nan
        else:
            d['pixel_x'], d['pixel_y'] = source.image_x, source.image_y
            d['observation_kind'] = source.status
            d['confidence'] = source.detector_confidence
            d['needs_source_review'] = source.verification.eq('unresolved') | source.object_audit.isin(
                ['suspected_hand','suspected_microphone','hand_model_disagreement'])
            d['source_scene_id'] = source.scene_id
        # Preserve every original field with its export prefix, including provisional points.
        d = pd.concat([d, source.add_prefix(f'h{half}_')], axis=1)
        pieces.append(d)
    d = pd.concat(pieces, ignore_index=True).sort_values('source_frame').reset_index(drop=True)
    assert not d.source_frame.duplicated().any()
    f = d.source_frame.to_numpy(int)
    xy = d[['pixel_x','pixel_y']].to_numpy(float)
    points = np.isfinite(xy).all(axis=1)
    calc = transform(inv[f], xy)
    projected = np.isfinite(calc).all(axis=1)
    d['h_frame'] = f
    d['homography_valid'] = valid[f]
    d['h_reprojection_error'] = h.reprojection_error.to_numpy()[f]
    d['h_quality_known'] = valid[f] & np.isfinite(d.h_reprojection_error) & (d.h_reprojection_error >= 0)
    d['recomputed_pitch_x_normalized'], d['recomputed_pitch_y_normalized'] = calc.T
    # Preserve the first-half coordinates; independently recompute them to expose discrepancies.
    canonical = calc.copy()
    m1 = d.half.to_numpy() == 1
    canonical[m1] = first[['pitch_x_normalized','pitch_y_normalized']].to_numpy(float)
    d['pitch_x_normalized'], d['pitch_y_normalized'] = canonical.T
    metres = canonical * PITCH
    d['pitch_x_m'], d['pitch_y_m'] = metres.T
    finite = np.isfinite(canonical).all(axis=1)
    assert np.array_equal(finite[m1], projected[m1])
    delta = np.linalg.norm((canonical[m1]-calc[m1])*PITCH, axis=1)
    assert np.nanmax(delta) < 1e-5
    assert np.allclose(first[['pitch_x_m','pitch_y_m']].to_numpy(), metres[m1], equal_nan=True, atol=1e-6)
    d['first_half_recalculation_difference_m'] = np.nan
    d.loc[m1, 'first_half_recalculation_difference_m'] = delta
    inside = finite & ((canonical >= 0) & (canonical <= 1)).all(axis=1)
    d['inside_pitch'] = pd.array(np.where(finite, inside.astype(float), np.nan), dtype='boolean')
    d['projection_status'] = np.select([~points, ~valid[f], ~projected, finite & ~inside],
        ['no_selected_image_point','no_valid_homography','projection_at_infinity','outside_pitch'], default='on_pitch')
    back = transform(matrices[f], calc)
    d['roundtrip_error_px'] = np.linalg.norm(back-xy, axis=1)
    d['image_point_out_of_frame'] = points & ((xy[:,0]<0)|(xy[:,0]>=meta['width'])|(xy[:,1]<0)|(xy[:,1]>=meta['height']))
    d['outside_pitch_distance_m'] = np.where(finite, np.linalg.norm(metres-np.clip(metres, [0,0], PITCH),axis=1), np.nan)
    # Adjacent-pair speed is diagnostic, never interpolated across a gap or half boundary.
    adjacent = np.r_[False, (np.diff(f)==1) & (np.diff(d.half)==0) & finite[1:] & finite[:-1]]
    scene_break = np.r_[False, (np.diff(d.source_scene_id.to_numpy(float)) != 0)] & ~m1
    adjacent &= ~scene_break
    speed = np.linalg.norm(np.diff(metres,axis=0,prepend=np.full((1,2),np.nan)),axis=1)*FPS
    d['adjacent_speed_mps'] = np.where(adjacent,speed,np.nan)
    # >45 m/s and H error >5 are review heuristics, not labels of incorrect tracking.
    d['flag_speed_gt45_mps'] = d.adjacent_speed_mps > 45
    d['flag_h_error_gt5'] = d.h_reprojection_error > 5
    d['flag_outside_gt2m'] = d.outside_pitch_distance_m > 2
    observed = d.observation_kind.eq('observed').to_numpy()
    d['review_recommended'] = points & (~valid[f] | ~d.h_quality_known | d.flag_h_error_gt5 |
        d.flag_outside_gt2m | d.flag_speed_gt45_mps | ~observed | d.needs_source_review)
    d['display_eligible'] = finite & inside & observed & d.h_quality_known & ~d.flag_h_error_gt5 & ~d.needs_source_review
    # Full broadcast axis retains the missing interval; no artificial join at halftime.
    timeline = d.set_index('source_frame').reindex(np.arange(f.min(), f.max()+1))
    absent = timeline.half.isna()
    timeline.loc[absent, 'source_status'] = 'not_in_source_exports'
    timeline.loc[absent, 'observation_kind'] = 'missing'
    timeline.loc[absent, 'projection_status'] = 'not_in_source_exports'
    timeline.loc[absent, 'video_seconds'] = timeline.index[absent]/FPS
    timeline.index.name = 'source_frame'
    assert absent.sum() == 185
    assert not np.isfinite(timeline.loc[absent,['pitch_x_m','pitch_y_m']].to_numpy(float)).any()
    return d, timeline.reset_index(), matrices, meta, len(h), first, second


def summarize(d):
    result = {}
    for half in [1,2]:
        part = d[d.half==half]
        point = np.isfinite(part[['pixel_x','pixel_y']]).all(axis=1)
        projected = np.isfinite(part[['pitch_x_m','pitch_y_m']]).all(axis=1)
        screened = (part.observation_kind.eq('observed') & part.inside_pitch.fillna(False)
                    & part.h_quality_known & (part.h_reprojection_error<=5))
        screened_pairs = screened & screened.shift(fill_value=False) & part.adjacent_speed_mps.notna()
        result[str(half)] = dict(frames=len(part), first_frame=int(part.source_frame.min()), last_frame=int(part.source_frame.max()),
            source_status_counts=part.source_status.value_counts().to_dict(),
            selected_image_points=int(point.sum()), homography_valid_frames=int(part.homography_valid.sum()),
            image_points_without_h=int((point & ~part.homography_valid).sum()),
            projected_points=int(projected.sum()), on_pitch=int(part.inside_pitch.fillna(False).sum()),
            outside_pitch=int((projected & ~part.inside_pitch.fillna(False)).sum()),
            outside_gt2m=int(part.flag_outside_gt2m.sum()),
            projected_observed=int((projected & part.observation_kind.eq('observed')).sum()),
            projected_interpolated=int((projected & part.observation_kind.eq('interpolated')).sum()),
            speed_gt45_pairs=int(part.flag_speed_gt45_mps.sum()),
            observed_only_speed_gt45_pairs=int((part.flag_speed_gt45_mps & part.observation_kind.eq('observed') &
                part.observation_kind.shift().eq('observed')).sum()),
            screened_observed_inside_pairs=int(screened_pairs.sum()),
            screened_observed_inside_speed_gt45_pairs=int((screened_pairs & part.flag_speed_gt45_mps).sum()),
            screened_observed_inside_speed_gt45_fraction=float((screened_pairs & part.flag_speed_gt45_mps).sum()/screened_pairs.sum()),
            h_error_gt5_on_selected_points=int((point & part.flag_h_error_gt5).sum()),
            roundtrip_px=distribution(part.roundtrip_error_px), adjacent_speed_mps=distribution(part.adjacent_speed_mps),
            h_error_valid=distribution(part.loc[part.homography_valid & part.h_quality_known,'h_reprojection_error']),
            first_half_existing_coordinate_difference_m=distribution(part.first_half_recalculation_difference_m))
    return result


def pitch_lines():
    lines = [np.array([[0,0],[1,0],[1,1],[0,1],[0,0]]), np.array([[.5,0],[.5,1]])]
    for depth,width in [(16.5,40.32),(5.5,18.32)]:
        for x0,x1 in [(0,depth/PITCH[0]),(1,1-depth/PITCH[0])]:
            y0,y1=(1-width/PITCH[1])/2,(1+width/PITCH[1])/2
            lines.append(np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]]))
    t=np.linspace(0,2*np.pi,181)
    lines.append(np.column_stack([.5+9.15/PITCH[0]*np.cos(t),.5+9.15/PITCH[1]*np.sin(t)]))
    return lines


def draw_sample(image, row, matrix, scale=.5):
    image = cv2.resize(image,None,fx=scale,fy=scale)
    for line in pitch_lines():
        ray=np.column_stack([line,np.ones(len(line))])@matrix.T
        for a,b in zip(ray[:-1],ray[1:]):
            if not np.isfinite([a,b]).all() or a[2]*b[2]<=0: continue
            xy=np.clip(np.stack([a[:2]/a[2],b[:2]/b[2]])*scale,-100000,100000).round().astype(int)
            ok,p1,p2=cv2.clipLine((0,0,image.shape[1],image.shape[0]),tuple(xy[0]),tuple(xy[1]))
            if ok: cv2.line(image,p1,p2,(255,220,0),1,cv2.LINE_AA)
    if np.isfinite([row.pixel_x,row.pixel_y]).all():
        p=tuple(np.rint(np.array([row.pixel_x,row.pixel_y])*scale).astype(int))
        color=(0,165,255) if row.review_recommended else (70,255,70)
        cv2.circle(image,p,9,color,2,cv2.LINE_AA)
        cv2.drawMarker(image,p,color,cv2.MARKER_CROSS,18,1)
    # Pitch inset uses the same fixed axes for both halves.
    x0,y0,w,h=720,370,220,142
    cv2.rectangle(image,(x0,y0),(x0+w,y0+h),(25,65,25),-1)
    cv2.rectangle(image,(x0,y0),(x0+w,y0+h),(230,230,230),1)
    cv2.line(image,(x0+w//2,y0),(x0+w//2,y0+h),(230,230,230),1)
    if np.isfinite([row.pitch_x_normalized,row.pitch_y_normalized]).all():
        p=(int(x0+row.pitch_x_normalized*w),int(y0+row.pitch_y_normalized*h))
        if x0<=p[0]<=x0+w and y0<=p[1]<=y0+h: cv2.circle(image,p,4,(30,255,255),-1)
    cv2.rectangle(image,(0,0),(image.shape[1],66),(20,20,20),-1)
    label=f'H{int(row.half)} f={int(row.source_frame)} {row.observation_kind} | H error={row.h_reprojection_error:.2f}'
    cv2.putText(image,label,(12,24),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),1,cv2.LINE_AA)
    label=f'pitch=({row.pitch_x_m:.2f}, {row.pitch_y_m:.2f})m | {row.projection_status} | v={row.adjacent_speed_mps:.1f}m/s'
    cv2.putText(image,label,(12,49),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    return image


def visuals(folder, d, matrices, video):
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/mtah-ball-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(16,12),layout='constrained')
    for col,half in enumerate([1,2]):
        part=d[d.half==half]
        ax=axes[0,col]
        for line in pitch_lines(): ax.plot(*(line*PITCH).T,color='#c5d3c3',lw=1)
        for kind,color in [('observed','#168e9b'),('interpolated','#e98a2b')]:
            q=part[part.observation_kind.eq(kind)&part.inside_pitch.fillna(False)]
            ax.scatter(q.pitch_x_m,q.pitch_y_m,s=1,alpha=.12,color=color,label=f'{kind} n={len(q):,}')
        ax.set(xlim=(-3,108.3),ylim=(71,-3),aspect='equal',title=f'Half {half}: ground-plane projections (not GT)',xlabel='x (m)',ylabel='y (m)')
        ax.legend(markerscale=6)
        x=part.video_seconds/60
        ax=axes[1,col]
        for mask,y,color,label in [(part.homography_valid,3,'#168e9b','H available'),(part.observation_kind.eq('observed'),2,'#305ac4','observed image point'),(part.observation_kind.eq('interpolated'),1,'#e98a2b','interpolated image point'),(part.flag_outside_gt2m,0,'#cd4c46','outside >2m')]:
            ax.scatter(x[mask],np.full(int(mask.sum()),y),marker='|',s=6,color=color,label=label)
        ax.set(yticks=[0,1,2,3],yticklabels=['outside >2m','interpolated','observed','valid H'],xlabel='Original broadcast video (minutes)',title='Coverage and review flags, full frame rate')
        ax=axes[2,col]
        ax.plot(x,part.adjacent_speed_mps.clip(upper=150),lw=.35,color='#168e9b')
        ax.axhline(45,c='#cd4c46',lw=1,label='review heuristic: 45 m/s')
        ax.set(ylim=(0,155),xlabel='Original broadcast video (minutes)',ylabel='Adjacent speed (m/s; display capped at 150)',title='No speed across half, gap or known scene cut')
        ax.legend()
    fig.savefig(folder/'homography_diagnostics.png',dpi=160)
    plt.close(fig)
    # Deterministic strata: three timeline positions and three diagnostic cases per half.
    selected=[]
    for half in [1,2]:
        part=d[d.half==half]
        good=part[part.display_eligible & (part.h_reprojection_error<2)]
        for q in [.15,.5,.85]:
            target=part.source_frame.quantile(q)
            selected.append((f'h{half}_timeline_{q}',good.iloc[np.argmin(abs(good.source_frame-target))]))
        bad=part[part.flag_outside_gt2m & part.homography_valid & part.observation_kind.eq('observed')]
        if len(bad): selected.append((f'h{half}_outside',bad.loc[bad.outside_pitch_distance_m.idxmax()]))
        bad=part[part.flag_speed_gt45_mps & part.h_quality_known & (part.h_reprojection_error<2)]
        if len(bad): selected.append((f'h{half}_jump',bad.loc[bad.adjacent_speed_mps.idxmax()]))
        bad=part[part.flag_h_error_gt5 & part.observation_kind.eq('observed')]
        if len(bad): selected.append((f'h{half}_high_h_error',bad.loc[bad.h_reprojection_error.idxmax()]))
    samples=folder/'visual_samples'; samples.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(str(video))
    thumbnails=[];records=[]
    for tag,row in selected:
        f=int(row.source_frame);cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read()
        assert ok and int(cap.get(cv2.CAP_PROP_POS_FRAMES))==f+1
        overlay=draw_sample(im,row,matrices[f])
        cv2.imwrite(str(samples/f'{tag}_f{f}.jpg'),overlay)
        thumbnails.append(cv2.resize(overlay,(640,360)))
        records.append(dict(tag=tag,source_frame=f,half=int(row.half),h_error=float(row.h_reprojection_error),
            pitch_x_m=float(row.pitch_x_m),pitch_y_m=float(row.pitch_y_m),speed_mps=float(row.adjacent_speed_mps),
            image=f'visual_samples/{tag}_f{f}.jpg'))
        if not np.isfinite(records[-1]['speed_mps']): records[-1]['speed_mps']=None
    cap.release()
    sheet=np.zeros((int(np.ceil(len(thumbnails)/2))*360,1280,3),np.uint8)
    for i,im in enumerate(thumbnails):sheet[(i//2)*360:(i//2+1)*360,(i%2)*640:(i%2+1)*640]=im
    cv2.imwrite(str(folder/'homography_contact_sheet.jpg'),sheet)
    (folder/'visual_samples.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    # Include all twelve strata, including low-error H on a clearly incompatible close-up.
    clips=selected
    writer=cv2.VideoWriter(str(folder/'homography_review_clips.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),FPS,(960,540))
    assert writer.isOpened()
    cap=cv2.VideoCapture(str(video));lookup=d.set_index('source_frame')
    manifest=[]
    for tag,center in clips:
        start=max(int(center.source_frame)-50,int(d[d.half==center.half].source_frame.min()))
        stop=min(start+100,int(d[d.half==center.half].source_frame.max())+1)
        cap.set(cv2.CAP_PROP_POS_FRAMES,start)
        for f in range(start,stop):
            ok,im=cap.read();assert ok
            row=lookup.loc[f].copy();row['source_frame']=f
            overlay=draw_sample(im,row,matrices[f])
            cv2.putText(overlay,tag,(12,530),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
            writer.write(overlay)
        manifest.append(dict(case=tag,start_frame=start,end_frame_exclusive=stop,center_frame=int(center.source_frame)))
    cap.release();writer.release()
    (folder/'review_clips_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=DEFAULT)
    p.add_argument('--homography',type=Path,default=ROOT/'data/broadcast/BAR-ATH_broadcast_homography.csv')
    p.add_argument('--video',type=Path,default=ROOT/'data/broadcast/BAR-ATH_broadcast-001.mp4')
    p.add_argument('--skip-visuals',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    d,timeline,matrices,meta,nh,first,second=prepare(a.out,a.homography,a.video)
    d.to_csv(a.out/'combined_corrected_ball_pitch.csv',index=False)
    timeline.to_csv(a.out/'combined_broadcast_timeline.csv.gz',index=False,compression='gzip')
    d[d.half==2].to_csv(a.out/'second_half_ball_pitch.csv',index=False)
    columns=['source_frame','half','video_seconds','source_status','observation_kind','pixel_x','pixel_y',
        'pitch_x_m','pitch_y_m','projection_status','h_reprojection_error','outside_pitch_distance_m',
        'adjacent_speed_mps','flag_outside_gt2m','flag_speed_gt45_mps','flag_h_error_gt5','needs_source_review']
    d.loc[d.review_recommended,columns].to_csv(a.out/'review_frames.csv',index=False)
    summary=dict(halves=summarize(d),combined_rows=len(d),full_timeline_rows=len(timeline),
        missing_interval=dict(first_frame=86638,last_frame=86822,frames=185,seconds=185/FPS),
        original_video=meta,homography_rows=nh,unused_homography_tail_rows=nh-meta['frames'],
        homography_sha256=sha(a.homography),coordinate_system=dict(normalized_range=[0,1],pitch_size_m=PITCH.tolist(),
        transform='inverse H[source_frame] @ [pixel_x,pixel_y,1], homogeneous divide',offset_frames=0,
        first_half='preserved source values and independently recomputed',half_flip=False,
        pixel_point='selected original 1920x1080 image point; second-half interpolated points remain labelled'),
        source_commits={'first_half':'843c33078559fe095b96e5fd7ecd5c5a15c2a29e','second_half':'39f6e1ddc8f7a2081f058a48f3adc384666bdeea'},
        source_sha256={p.name:sha(p) for p in sorted((a.out/'sources').glob('*'))},
        thresholds=dict(outside_distance_m=2,adjacent_speed_mps=45,h_reprojection_error=5),
        accuracy_validated=False)
    (a.out/'audit_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if not a.skip_visuals:visuals(a.out,d,matrices,a.video)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
