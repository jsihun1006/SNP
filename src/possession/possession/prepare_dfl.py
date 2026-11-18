"""Local DFL XML/Parquet -> ready-to-run possession NPZ. No model inference."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

from locations import ARTIFACT, REPO
MATCHES=('J03WMX','J03WN1','J03WOH','J03WOY','J03WPY','J03WQQ','J03WR9')


def local_raw(raw_dir,prefix,match):
    files=sorted(Path(raw_dir).rglob(f'{prefix}*{match}.xml*'))
    if not files:raise FileNotFoundError(f'Local DFL file missing: {prefix} / {match} in {raw_dir}')
    return next((p for p in files if p.suffix=='.xml'),files[0])


def load_dfl(match,data_dir,raw_dir):
    tracking_path=data_dir/f'{match}_tracking.parquet';meta_path=data_dir/f'{match}_meta.parquet'
    if tracking_path.is_file() and meta_path.is_file():
        return pd.read_parquet(tracking_path),pd.read_parquet(meta_path)
    from kloppy import sportec
    meta_file=local_raw(raw_dir,'DFL_02_01_matchinformation',match)
    tracking_file=local_raw(raw_dir,'DFL_04_03_positions_raw_observed',match)
    dataset=sportec.load_tracking(raw_data=str(tracking_file),meta_data=str(meta_file),only_alive=False)
    frame=dataset.to_df(engine='pandas');home,away=dataset.metadata.teams
    pitch=dataset.metadata.pitch_dimensions
    meta=pd.DataFrame([{'player_id':p.player_id,'team':side,'jersey':p.jersey_no,
                       'position':str(p.starting_position) if p.starting_position else '',
                       'pitch_length':pitch.pitch_length,'pitch_width':pitch.pitch_width}
                      for side,t in [('home',home),('away',away)] for p in t.players])
    columns=['frame_id','period_id','timestamp','ball_state','ball_x','ball_y']
    columns += [c for c in frame if c.endswith(('_x','_y')) and not c.startswith('ball_')]
    tracking=frame[columns].rename(columns={'period_id':'period'}).copy()
    tracking['timestamp_s']=tracking.pop('timestamp').dt.total_seconds()
    tracking['ball_state']=tracking.ball_state.astype(str).str.lower().eq('alive').astype(np.int8)
    data_dir.mkdir(parents=True,exist_ok=True)
    tracking.to_parquet(tracking_path,index=False);meta.to_parquet(meta_path,index=False)
    return tracking,meta


def prepare_match(match,data_dir,raw_dir,native_dir,base_path,output,dead_csv=None):
    tracking,meta=load_dfl(match,Path(data_dir),Path(raw_dir))
    source=Path(native_dir)/f'{match}.parquet' if native_dir else None
    if source and source.is_file():
        # Reuse the original 105x68m DFL preprocessing, with its home_/away_ track identities.
        native=pd.read_parquet(source).set_index('frame_id',verify_integrity=True)
        names=sorted(c[:-2] for c in native if c.startswith(('home_','away_')) and c.endswith('_x'))
    else:
        names=[];ids=[]
        for row in meta.dropna(subset=['jersey']).itertuples(index=False):
            if str(row.player_id)+'_x' in tracking:
                names.append(f'{row.team}_{int(row.jersey)}');ids.append(str(row.player_id))
        order=np.argsort(names);names=[names[i] for i in order];ids=[ids[i] for i in order]
        native=pd.DataFrame(index=tracking.frame_id.to_numpy())
        for name,pid in zip(names,ids,strict=True):
            native[name+'_x']=tracking[pid+'_x'].to_numpy()*105
            native[name+'_y']=tracking[pid+'_y'].to_numpy()*68
        native['ball_x']=tracking.ball_x.to_numpy()*105;native['ball_y']=tracking.ball_y.to_numpy()*68
    team=np.array([0 if name.startswith('home_') else 1 for name in names],np.int8)
    roles={}
    for row in meta.dropna(subset=['jersey']).itertuples(index=False):
        role=str(row.position).strip().lower()
        if role not in ('','none','nan','unknown'):
            roles[f'{row.team}_{int(row.jersey)}']=int(role in ('gk','goalkeeper'))
    keeper=np.array([roles.get(name,-1) for name in names],np.int8)
    tracking=tracking.loc[tracking.period.isin([1,2])].sort_values(['period','frame_id']).reset_index(drop=True)
    axis=tracking.frame_id.to_numpy(np.int64);frame=np.arange(len(axis));aligned=native.reindex(axis)
    xy=aligned[[name+'_'+coord for name in names for coord in ('x','y')]].to_numpy(np.float32)
    players=xy.reshape(len(axis),len(names),2)
    ball=aligned[['ball_x','ball_y']].to_numpy(np.float32)
    fallback=None
    if base_path and Path(base_path).is_file():
        dead=np.zeros(len(axis),bool);fallback=np.full(len(axis),2,np.int8)
        with np.load(base_path,allow_pickle=False) as base:
            for half in (1,2):
                rows=tracking.period.eq(half).to_numpy()
                if not rows.any():continue
                if not np.array_equal(base[f'h{half}_frame'],axis[rows]):raise ValueError('DFL/base-cache frame mismatch')
                prediction=base[f'h{half}_base'];dead[rows]=prediction==2;fallback[rows]=prediction
    elif dead_csv:
        table=pd.read_csv(dead_csv)
        if not np.array_equal(table.frame,frame):raise ValueError('Dead-mask CSV axis mismatch')
        dead=table.dead.to_numpy(bool) if 'dead' in table else table.label.eq('D').to_numpy()
    else:
        raise ValueError('Provide the existing model D mask with --base-dir or --dead; DFL truth is not used as an input mask.')
    result=dict(players=players,ball=ball,team=team,player_names=np.asarray(names),goalkeeper=keeper,
                frame=frame,source_frame=axis,half=tracking.period.to_numpy(np.int8),
                seconds=tracking.timestamp_s.to_numpy(np.float64),dead=dead,pitch=np.array([105.,68.]))
    if fallback is not None:result['fallback_ha']=fallback
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**result)
    print(f'DFL preprocessing saved: {output}')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('match',nargs='?',default='J03WR9',choices=(*MATCHES,'all'))
    parser.add_argument('--data-dir',type=Path,default=REPO/'data/external/dfl')
    parser.add_argument('--raw-dir',type=Path,default=REPO/'data/external/dfl/raw')
    parser.add_argument('--native-dir',type=Path,default=REPO/'data/transition_final/source')
    parser.add_argument('--base-dir',type=Path,default=REPO/'data/transition_final/frozen_7a9ebd31eba97ba5')
    parser.add_argument('--dead',type=Path)
    parser.add_argument('--output-dir',type=Path,default=ARTIFACT/'inputs')
    args=parser.parse_args()
    for match in MATCHES if args.match=='all' else (args.match,):
        local=ARTIFACT/'inputs'/f'{match}_base5.npz'
        base=local if local.is_file() else args.base_dir/f'{match}_pred.npz'
        prepare_match(match,args.data_dir,args.raw_dir,args.native_dir,base,args.output_dir/f'{match}.npz',args.dead)


if __name__=='__main__':main()
