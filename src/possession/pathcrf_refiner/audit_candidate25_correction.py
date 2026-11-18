"""Audit saved epoch-29 hybrid before/after corrections without refitting."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json
import numpy as np
import pandas as pd
import audit_errors as audit
from inputs import ROOT
from transportmer_ha_data import contiguous_runs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'artifacts/possession/pathcrf/candidate25_dfl/historical_run/correction_epoch29')
    parser.add_argument('--source',type=Path,default=ROOT/'artifacts/possession/pathcrf/candidate25_dfl/historical_run')
    args=parser.parse_args();out=args.run/'error_audit';out.mkdir(parents=True,exist_ok=True)
    audit.MODELS={'before_epoch29':'base','after_correction':'refined'}
    worker=audit.Audit(SimpleNamespace(run=args.source,cache=ROOT/'data/external/dfl',
                     events=Path('/private/tmp/pathcrf-official-dfl/data/sportec_synced/J03WR9.parquet')))
    frame_changes=[];exit_counts={}
    for half in worker.halves:
        d=pd.read_parquet(args.run/f'J03WR9_H{half["half"]}.parquet')
        np.testing.assert_array_equal(d.frame,half['frame']);np.testing.assert_array_equal(d.truth,half['truth'])
        half['base']=d.epoch29_saved.to_numpy();half['refined']=d.corrected_selected.to_numpy()
        half['candidate']=d.candidate_corrected.to_numpy(bool)
        np.testing.assert_array_equal(half['base']==2,half['refined']==2)
        for kind,mask in [('fixed',(half['truth']>=0)&(half['base']!=half['truth'])&(half['refined']==half['truth'])),
                          ('damaged',(half['truth']>=0)&(half['base']==half['truth'])&(half['refined']!=half['truth']))]:
            for a,b in contiguous_runs(mask,half['frame']):
                frame_changes.append({'kind':kind,**worker.location(half,a,b,half['refined'])})
    for model in audit.MODELS:
        for half in worker.halves:
            worker.audit_transitions(half,model);worker.audit_spells(half,model);worker.audit_frames(half,model)
    data=pd.DataFrame(worker.rows)
    data.to_csv(out/'all_errors.csv',index=False,encoding='utf-8-sig')
    counts=data.groupby(['category','subtype','model']).size().unstack('model',fill_value=0).reset_index()
    counts.to_csv(out/'error_subtype_comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(frame_changes).to_csv(out/'fixed_damaged_intervals.csv',index=False,encoding='utf-8-sig')
    transitions=pd.DataFrame(worker.transitions)
    cross=transitions.pivot(index=['match','half','truth_frame','truth_half_clock','direction'],columns='model',values='matched').reset_index()
    cross['comparison']=np.select([cross.before_epoch29&~cross.after_correction,~cross.before_epoch29&cross.after_correction,
                                   cross.before_epoch29&cross.after_correction],['lost','recovered','both_matched'],default='both_missed')
    cross.to_csv(out/'transition_changes.csv',index=False,encoding='utf-8-sig')
    for model in audit.MODELS:
        folder=out/model;folder.mkdir(exist_ok=True)
        lines=[f'# {model} 전체 오류 목록','','시간은 각 하프 시작 기준이다. 전환 방향 일치 ±1초, 점유 IoU 0.5 기준이며 종류끼리는 중복될 수 있다.','']
        for category,title in audit.CATEGORIES.items():
            rows=data[data.model.eq(model)&data.category.eq(category)].sort_values(['half','start_frame'])
            rows.to_csv(folder/f'{category}.csv',index=False,encoding='utf-8-sig')
            lines += [f'## {title} {len(rows)}개','','| 하프 | 시간 | 원본 프레임 | 정답 / 예측 | 유형 | 상황 |',
                      '|---:|---|---|---|---|---|']
            for r in rows.itertuples():
                subtype=(str(r.direction)+' · ' if pd.notna(r.direction) else '')+r.subtype
                lines.append(f'| {r.half} | {r.half_start_clock}~{r.half_end_clock} | {r.start_frame}~{r.end_frame_inclusive} | {r.truth_sequence} / {r.prediction_sequence} | {subtype} | {r.context_tags.replace(" | ",", ")} |')
            lines.append('')
        (out/f'{model}_errors.md').write_text('\n'.join(lines))
    summary={'transition_changes':cross.comparison.value_counts().to_dict(),
             'frame_changes':pd.DataFrame(frame_changes).groupby('kind').frame_count.sum().to_dict()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(counts.to_string(index=False));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
