"""Enumerate saved baseline/corrected errors without fitting or selecting settings."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd

import audit_errors as audit
from inputs import ROOT

RUN = ROOT / 'artifacts/possession/pathcrf/correct_5fps_v2'
SOURCE = ROOT / 'artifacts/possession/pathcrf/dfl_5train_1val_1test'
OUTPUT = RUN / 'error_audit'


class CorrectionAudit(audit.Audit):
    def location(self, *args, **kwargs):
        row = super().location(*args, **kwargs)
        row['context_tags'] = (row['context_tags'].replace('25fps 대상 안', '보정 대상 인플레이')
                               .replace('25fps 대상 밖', '고정 D 구간').replace('25fps 대상 경계', '인플레이/D 경계'))
        return row


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit.MODELS = {'5fps': 'base', '5fps_corrected': 'refined'}
    args = SimpleNamespace(run=SOURCE, cache=ROOT / 'data/external/dfl',
                           events=Path('/private/tmp/pathcrf-official-dfl/data/sportec_synced/J03WR9.parquet'))
    worker = CorrectionAudit(args)
    for half in worker.halves:
        table = pd.read_parquet(RUN / f'J03WR9_H{half["half"]}.parquet')
        np.testing.assert_array_equal(table.frame, half['frame'])
        np.testing.assert_array_equal(table.truth, half['truth'])
        np.testing.assert_array_equal(table.base, half['base'])
        half['refined'] = table.validation_selected.to_numpy()
        half['candidate'] = half['base'] != 2
        np.testing.assert_array_equal(half['refined'] == 2, half['base'] == 2)
    for model in audit.MODELS:
        for half in worker.halves:
            worker.audit_transitions(half, model)
            worker.audit_spells(half, model)
            worker.audit_frames(half, model)
    data = pd.DataFrame(worker.rows)
    data.to_csv(OUTPUT / 'all_errors.csv', index=False, encoding='utf-8-sig')
    counts = data.groupby(['category','subtype','model']).size().unstack('model',fill_value=0).reset_index()
    counts.to_csv(OUTPUT / 'error_subtype_comparison.csv',index=False,encoding='utf-8-sig')
    for model in audit.MODELS:
        folder = OUTPUT / model
        folder.mkdir(exist_ok=True)
        for category in audit.CATEGORIES:
            data.loc[data.model.eq(model) & data.category.eq(category)].to_csv(folder / f'{category}.csv',
                                                                              index=False,encoding='utf-8-sig')
    transitions = pd.DataFrame(worker.transitions)
    cross = transitions.pivot(index=['match','half','truth_frame','truth_half_clock','direction'],
                              columns='model',values='matched').reset_index()
    cross['comparison'] = np.select([cross['5fps'] & ~cross['5fps_corrected'],
                                    ~cross['5fps'] & cross['5fps_corrected'],
                                    cross['5fps'] & cross['5fps_corrected']],
                                   ['lost','recovered','both_matched'],default='both_missed')
    cross.to_csv(OUTPUT / 'transition_changes.csv',index=False,encoding='utf-8-sig')
    wrong = data.loc[data.category.eq('wrong_run')]
    frames = wrong.groupby('model').frame_count.sum()
    metrics = pd.read_csv(RUN / 'holdout_comparison.csv').set_index('condition')
    for model,key in [('5fps','base'),('5fps_corrected','validation_selected')]:
        for category,column in [('missed_transition','missed_transitions'),('spurious_transition','spurious_transitions'),
                                ('missed_spell','missed_spells'),('short_missed_spell','short_missed')]:
            assert int((data.model.eq(model)&data.category.eq(category)).sum()) == int(metrics.loc[key,column])
        assert frames[model] == metrics.loc[key,'wrong_frames']
    changes = cross.comparison.value_counts().to_dict()
    (OUTPUT / 'summary.json').write_text(json.dumps({'transition_changes':changes},indent=2))
    print(counts.to_string(index=False))
    print('Transition changes:',changes)


if __name__=='__main__': main()
