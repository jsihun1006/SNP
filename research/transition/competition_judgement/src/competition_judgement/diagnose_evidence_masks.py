"""Independent paired-mask rerun in a fresh folder; source scores stay frozen."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .evidence_source import EvidenceSource,CLUSTER_COLUMNS,MATCH_COLUMNS
from .evidence_diagnostics import masking_analysis


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path.cwd();run=Path(a.run_dir).resolve();out=Path(a.output).resolve()
    if not out.is_relative_to(root/'data/derived/competition_judgement'):raise ValueError('Local output required')
    if out.exists():raise FileExistsError(out)
    config=json.loads((run/'preregistered_config.json').read_text())
    if config!=json.loads((root/'competition_judgement/configs/evidence_stage_b.json').read_text()):raise ValueError('Configuration changed')
    base=root/'data/derived/competition_judgement'
    frames=pd.read_csv(base/'local_v2_1_20261109/local_frame_features.csv',usecols=['frame','period','n_home','n_away','strict_quality_valid'])
    clusters=pd.read_csv(base/'local_v2_1_20261109/local_clusters.csv',usecols=CLUSTER_COLUMNS)
    matching=pd.read_csv(base/'temporal_v2_2_final_20261109/cluster_temporal_features.csv',usecols=MATCH_COLUMNS)
    manifest=json.loads((root/'data/derived/relational_temporal_v1_20261109/split_manifest.json').read_text())
    source=EvidenceSource(root,frames,manifest,clusters,matching,config);out.mkdir(parents=True)
    result=masking_analysis(source,config,out,base/'missingness_stage_a_final_20261109')
    result['comparison_revision']='Match P4 term mask in addition to composite source masks; scores and policies unchanged.'
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
