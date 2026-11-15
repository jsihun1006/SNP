"""Archive immutable correction results, verifying every copied file byte-for-byte."""
import hashlib,json,shutil
from pathlib import Path

BASE=Path(__file__).resolve().parent.parent
RUN=Path(r'H:\data\ball-tracking-results\first_half\correction_recall_v1')
OUT=BASE/'outputs/full-correction'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def main():
    summary=json.loads((OUT/'completion_report.json').read_text(encoding='utf-8'))
    records=[]
    def include(source,destination,expected=None):
        before=sha(source)
        if expected is not None:assert before==expected,(source,before,expected)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists() or sha(destination)!=before:shutil.copy2(source,destination)
        assert sha(destination)==before,(source,destination)
        records.append(dict(source_path=str(source),repository_path=destination.relative_to(BASE).as_posix(),bytes=destination.stat().st_size,sha256=before))

    include(RUN/'qa_adjusted/frame_decisions.jsonl.gz',OUT/'frame_decisions.jsonl.gz',summary['qa_decisions_sha256'])
    include(RUN/'qa_adjusted/corrected_ball_coordinates.csv',OUT/'corrected_ball_coordinates.csv',summary['csv_sha256'])
    include(RUN/'qa_adjusted/job_complete.json',OUT/'completion_report.json')
    archive=OUT/'run_archive'
    for name in ('run_config.json','job_complete.json','corrected_ball_coordinates.csv','frame_decisions.jsonl.gz'):
        expected=summary['automatic_phase']['decisions_sha256'] if name=='frame_decisions.jsonl.gz' else summary['automatic_phase']['csv_sha256'] if name=='corrected_ball_coordinates.csv' else None
        include(RUN/name,archive/name,expected)
    chunks=sorted((RUN/'chunks').iterdir(),key=lambda x:int(x.name))
    assert len(chunks)==80
    for chunk in chunks:
        assert chunk.is_dir()
        assert {p.name for p in chunk.iterdir()}=={'decisions.jsonl.gz','manifest.json'}
        for name in ('manifest.json','decisions.jsonl.gz'):
            include(chunk/name,archive/'chunks'/chunk.name/name)
    # Check that each exact visual adjudication still refers to archived evidence.
    actions=json.loads((OUT/'visual_adjudications.json').read_text(encoding='utf-8'))['decisions']
    assert len(actions)==45
    by_source={r['source_path'].replace('\\','/'):r for r in records}
    for action in actions:
        assert by_source[action['source_chunk_decisions'].replace('\\','/')]['sha256']==action['source_chunk_sha256']
    code_names=('run_full_ball_correction.py','full_ball_tracking.py','full_ball_recovery.py','finalize_full_ball_correction.py','full-run-adjudications.json','render_full_correction_examples.py','render_full_comparison_video.py','verify_comparison_audio.py','package_full_correction.py')
    code=[dict(repository_path='work/'+name,sha256=sha(BASE/'work'/name)) for name in code_names]
    assert sha(BASE/'work/finalize_full_ball_correction.py')==summary['code_sha256']
    report=dict(status='verified',source_frame_range=summary['source_frame_range'],frames=summary['frames'],archived_chunk_count=len(chunks),source_to_repository_mapping=records,core_code=code,total_archived_data_bytes=sum(x['bytes'] for x in records),transient_files_excluded=['run.lock','progress.json','recovery_progress.json'],external_inputs=['Original source video','Detector model weights','Original detection input JSONL','Homography and ATD source datasets'],notes='Final results are at outputs/full-correction; run_archive preserves the automatic phase and all completed chunks. Absolute paths in frozen metadata identify the original run; source_to_repository_mapping locates identical archived bytes.')
    (OUT/'git-package-manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],archived_files=len(records),archived_chunks=len(chunks),bytes=report['total_archived_data_bytes'],core_code_files=len(code))),flush=True)

if __name__=='__main__':main()
