import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

folder = Path(r'H:\data\ball-tracking-results\first_half')
outputs = Path(__file__).resolve().parent.parent/'outputs'
complete = json.loads((folder/'job_complete.json').read_text(encoding='utf-8'))
config = json.loads((folder/'run_config.json').read_text(encoding='utf-8'))
assert complete['status'] == 'complete'
first, last = config['start_frame'], config['end_frame']
digest = hashlib.sha256()
count = candidates = nonempty = 0
with (folder/'candidates.jsonl').open('rb') as stream:
    for line in stream:
        digest.update(line)
        row = json.loads(line)
        assert row['frame_idx'] == row['source_frame'] == first + count
        assert row['part_frame'] == count
        assert row['part_id'] == 'first_half'
        candidates += len(row['candidates'])
        nonempty += bool(row['candidates'])
        count += 1
assert count == last - first == complete['frames'] == 79467
assert digest.hexdigest() == complete['file_sha256']
verified = dict(verified=True, frames=count, source_start_inclusive=first, source_end_exclusive=last,
                frame_coverage_contiguous=True, file_sha256_matches=True, sha256=digest.hexdigest(),
                total_candidate_observations=candidates, frames_with_candidates=nonempty,
                result_file=str(folder/'candidates.jsonl'), bytes=(folder/'candidates.jsonl').stat().st_size,
                completed_at_kst=datetime.fromtimestamp(complete['finished_unix'],timezone(timedelta(hours=9))).isoformat(),
                source_files_modified=complete['source_files_modified'],
                accuracy_validated=complete['ground_truth_accuracy_validated'])
(outputs/'completion-verification.json').write_text(json.dumps(verified,indent=2),encoding='utf-8')
(outputs/'pc1-job-complete.json').write_text(json.dumps(complete,indent=2),encoding='utf-8')
text = f'''# PC1 전반 처리 완료

- 완료 시각(한국 시간): {verified['completed_at_kst']}
- 처리: {count:,} / {count:,}프레임, 100%
- 원본 프레임: [{first}, {last})
- 원본 재생 시간: 00:04:46.84 ~ 00:57:45.52 (끝 제외)
- 검증: 전 프레임 순서·연속성·중복/누락 없음, 파일 SHA256 일치
- 원본 영상·모델 파일 변경: 없음
- 결과 파일 크기: {verified['bytes']:,}바이트

## 결과 위치

`{folder}`

- `candidates.jsonl`: 전반 통합 공 후보 결과
- `run_config.json`: 입력 파일 해시·범위·검출 설정
- `job_complete.json`: 완료 상태·결과 SHA256
- `chunks`: 중단·재개용 청크 기록

후반 PC와 병합할 때는 `run_config.json`, `job_complete.json`, `candidates.jsonl`을 같은 구조로 전달하면 됩니다. 두 PC는 같은 영상과 같은 모델 파일을 사용해야 합니다.

총 후보 관측은 {candidates:,}개이며, {nonempty:,}프레임에 후보가 있습니다. 후보 개수는 정확도를 의미하지 않습니다. 이 결과는 공 후보의 이미지 좌표이며 최종 경기 공 트래킹이나 정답 대비 검출 정확도 검증 결과는 아닙니다.

최적화 시험 결과, 큰 배치는 속도 이득이 없고 일부 관측값이 달라져 적용하지 않았습니다. 원래 모델·FP32·해상도·배치 1/4/8·타일·프레임 범위·온도 제어를 유지했습니다.
'''
(outputs/'COMPLETED_KO.md').write_text(text,encoding='utf-8')
print(json.dumps(verified))
