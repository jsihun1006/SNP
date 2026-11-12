# V4 Full-Match Streaming Runner

2026-11-12, 스트리밍 실행기 `1.0.0`, 분류 알고리즘은 기존 V4 `4.0.1` 그대로다. 이번 작업은 **구현과 소규모 회귀 검증까지 완료**했다. **전체 164,487프레임은 실행하지 않았다.** Windows/D 드라이브 네이티브 실행도 이 Mac 환경에서는 검증하지 않았다. Windows의 기존 1,250프레임 검증 성공은 사용자가 전달한 상황이며, 아래 실제 수치는 로컬 원본·캐시에서 확인한 것이다.

추가 파일:

| 파일 | 기능 |
|---|---|
| [run_tryolabs_v4_streaming.py](../starter/run_tryolabs_v4_streaming.py) | run/resume, 검증, 병합, 상태 조회, 기존 결과 회귀 비교 |
| [tryolabs_v4_stream_state.py](../starter/tryolabs_v4_stream_state.py) | 프레임 단위 실행·집계·정확한 분류 상태 직렬화 |
| [tryolabs_v4_stream_io.py](../starter/tryolabs_v4_stream_io.py) | 원본을 건드리지 않는 DB snapshot·캐시 순회·원자적 저장 |
| [run_tryolabs_v4_full_match.cmd](../windows/run_tryolabs_v4_full_match.cmd) | 기존 Windows Python3.11 가상환경으로 실행 |
| [test_tryolabs_v4_streaming.py](../tests/test_tryolabs_v4_streaming.py) | 새 스트리밍/재개/완료 판정 테스트 13개 |

기존 V1/V2/V3/V4 코드·설정·실험 보고서·모델은 수정하지 않았다. 기존 5,000프레임 실행기의 제한을 제거하거나 우회하는 방식도 사용하지 않았다. 새 실행기는 기존 분류기와 표시 함수만 불러 프레임별로 호출하며 YOLO/Norfair 생성·추론·업데이트 경로를 갖지 않는다. PyTorch를 import하거나 새 GPU 모델을 학습하지 않는다.

입력은 `full_match_v2/<run>/run.sqlite`와 **DB의 chunks 테이블에 완료 등록된** 추적·분류 청크다. 디렉터리에 있는 `.work`나 미등록 청크는 사용하지 않는다. 필요한 파일은 `chunk.json`, `tracking.jsonl.gz`, `objects.jsonl.gz`다. 검출이 끝났어도 추적 청크가 끝나지 않았다면 전체 실행을 진행할 수 없다.

원본 SQLite에 연결하면 읽기 연결도 SHM에 영향을 줄 수 있으므로 DB/WAL/SHM을 먼저 새 V4 출력으로 OS 복사한다. 복사 전후 파일 집합·SHA256 및 복사본 해시가 같아야 한다. SQLite는 복사본에만 연결하고 backup API로 WAL을 합친 독립 `source.sqlite`를 만든다. 이후에는 이 D 드라이브 snapshot만 `mode=ro&immutable=1`로 연다. 원본 V2의 DB·WAL·SHM은 연결하거나 변경하지 않는다. 시작 전에 V2 캐시 작성/읽기 프로세스를 종료해 입력을 고정해야 한다.

입력의 검출 개수, 완료 청크 구간의 연속성, metadata start/end를 먼저 확인한다. 처리할 때도 tracking/object 두 스트림의 프레임 축·행 수·gzip 종료, DB 검출 상태를 확인한다. 누락·중복·손상은 오류로 중단하며 빈 검출 성공으로 바꾸거나 새 추론으로 채우지 않는다. 원본 캐시를 수정해 복구하지 않는다.

전체 객체를 list에 모으지 않는다. 입력과 출력 JSONL은 한 프레임씩 순회하고 즉시 gzip에 기록한다. 비교 영상도 프레임마다 FFmpeg pipe로 보낸다. 메모리에 남는 것은 현재 프레임 객체, 기존 FrameReader의 최대 8프레임 캐시, 최근 시간 상태, 카메라 모니터 10개 작은 영상, 집계 카운터와 입력 청크 목록이다. 소규모 회귀에서는 최대 프레임 객체 24개·시간 상태 Track 25개·카메라 이력 10개였다. 이 값은 RSS/전체 경기 메모리 실측이 아니다.

분류기에서 10프레임보다 오래 끊긴 Track은 다음 fresh 관측에서 원래 초기화되는 항목이므로 같은 조건으로 제거한다. 오래된 객체 배열이 누적되지 않으며, 원래의 lazy reset과 출력이 같음을 기존 1,250프레임 회귀로 검사했다. 카메라/시간 문맥·팀/역할·GK/OTHER 판정 정책은 재설정하거나 튜닝하지 않았다.

청크 경계는 저장 경계일 뿐 분류 초기화 조건이 아니다. 다음 상태를 **JSON 형식**으로 보존한다. Norfair checkpoint나 pickle을 읽지 않는다.

- 독립 V3 TemporalClassifier의 Track별 투표 deque·안정 레이블·최근 A 근거·충돌/연속 REF 이력.
- V4 TemporalClassifier와 내부 V3의 상태, GK 역할, OTHER 고정 이력, STAFF 연속 관측 등 모든 Track 상태.
- V3/V4의 팀별 20프레임 근거 카운터.
- CameraTransitionMonitor의 이미지·히스토그램 ndarray, dtype/shape, deque 길이와 순서.
- 직전 sequence/source frame, V4 epoch, 누적 통계·직전 프레임의 전환 비교 정보.

dict 정수/tuple 키와 deque maxlen, ndarray의 dtype·값을 구분해 직렬화한다. Scene/원본 프레임 불연속에서는 기존 실행기처럼 관련 상태를 초기화한다. 기존 V4 화면 전환 경보에서는 V4 시간 상태·점유 근거만 재확인하며 V3 기준선은 유지한다. 이 경보는 실제 컷의 정답이 아니고 빠른 팬/줌에도 반응할 수 있다. 청크 크기에 따른 새 초기화는 없다.

각 청크는 같은 D 볼륨의 `.work-*`에 다음을 쓴다.

```text
input_frames.jsonl.gz
objects.jsonl.gz
frames.jsonl.gz
comparison.mp4          (기본 생성)
comparison.log
state.json
chunk.json
```

영상 프레임/FPS/크기를 디코딩 검증하고 모든 파일을 flush/fsync한 뒤, 상태·출력 SHA256, 설정 해시, manifest 해시, 다음 ordinal/source frame을 기록한다. 마지막에 작업 디렉터리를 `chunks/<시작ordinal>-<끝ordinal>`로 한 번에 rename한다. 출력·분류 상태가 같은 트랜잭션으로 보이는 구조다. 파일 flush와 같은 볼륨의 원자적 rename을 사용하지만 저장 장치 자체의 전원 장애 보증을 의미하지 않는다.

`checkpoint.json`과 진행 summary는 그 뒤 원자적으로 갱신한다. 디렉터리 commit 직후 전역 포인터 갱신 전에 중단돼도, 재개 시 완료 청크 체인과 체크섬을 확인해 마지막 `state.json`을 복원한다. 미완성 `.work-*`는 `interrupted/`로 이동해 남기고 그 꼬리만 다시 분류한다. 완료 청크를 덮어쓰거나 중복 누적하지 않는다. 출력 lock은 Windows msvcrt/POSIX flock으로 두 실행기의 동시 기록을 막는다.

재개 시 입력 DB/WAL/SHM·Track·영상 SHA256, 분류/실행 코드·설정·Python/OpenCV/NumPy·FFmpeg 해시와 chunk size/video/start/end가 같아야 한다. 다르면 재개를 거부한다. 진행 중 Git pull이나 설정 변경을 하지 말고, 변경된 코드로 실행하려면 새 출력 폴더를 사용한다. 원본 영상 SHA256 계산은 스트리밍 읽기이며 파일 전체를 RAM에 올리지 않는다. 시작·재개·검증의 해시 계산에는 시간이 든다.

실제 회귀 결과는 다음과 같다.

| 검사 | 확인 결과 |
|---|---|
| 기존 V4 표본 | 1,250프레임, 14,471개 객체 관측 |
| 새 실행기 연속 실행 vs 기존 결과 | 모든 객체 필드·프레임 CSV 필드 일치 |
| 137프레임 청크, 4개 commit 후 중단 | 548프레임에서 정지, 골키퍼 상태가 성립한 구간 포함 |
| 중단 후 재개 vs 기존 결과 | 모든 객체 필드·프레임 필드 일치 |
| 연속 vs 재개 | 로그 내용과 모든 청크의 상태 JSON SHA256 일치 |
| 실제 V2 SQLite/Track 경로 | 원본 frame 8000–8249, 250프레임·4,381개 객체 관측 |
| 실제 캐시 중단·재개 | chunk size 73, 146프레임 commit 후 재개해 250개 완료 |
| 실제 비교 영상 | 병합 후 250프레임·25fps 디코딩 확인, 누락/중복 0 |
| 전체 경기 완료 표시 | 모든 실제 실험에서 `scope=sample`, `full_match_complete=false` |

[1,250프레임 회귀 보고서](../data/derived/tryolabs_broadcast/classification_v4/full_match_streaming/regression_02/regression.json), [연속 실행 요약](../data/derived/tryolabs_broadcast/classification_v4/full_match_streaming/regression_02/continuous/summary.json), [실제 V2 캐시 검증](../data/derived/tryolabs_broadcast/classification_v4/full_match_streaming/cache_smoke_01/verification.json), [250프레임 병합 영상](../data/derived/tryolabs_broadcast/classification_v4/full_match_streaming/cache_smoke_01/assembled/comparison_full.mp4).

로컬 `python -m pytest -q tests`는 **241개 테스트와 2개 하위 테스트 통과, 7.64초**였다. 신규 13개는 포터블 상태·GK 유지·20프레임 근거·청크/scene 구분·메모리 이력 상한·중간 중단·commit 직후 중단·코드/설정/입력 변경 거부·source 중복 프레임의 정확한 실패 위치·출력 손상 거부·영상 병합·완료 gate·출력 lock·D 드라이브 제한을 검사했다. 합성 영상에서는 연속/재개 실행의 비교 MP4 청크 SHA256도 일치했다. 완료 gate의 true 경로는 작은 합성 extent로 검사했으며 실제 전체 경기 완료로 해석하지 않는다.

재현된 V4 관측 분포는 TEAM_A 3,018 / TEAM_B 3,933 / GK_B 167 / REFEREE 778 / OTHER 297 / UNKNOWN 6,278이다. GK_A/STAFF는 기존 정책대로 확정하지 않았다. 같은 epoch의 분류 전환 V3 513/V4 454회, 점유 후보 ID 차이는 0프레임으로 이전 결과와 같다. 이 비교는 자동 결과의 회귀 검사이며 정답 평가나 분류 정확도 개선 주장이 아니다.

전체 통계는 `summary.json`에 누적된다. 클래스 분포·UNKNOWN 사유·모델 간 변경 표, GK 후보/OTHER·V3 REF→GK 의심 전환, 심판 제외, 역할/클래스 변경, 공 후보 0/1/2개 이상, 점유 상태/후보 변경, GK 몸통 근접·발 거리 부족, 파일 5분 간격 통계를 제공한다. 실제 STAFF 오류는 정답/문맥이 없어 `null`이다. 원본 45px와 20프레임 기준은 유지한다.

Windows CMD에서는 먼저 코드와 테스트를 받는다. 기존 CUDA 가상환경을 그대로 사용하며 필요한 후처리 의존성만 설치한다.

```bat
cd /d C:\Users\kjkim\mtah_gpu\SNP
git switch KKJ
git pull --ff-only origin KKJ
call C:\Users\kjkim\mtah_gpu\.venv\Scripts\activate.bat
python -m pip install -r requirements-tryolabs-v4-windows.txt
python -m pytest -q tests
```

먼저 Windows에서 성공했던 **1,250프레임 V4 결과 폴더**를 지정해 회귀 검사를 한다. 이 폴더 안에 `input_frames.jsonl`, `objects.jsonl`, `frames.csv`가 있어야 한다. 아래 `V4_REFERENCE` 한 줄은 실제 성공 결과 폴더로 바꾼다. 기존 실행기가 출력한 `SUCCESS: ...\result` 경로다.

```bat
set "V4_REFERENCE=C:\Users\kjkim\mtah_gpu\SNP\data\derived\tryolabs_broadcast\classification_v4\windows_jobs\실제1250실행폴더\result"
windows\run_tryolabs_v4_full_match.cmd regression --reference "%V4_REFERENCE%" --output "D:\MTAH2027\tryolabs_v4_stream\regression_001" --chunk-size 137
```

`regression.json`의 `passed=true`, continuous/resumed 각각 frames=1250, objects 일치와 `checkpoint_states_identical=true`를 확인한다. 원래 결과를 수정하지 않고 D 드라이브에 별도 두 실행을 만든다. regression은 비교 속도를 위해 영상은 만들지 않는다.

완료된 Windows 캐시를 이용한 작은 구간 재개·영상 검사 명령이다. D 드라이브 폴더는 자동 생성한다. 이미 있으면 새 이름을 지정하거나 같은 실행을 명시적으로 재개한다.

```bat
windows\run_tryolabs_v4_full_match.cmd run --cache "data\derived\tryolabs_broadcast\full_match_v2\windows_gtx1660_full" --start 8000 --end 8250 --chunk-size 73 --limit-chunks 2 --output "D:\MTAH2027\tryolabs_v4_stream\smoke_001"
windows\run_tryolabs_v4_full_match.cmd run --cache "data\derived\tryolabs_broadcast\full_match_v2\windows_gtx1660_full" --start 8000 --end 8250 --chunk-size 73 --resume --output "D:\MTAH2027\tryolabs_v4_stream\smoke_001"
windows\run_tryolabs_v4_full_match.cmd assemble --output "D:\MTAH2027\tryolabs_v4_stream\smoke_001"
windows\run_tryolabs_v4_full_match.cmd verify --decode-video --output "D:\MTAH2027\tryolabs_v4_stream\smoke_001"
```

위 검사를 통과한 후 전체 처리 명령은 다음이다. 입력 후보 경로를 그대로 사용한다. 기본 chunk size는 250이며, end는 미포함이므로 실제 frame 0–164486을 처리한다. 영상은 프로젝트의 `data\broadcast\BAR-ATH_broadcast-001.mp4`에서 읽는다.

```bat
windows\run_tryolabs_v4_full_match.cmd run --cache "data\derived\tryolabs_broadcast\full_match_v2\windows_gtx1660_full" --start 0 --end 164487 --chunk-size 250 --output "D:\MTAH2027\tryolabs_v4_stream\full_001"
```

중단 후에는 **같은 설정과 output**에 `--resume`만 추가한다.

```bat
windows\run_tryolabs_v4_full_match.cmd run --cache "data\derived\tryolabs_broadcast\full_match_v2\windows_gtx1660_full" --start 0 --end 164487 --chunk-size 250 --resume --output "D:\MTAH2027\tryolabs_v4_stream\full_001"
```

다른 CMD에서 진행 상황을 조회할 수 있다. 이 명령은 빠른 commit metadata 조회이며 전체 파일 무결성을 재검증한 결과는 아니다.

```bat
cd /d C:\Users\kjkim\mtah_gpu\SNP
windows\run_tryolabs_v4_full_match.cmd status --output "D:\MTAH2027\tryolabs_v4_stream\full_001"
```

처리가 끝나면 병합하고 완료 여부를 검증한다. verify는 출력 gzip/체크섬·프레임 축·클래스 합계·원본 입력 불변과 병합 영상의 검증 기록/해시를 확인한다. 다시 전부 디코딩하려면 `--decode-video`를 추가한다. 병합 자체에서도 전체 영상을 디코딩하므로 대용량에서는 추가 시간이 필요하다.

```bat
windows\run_tryolabs_v4_full_match.cmd assemble --output "D:\MTAH2027\tryolabs_v4_stream\full_001"
windows\run_tryolabs_v4_full_match.cmd verify --require-full --output "D:\MTAH2027\tryolabs_v4_stream\full_001"
```

전체 완료 조건은 `verification.json`에서 **expected_frames=164487, committed_frames=164487, missing_frames=0, duplicates=0, comparison_video_complete=true, full_match_complete=true**다. 명령 종료 코드도 0이어야 한다. 처리만 끝나고 영상이 없거나 표본 범위만 끝난 경우 `--require-full`은 실패한다. `status.json`은 처리 진행 기록이며 최종 완료 판단은 verify 결과를 사용한다.

최종 D 드라이브 구조:

```text
D:\MTAH2027\tryolabs_v4_stream\full_001\
  manifest.json                코드·설정·입력 해시
  source.sqlite                V2 검출의 독립 snapshot
  checkpoint.json              최근 commit 포인터
  summary.json                 전체/5분 단위 누적 진단
  status.json                  처리 진행
  failure.json                 오류/중단 위치 (발생했을 때)
  chunks\000000000-000000250\
    input_frames.jsonl.gz
    objects.jsonl.gz
    frames.jsonl.gz
    comparison.mp4
    state.json
    chunk.json
  interrupted\                미완성 작업 보존
  assembled\comparison_full.mp4
  assembled\assembly.json
  verification.json
```

`failure.json`은 재개 성공 후에도 과거 진단으로 남을 수 있다. status 명령은 이를 `historical_failure`로 표시한다. 누락/손상/설정 변경을 발견한 뒤 입력을 몰래 수정해 이어 붙이지 않는다. 원인을 수정해 입력이 달라졌다면 새 출력 폴더로 실행한다. 기존 청크·영상·보고서는 덮어쓰지 않는다.

이 구현에서는 원본 영상과 source cache도 SHA256으로 시작/재개/처리 후 검증한다. 로컬 실제 V2 캐시 250프레임 실험은 `source_files_unchanged=true`였다. 기존 V4 코드·설정 및 기존 결과 보존 검사는 별도로 수행했다. 새 결과만 `classification_v4/full_match_streaming/`에 만들었으며 Windows 전체 실행에서는 반드시 D 드라이브가 강제된다.

대회 [RULES.md](../RULES.md)를 준수했다. 제공 경기 수동 정답·이벤트·Smart Tagging을 생성/사용하지 않았다. 비교 기준은 기존 자동 V4 결과다. GK_A 미확정, STAFF 문맥 부족, 약한 골문·화면 전환 휴리스틱, 손으로 공을 잡은 GK의 발 거리 한계는 기존 V4 그대로 남아 있다. 전체 구간을 경기 중/실시간 플레이로 단정하지 않고 `UNVERIFIED`를 유지한다.

Windows 전체 입력 후보 `windows_gtx1660_full`은 이 로컬 작업 환경에 없어 전 범위 완료 여부를 확인하지 못했다. 로컬의 완료된 V2 tracking은 0–8,999프레임뿐이며 그중 8,000–8,249만 실제 연결 검사했다. 164,487프레임의 처리량·RAM/디스크 사용량·전체 클래스/점유 통계는 실제 Windows 전체 실행 후에 확인해야 한다. 현재 전체 완료를 주장하지 않는다.
