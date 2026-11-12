# Windows 10 / GTX 1660 SUPER에서 Tryolabs V2 실행

대상: Python3.11 x64, 기존 PyTorch2.9.1+cu126, GTX1660 SUPER6GB, i5-10400F, RAM16GB.
프로젝트는 `C:\Users\kjkim\mtah_gpu\SNP`, 가상환경은 `C:\Users\kjkim\mtah_gpu\.venv`를 사용한다.

이번 작업에서 Windows GPU나 MacBook GPU의750프레임 벤치마크를 실행하지 않았다.
Windows 분기/CUDA/OOM 모의 테스트와 합성 영상의 실제 CPU 재개 테스트를 수행했다.
Windows에서의 실제 처리량·최대 VRAM·CUDA 드라이버 호환성은 아래 명령으로 확인해야 한다.

## 바뀐 파일과 유지한 설정

- `starter/run_tryolabs_broadcast.py`: `load_model(path, device='cpu')`, `--device`, `--model-residency` 추가.
  CPU 기본값은 유지하며 CUDA/MPS를 명시할 수 있다. Matplotlib 캐시는 OS 임시 폴더를 사용한다.
- `starter/tryolabs_runtime.py`: 장치 선택, 공유 detector pair, GPU 상주 관리, OOM 기록·microbatch 재시도,
  Windows `msvcrt`/Mac·Linux `flock` 잠금.
- `starter/tryolabs_full_match_v2.py`: CUDA 장치, 작은 기본 배치, 선택적 기존 캐시, 청크 저장·재개,
  작업별 입력/설정/모델 해시 snapshot, 중단·오류 기록. 이번에 실행 가능한 소스를 Git에 추가했다.
- `starter/analyze_tryolabs_full_match_v2.py`: 기존 Mac750프레임 결과 없이도 분석 가능,
  Windows FFmpeg concat 경로, UTF-8/CSV 처리, 별도 보고서 경로, KST 시간 처리.
- `starter/setup_tryolabs_windows.py`: 공개 코드·가중치 준비 및 기존 CUDA 환경 검사.
- `starter/benchmark_tryolabs_windows.py`: 새 검출을 수행하는 동일3구간750프레임 명령.
- `requirements-tryolabs-windows.txt`, `tests/test_tryolabs_windows.py`, `tests/test_tryolabs_full_match_v2.py`.

V2 HSV JSON·분류 정책·`tryolabs_uniform_v2.py`, 가중치, 기존 V1/V2 결과는 변경하지 않았다.
기존 Mac용 `requirements-tryolabs.txt`도 그대로다.640/BGR/FP32, confidence 기준,45px/20프레임 기준을 유지한다.
이 작업은 Windows 실행 지원이며 새 분류 모델을 학습하거나 분류 성능을 측정한 작업이 아니다.

## 1. 설치 — Windows CMD

새 CMD를 열고 다음을 실행한다. 이후 명령도 모두 같은 가상환경에서 실행한다.

```bat
cd /d C:\Users\kjkim\mtah_gpu\SNP
call C:\Users\kjkim\mtah_gpu\.venv\Scripts\activate.bat
git switch KKJ
git pull --ff-only origin KKJ
python -c "import sys,torch; print(sys.executable); print(torch.__version__,torch.version.cuda); assert sys.version_info[:2]==(3,11); assert torch.__version__.split('+')[0]=='2.9.1' and torch.version.cuda=='12.6'"
python -m pip install -r requirements-tryolabs-windows.txt
```

Windows requirements에는 torch/torchvision/torchaudio가 없으며, 이를 요구하는 thop도 추가하지 않았다.
기존 CUDA torch를 일반 PyPI 버전으로 교체하는 명령은 사용하지 않는다.
[PyTorch 공식 버전 표](https://pytorch.org/get-started/previous-versions/)에서2.9.1의 torchvision 조합은0.24.1이며 CUDA12.6 wheel index가 제공된다.

torchvision이 없거나 맞지 않을 때만 다음을 실행한다. `--no-deps`가 torch 재설치를 막는다.

```bat
python -m pip install --no-deps --force-reinstall torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
```

공개 코드/모델을 로컬 `data/`에 준비하고 검사한다. Git과 인터넷 연결이 필요한 준비 단계이며,
이미 있는 checkout/모델의 버전이나 해시가 다르면 덮어쓰지 않고 중단한다.

```bat
python starter\setup_tryolabs_windows.py --prepare-assets
python starter\setup_tryolabs_windows.py --check
```

원본 영상은 별도로 다음 로컬 경로에 복사돼 있어야 한다.

```text
C:\Users\kjkim\mtah_gpu\SNP\data\broadcast\BAR-ATH_broadcast-001.mp4
```

검사는 Python/torch/torchvision/CUDA, 작은 CUDA 연산, CPU NMS, 모델 SHA256, 소스 commit,
원본164,487프레임·1920×1080·25fps 및 첫 프레임 디코딩을 확인한다. 처리량 벤치마크가 아니다.
기존 Mac의 검출·분류 결과를 Windows로 복사할 필요는 없다.

## 2. 회귀 테스트와750프레임 벤치마크

```bat
set PYTHONHASHSEED=0
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=2
python -m pytest -q tests\test_tryolabs_classification_v2.py tests\test_tryolabs_full_match_v2.py tests\test_tryolabs_windows.py
python starter\benchmark_tryolabs_windows.py --device cuda --batch 1 --model-residency single --output data\derived\tryolabs_broadcast\full_match_v2\windows_benchmark
```

파일10분·30분·60분부터 각250프레임, 총750프레임을 새로 추론한다.
동일한 검출 캐시를 V1/V2가 공유하며 각 구간의 추적·분류·비교 영상까지 처리한다.
`--reuse-750-cache`는 사용하지 않는다. 모델을 구간별로 로드하므로 측정값은 각stage의wall time으로 읽는다.

結果は `windows_benchmark\benchmark_summary.json`、各`segment_1..3\chunks\...\comparison.mp4`に保存する。
`detected_frames=750`、`processed_frames=750`と各区間の`coverage.complete=true`を確認する。
同じコマンドは再開用にも使えるが、その場合`fresh_benchmark=false`になり新規処理速度の測定とは区別する。
最初から測り直す場合は `--output ...\windows_benchmark_2`のように新しい名前を指定する。

## 3. 全164,487フレーム実行

벤치마크가 완료되면 다음 명령을 순서대로 실행한다. Mac의`main`출력과 다른 이름을 사용한다.

```bat
python starter\tryolabs_full_match_v2.py --stage detect --device cuda --batch 1 --model-residency single --output data\derived\tryolabs_broadcast\full_match_v2\windows_gtx1660
python starter\tryolabs_full_match_v2.py --stage process --device cuda --batch 1 --model-residency single --chunk-size 500 --output data\derived\tryolabs_broadcast\full_match_v2\windows_gtx1660
python starter\analyze_tryolabs_full_match_v2.py --output data\derived\tryolabs_broadcast\full_match_v2\windows_gtx1660 --assemble --clips --verify
```

검출은 GPU, 추적·분류·동영상 출력은 CPU에서 처리한다. 위 순차 실행은 한 번에 하나의 큰 작업을 실행한다.
필요하면 detect를 실행 중인 상태에서 **두 번째 CMD**에 같은 가상환경과 `set PYTHONHASHSEED=0`을 설정한 뒤
process 명령에 `--follow`를 추가해 병행할 수 있다. 두 작업은 동일한 output/device/batch/residency를 사용해야 한다.

완료 후 파일:

- `comparison_full.mp4`: 전체 V1/V2 비교 영상.
- `examples\*.mp4`, `examples.json`: 자동 선정한 짧은 진단 구간.
- `summary.json`, `time_bins.csv`, `report.md`, `video_verification.json`: 통계·보고서·전체 출력 검증.
- `run.sqlite`: 공유 검출과 프레임 요약; 실행 중의`-wal/-shm`파일도 작업 데이터다.
- `chunks\*\tracking.jsonl.gz`, `objects.jsonl.gz`, `frames.jsonl.gz`, `state.pkl`: 단계별 로그·재개 상태.
- `preservation_before.json`, `preservation_after.json`: 해당 작업의 코드·설정·모델과 원본 영상size/mtime 보존 검사.

`summary.json`의`coverage.complete=true`와`video_verification.json`의
`passed=true`, `decoded_frames=164487`를 모두 확인해야 전체 처리가 완료된 것이다.
보고서가 이미 있으면 `--report data\derived\tryolabs_broadcast\full_match_v2\windows_gtx1660\report_2.md`처럼 새 경로를 지정한다.
경기 진행·리플레이 여부는 진단 가설이며, 근접 근거를 실제 controlled possession 정답으로 해석하지 않는다.

## 4. 6GB VRAM과 오류·재개

기본 권장은`--batch 1 --model-residency single`이다. 두 모델은 CPU에서 로드하고,
추론할 모델만 GPU에 올린 뒤 CPU로 돌려놓는다. FP16으로 조용히 바꾸지 않는다.
상주 전환 비용이 있으며 속도나6GB 내 성공을 보증하는 설정은 아니다.
[PyTorch CUDA 메모리 설명](https://docs.pytorch.org/docs/2.9/notes/cuda.html)에 따라 사용 중인 메모리와 allocator 예약량을 구분해서 기록한다.

- CUDA를 명시했는데 사용할 수 없으면 명확히 실패한다. CPU로 조용히 전환하지 않는다.
- 큰 batch의 CUDA OOM은 더 작은 microbatch로 재시도하고, 두 모델 상주 모드도 single로 낮춘다.
- single/batch1에서도 OOM이면 실패를 기록하고 중단한다. GPU를 쓰는 다른 프로그램을 닫고 같은 명령을 다시 실행한다.
- `runtime_events.jsonl`, `detect_failure.json`/`process_failure.json`에 오류와 대상 범위를 기록한다.
  `detector_status.json`에는 현재 메모리·최대 할당량과 실제 상주 모드도 기록한다.
- 검출은 성공한 batch의DB transaction만 완료 처리한다. OOM을 공/선수0개인 성공 프레임으로 저장하지 않는다.
- 추적/분류는 청크의 영상 전체 디코딩까지 확인한 뒤 checkpoint를 완료 처리한다.
  Ctrl+C 또는 비정상 종료 후 같은 명령을 반복하면 완료 batch/청크는 유지하고 미완성 부분부터 다시 처리한다.
- 중단 시점에 추론만 끝나고 DB에 commit되지 않은 batch는 재실행될 수 있다. V1/V2별 이중 추론을 의미하지 않는다.
- partial 청크는`interrupted\`에 남긴다. 완료 청크의 상태는 복원하며 청크 경계에서 임의로 추적을 초기화하지 않는다.

재개할 때 output·source·device·요청batch·residency·코드/설정/가중치가 달라지면 manifest 검사가 중단한다.
옵션을 바꾸거나 Mac의 예전 manifest를 Windows에 가져오는 경우에는 새 output 폴더를 사용한다.
이번 수정 이전 Mac 실행 결과/manifest는 변환하거나 덮어쓰지 않았다. 새 Mac 실행에는`--device cpu`또는`--device mps`를 사용할 수 있다.

## 검증 범위

2026-11-12 로컬 Python3.11 환경에서 관련 테스트 **46개 통과**.
합성 CPU 동영상에서4프레임 중단 후8프레임 재개가 연속 실행과 일치했다.
Windows 잠금, CUDA 장치 선택, single 상주, 모델 로드 실패, OOM 분할/중단, 실패 캐시 방지,
750프레임 명령과 fresh/resume 구분은 모의 테스트로 확인했다. 기존 유니폼·심판 테스트도 실행했다.
이 테스트를 Windows 네이티브 실행 또는 GTX1660 SUPER 처리량 측정으로 표현하지 않는다.
원본 영상·모델·검출/분류 결과·SQLite·동영상은 GitHub 커밋에 포함하지 않는다.
