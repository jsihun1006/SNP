# Windows용 Tryolabs V4 실행·전달 패키지

대상: Windows 10, Python3.11 x64, 기존 GTX1660 SUPER/CUDA 가상환경. 프로젝트는 `C:\Users\kjkim\mtah_gpu\SNP`, 가상환경은 `C:\Users\kjkim\mtah_gpu\.venv`다. 그 아래에 `more-than-a-hack-2027` 폴더를 추가로 만들지 않는다.

이 전달 코드는 기존 V4 4.0.1의 실행을 묶은 것이다. HSV·역할 정책·V3/V4 분류 코드는 수정하지 않았다. V4는 CPU 후처리이며 YOLO·Norfair를 재실행하지 않는다. PyTorch/CUDA가 없어도 V4 후처리 자체에는 필요 없지만, 기존 설치를 교체하지 않는다. GK_A 미확정·STAFF 문맥 부족 등의 기존 한계도 유지한다. [V4 실험 보고서](TRYOLABS_CLASSIFICATION_V4.md).

새 파일:

- [run_tryolabs_v4.cmd](../windows/run_tryolabs_v4.cmd): 기존 가상환경의 Python으로 진입.
- [windows_tryolabs_v4.py](../starter/windows_tryolabs_v4.py): 환경·영상·캐시 점검, 입력 선택, 테스트/실행/분석, 단계별 로그.
- [실행 설정](../windows/tryolabs_v4_windows.json): 캐시 후보와 기본 표본 시간. 유니폼 분류 설정과 별도다.
- [V4 전용 requirements](../requirements-tryolabs-v4-windows.txt): numpy/OpenCV/Matplotlib/FFmpeg/pytest만 설치. torch/torchvision/torchaudio와 그것을 요구하는 의존성이 없다.
- [패키지 생성기](../starter/package_tryolabs_v4_windows.py), [덮어쓰기 없는 설치기](../windows/install_tryolabs_v4_bundle.py), [전달 테스트](../tests/test_tryolabs_v4_windows_handoff.py).

Git으로 받는 경우 CMD에서 다음을 순서대로 실행한다.

```bat
cd /d C:\Users\kjkim\mtah_gpu\SNP
git switch KKJ
git pull --ff-only origin KKJ
windows\run_tryolabs_v4.cmd --install --check
windows\run_tryolabs_v4.cmd --tests
```

`--install --check`는 기존 venv에 V4 후처리 의존성을 설치하고 사전 점검만 한다. 설치 전후 torch/torchvision/torchaudio의 패키지 버전 메타데이터가 바뀌면 중단한다. CUDA 테스트·GPU 벤치마크·모델 로딩은 하지 않는다. `--check`가 통과하면 다음 `--tests` 명령이 전달 패키지의 V3/V4/실행기 테스트, V3/V4 비교 처리, 추가 분석을 순서대로 실행한다. 모델과 결과를 준비하는 `setup_tryolabs_windows.py --prepare-assets`는 필요 없다.

기본 Python 경로가 다를 때만 다음처럼 지정한다. 전역 Python이나 새 venv로 자동 대체하지 않는다.

```bat
set "MTAH_V4_PYTHON=C:\Users\kjkim\mtah_gpu\.venv\Scripts\python.exe"
windows\run_tryolabs_v4.cmd --check
```

필요한 데이터 구조는 다음과 같다. 영상·검출·Track 데이터는 Git/코드 ZIP에 포함되지 않는다.

```text
SNP\
  windows\run_tryolabs_v4.cmd
  starter\windows_tryolabs_v4.py
  starter\configs\...
  data\broadcast\BAR-ATH_broadcast-001.mp4
  data\derived\tryolabs_broadcast\
    classification_v3\<V3실행폴더>\input_frames.jsonl
    또는
    full_match_v2\<V2실행폴더>\run.sqlite
    full_match_v2\<V2실행폴더>\chunks\...\chunk.json
    full_match_v2\<V2실행폴더>\chunks\...\tracking.jsonl.gz
    full_match_v2\<V2실행폴더>\chunks\...\objects.jsonl.gz
```

V3의 `input_frames.jsonl`에는 검출 공 후보·Track·프레임 정보가 들어 있으므로 그대로 사용한다. 이 파일과 원본 영상만 있어도 실행할 수 있다. V2 캐시 경로를 쓰는 경우 SQLite만으로는 부족하고 요청 시간대의 Track 청크가 완료돼 있어야 한다. 복사할 때 캐시 작성 프로세스를 먼저 멈춘다. 실행 중 DB의 `-wal/-shm`이 있으면 SQLite와 함께 보존해야 한다.

자동 모드는 설정에 기재한 V3 입력을 먼저 찾고, 다른 V3 실행 폴더도 탐색한다. 없으면 V2 전체 처리 폴더를 찾는다. 사용할 수 없는 후보는 이유를 출력하고 다음 후보를 검사한다. V3 입력은 들어 있는 표본 전체를 최대 5,000프레임까지 처리하며, V2 기본 표본은 파일 10·30·60분부터 각각 10초, 총 750프레임이다. 선택한 실제 입력 경로와 프레임 수를 실행 전에 출력한다. 특정 실험을 재현하려면 자동 선택 대신 경로를 명시한다.

```bat
windows\run_tryolabs_v4.cmd --source prepared --cache "data\derived\tryolabs_broadcast\classification_v3\windows_samples_01\input_frames.jsonl" --check
windows\run_tryolabs_v4.cmd --source prepared --cache "data\derived\tryolabs_broadcast\classification_v3\windows_samples_01\input_frames.jsonl" --tests
```

V2 전체 캐시에서 실행하려면 실제 폴더명으로 바꾼다.

```bat
windows\run_tryolabs_v4.cmd --source full --cache "data\derived\tryolabs_broadcast\full_match_v2\windows_gtx1660" --check
windows\run_tryolabs_v4.cmd --source full --cache "data\derived\tryolabs_broadcast\full_match_v2\windows_gtx1660" --tests
```

사전 점검은 청크 메타데이터와 필수 파일, 영상 1920×1080/25fps/164487프레임 및 첫 프레임 디코딩, FFmpeg 실행을 확인한다. 개별 gzip 행과 검출 DB 내용은 V4 본 실행에서 복사본으로 추가 검증한다. 캐시가 없거나 깨졌다면 오류로 중단한다. 누락을 채우기 위한 재추론·원본 캐시 수정·V3 결과 덮어쓰기는 하지 않는다.

결과는 매번 새 UTC 시간 이름으로 저장한다.

```text
data\derived\tryolabs_broadcast\classification_v4\windows_jobs\<시간>\
  launcher_status.json
  stage_01.log ...
  result\comparison_v3_v4.mp4
  result\summary.json
  result\objects.jsonl
  result\objects.csv
  result\frames.csv
  result\manifest.json
  result\verification.json
  result\analysis\diagnostics.json
```

콘솔의 마지막 `SUCCESS`와 `launcher_status.json`의 `status=SUCCESS`를 확인한다. 영상은 출력 프레임 수까지 디코딩을 확인한 후 완료 처리한다. 영상/Contact Sheet/CSV/HSV 그래프와 코드·설정 해시는 기존 V4 형식을 그대로 사용한다. `--output`으로 직접 지정해도 새 `classification_v4` 하위 폴더만 허용한다.

`--dry-run`은 점검 후 실제 실행할 명령을 출력하고 멈춘다. 빠른 로그 전용 검증은 `--no-video --no-visuals --no-analysis`로 실행할 수 있다. 이 경우 비교 영상이 생성되지 않는 것이 정상이다.

```bat
windows\run_tryolabs_v4.cmd --dry-run
windows\run_tryolabs_v4.cmd --no-video --no-visuals --no-analysis
```

실패하면 단계 로그와 `launcher_status.json`에 FAILED/오류를 남기며, Ctrl+C는 INTERRUPTED로 남긴다. 현재 V4에는 자체 중간 재개가 없으므로 재실행하면 새 출력 폴더에서 표본을 처음부터 처리한다. 이미 완료된 원본 검출·Track 캐시를 재생하므로 YOLO가 다시 실행되지는 않는다. 전체 경기 실행용으로 이 CMD를 사용하지 않는다.

ZIP으로 전달하는 경우에는 기존 프로젝트에 바로 덮어 풀지 않는다. 새 폴더에 압축을 풀고 설치기를 실행한다. 아래는 다운로드한 ZIP을 `C:\Users\kjkim\Downloads\tryolabs_v4_windows.zip`에 둔 예시다.

```bat
powershell -NoProfile -Command "Expand-Archive -LiteralPath 'C:\Users\kjkim\Downloads\tryolabs_v4_windows.zip' -DestinationPath 'C:\Users\kjkim\Downloads\tryolabs_v4_windows'"
"C:\Users\kjkim\mtah_gpu\.venv\Scripts\python.exe" "C:\Users\kjkim\Downloads\tryolabs_v4_windows\windows\install_tryolabs_v4_bundle.py" --target "C:\Users\kjkim\mtah_gpu\SNP" --check
"C:\Users\kjkim\mtah_gpu\.venv\Scripts\python.exe" "C:\Users\kjkim\Downloads\tryolabs_v4_windows\windows\install_tryolabs_v4_bundle.py" --target "C:\Users\kjkim\mtah_gpu\SNP"
cd /d C:\Users\kjkim\mtah_gpu\SNP
windows\run_tryolabs_v4.cmd --install --check
windows\run_tryolabs_v4.cmd --tests
```

설치기는 ZIP 안의 SHA256 manifest를 확인하고 없는 코드만 만든다. 기존 파일은 동일하거나 LF/CRLF 차이뿐이면 그대로 둔다. 실질적인 내용이 다르면 **어떤 대상 파일도 쓰기 전에 중단**한다. 충돌하는 기존 코드는 백업·확인 후 Git의 KKJ 버전과 맞추거나 새 대상 프로젝트를 사용한다. `data/`, 영상·가중치·캐시·Track·대용량 결과는 설치 목록에 들어갈 수 없다. 해시 manifest는 손상 검사용이며 발신자 신원을 인증하는 서명은 아니다.

패키지를 다시 생성하는 명령이다. 출력 ZIP이 이미 있으면 중단한다.

```bat
python starter\package_tryolabs_v4_windows.py --output data\derived\tryolabs_broadcast\classification_v4\windows_delivery_02\tryolabs_v4_windows.zip
```

2026-11-12 Mac Python3.11 CPU에서 관련 테스트 **104개 통과(신규 전달 테스트 14개 + 기존 90개)**를 확인했다. 동봉한 V3/V4/전달 테스트만 실행하는 `--tests` 단계는 **58개 통과**였다. ZIP의 코드 허용 목록, SHA256 손상 검사, 충돌 시 무변경, 재설치 무변경, LF/CRLF 기존 파일 보존, 공백 경로, 캐시 선택·누락·5,000프레임 상한·자식 프로세스 실패 처리를 검사했다.

실행기를 통한 실제 CPU 후처리도 확인했다. 로그 전용 모드와 기본 영상/분석 모드에서 각각 기존 **1,000프레임·12,513개 관측**을 처리했고 SUCCESS를 기록했다. 기본 모드 비교 영상은 1,000프레임 디코딩 검사를 통과했다. 객체 분류 로그는 이전 V4 결과의 같은 1,000프레임과 바이트 단위로 같고, 기존 파일 1,084개 보존 검사도 통과했다. 첫 테스트 시 Mac 임시 경로의 `/var`와 `/private/var`를 같은 경로로 비교하지 못한 테스트 실패가 있었으며, 정규화된 경로를 비교하도록 테스트를 고쳤다. 실패 로그도 보존했다.

실행 로그는 로컬 `data/derived/tryolabs_broadcast/classification_v4/windows_jobs/20261112_084326_427073/launcher_status.json`에 있다. 전달 ZIP은 `data/derived/tryolabs_broadcast/classification_v4/windows_delivery_01/tryolabs_v4_windows.zip`에 생성한다. 코드·설정·테스트·문서 30개와 해시 manifest만 포함하며 데이터·모델 파일은 0개다.

Windows CMD 네이티브 실행과 GTX1660 SUPER 실측은 이번 환경에서 확인하지 못했다. Mac의 실제 CPU 실행과 모의/합성 테스트를 Windows GPU 검증으로 표현하지 않는다.
