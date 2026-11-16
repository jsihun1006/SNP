# Naive possession baseline

`starter/baseline.py`의 최근접 선수 규칙과 고정 평활화 네 가지를 재사용한다.
학습은 하지 않으며 기본 기준은 주최 측의 `baseline_smooth50`으로 고정했다.
공개 GT는 생성 단계에서 읽지 않는다. 검증·평가는 공개 5분의 파이프라인 확인용이다.

## 실행

저장소 루트에서 NumPy/pandas가 있는 Python으로 실행한다.

```bash
python3 -m pip install -r requirements.txt
python3 model/점유율계산/계산알고리즘/naive/run.py
```

개별 실행:

```bash
python3 model/점유율계산/계산알고리즘/naive/baseline.py
python3 model/점유율계산/계산알고리즘/naive/validate.py
python3 model/점유율계산/계산알고리즘/naive/evaluation.py --bootstrap 1000
```

자동 입력 선택:

1. `data/tactical/BAR-ATH_tactical_atd.txt`와 `_atd.xml`이 모두 있으면 좌표 기반 추론을 한다.
   `MTAH_DATA_DIR`도 공식 `games.py`와 동일하게 따른다.
2. 없으면 `evaluation/results/baseline_raw.csv`의 H/A/D를 원시 코드로 복원하고 평활화한다.
   이 경로는 좌표 추론의 신규 실행이 아니다. `run_manifest.json`에 사용 경로·해시를 기록한다.

원본 위치가 다른 경우:

```bash
python3 model/점유율계산/계산알고리즘/naive/run.py --input-mode atd \
  --atd-raw /absolute/path/BAR-ATH_tactical_atd.txt \
  --atd-meta /absolute/path/BAR-ATH_tactical_atd.xml
```

임의의 `frame,label` CSV나 homography CSV는 선수·공 좌표 파일이 아니다.
이 baseline은 MP4를 읽지 않으며, `mp4/BAR-ATH_tactical-001.mp4`는 결과 시각 확인에 사용 가능하다.
시각은 원본 전술 영상 프레임/25로 계산하므로 편집된 영상에는 별도 정렬 검증이 필요하다.

## 모델

- ATD 프레임 ID에서 1을 빼서 제출 프레임으로 바꾼다.
- 정규화 좌표를 105.3 × 68m로 변환하고 공에 가장 가까운 선수의 팀을 선택한다.
- 최대 거리/실제 컨트롤 조건이 없고 속도·태그·영상·homography를 사용하지 않는다.
- 관측 답이 없으면 D로 처리한다. D 예측이 실제 데드볼 판단의 증거는 아니다.
- raw, ±25, ±50, ±125프레임을 그대로 만든다. 투표는 H/A만 세며 동률이면 H.
  윈도 안 H/A가 하나도 없을 때만 D. 미래 프레임도 쓰는 오프라인 규칙이다.

## 파일과 결과

| 파일 | 역할 |
|---|---|
| `baseline.py` | GT 없이 예측 생성; 공개 예측과 바이트 단위 재현성 확인 |
| `validate.py` | 공식 validator로 199,319행·0기반 연속 프레임·라벨 검사 |
| `evaluation.py` | 공식 scorer를 그대로 호출하고 추가 진단 CSV 작성 |
| `common.py` | 경로·공식 모듈·해시·영상 시각 변환 |
| `run.py` | 생성 → 검증 → 평가를 순서대로 실행 |
| `RESULTS.md` | 실제 실행의 점수·한계·해석 |
| `predictions/` | 제출 형식의 네 가지 `frame,label` CSV |
| `scoreboard.csv`, `reports/` | 공식 점수·클래스별 결과·전환 매칭/누락/가짜 전환 |
| `diagnostics/` | 프레임별 비교, 오답 구간, 혼동행렬, 라벨 분포 |
| `validation.json`, `run_manifest.json` | 형식검증·입력 출처·재현성 해시 |

`frame_diagnostics.csv`의 `is_graded=0`, `primary_correct=-1`은 정답이 없는 구간이다.
오답으로 세지 않는다. `gt_transition`은 평가 구간 내부의 실제 라벨 전환이며,
`primary_transition`은 전체 영상에서 예측 라벨이 바뀐 표시다.
`error_segments.csv`는 같은 GT/예측 쌍이 연속된 오답 구간으로 실제 점유 구간과 다르다.
구간 CSV는 끝 프레임 포함/제외를 열 이름으로 명시했다.

생성된 프레임 진단·예측·로그는 이 폴더의 `.gitignore`로 제외한다.
정리 이전 실행의 manifest에 기록된 절대경로는 실행 당시 위치를 나타내는 이력이다.

정리된 구조에서 `predictions/`, `diagnostics/`, `reports/` 등은 상위 결과 폴더를 가리키는 상대 심볼릭 링크다. 기존 실행 manifest의 절대경로는 생성 당시 경로이므로 현재 경로와 다를 수 있다.
