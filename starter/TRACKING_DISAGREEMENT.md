# Broadcast–Tactical 트래킹 비교

두 ATD의 좌표 불일치, 관측 선수 수, 공의 관측·결측을 로컬에서 확인하는 도구다. 기존 프레임 정렬과 원본 데이터를 변경하지 않는다. 실제 선수 위치의 정답 정확도를 측정하는 도구는 아니다.

## 포함된 코드

| 파일 | 역할 |
| --- | --- |
| `export_tracking_disagreement.py` | 같은 팀 좌표 집합의 대응, 선수 거리·관측 수 비교, MP4와 탐색 플레이어 생성 |
| `export_tracking_disagreement_ball.py` | 기존 선수 비교 영상 아래에 독립된 공 전용 패널 추가 |
| `add_tracking_match_clock.py` | 재생·프레임 탐색에 연동되는 추정 경기시계를 HTML 플레이어에 추가 |
| `../tests/test_tracking_disagreement*.py` | 합성 좌표, 대응, 결측, 선수·공 분리 및 영상 검증 |
| `../tests/test_tracking_match_clock.py` | 경기시계·추가시간·탐색 검증 |

공유 함수는 이미 저장소에 포함된 `tracking_utils.py`, `compare_tracking_player.py`, `view_all_movement_vectors.py`를 사용한다. 외부 모델이나 이전 점유 모델의 가중치는 필요하지 않다.

## 환경과 입력

프로젝트 루트에서 실행한다. Python 3.11과 `requirements-tracking.txt`의 환경을 사용한다.

```bash
conda activate mtah2027
python -m pip install -r requirements-tracking-comparison-dev.txt
```

기본 계산은 NumPy와 OpenCV로 실행된다. SciPy가 이미 설치돼 있으면 대응 계산에 선택적으로 사용하며, 없어도 NumPy Hungarian 구현으로 작동한다. SciPy를 필수로 설치하거나 다른 실험의 격리 의존성 폴더를 복사할 필요는 없다. 동률 대응은 backend에 따라 달라질 수 있다.

실제 영상 출력에는 OpenCV의 H.264 `avc1` 인코더가 필요하다. 합성 영상 테스트에서 인코더를 확인한다. 플랫폼과 OpenCV 빌드에 따라 지원 여부가 다를 수 있다. macOS의 JavaScript 실행 테스트는 로컬 `osascript`를 사용하며, 해당 실행기가 없는 환경에서는 그 두 테스트만 건너뛴다.

로컬 원본 ATD TXT/XML은 `data/tactical/`, `data/broadcast/`에 둔다. 다른 위치는 `MTAH_DATA_DIR` 또는 비교 스크립트의 `--data-dir`로 지정한다. 원본 경기 MP4는 이 좌표 비교를 생성하는 데 필요하지 않다.

기존 정렬표 `data/derived/sync/final_tracking_alignment.csv`가 필요하다. Git에 포함하지 않는 로컬 산출물이다. 정렬표가 없다면 기존 도구로 먼저 생성한다.

```bash
python starter/final_tracking_alignment.py
```

이 기존 정렬 도구는 후반 기준점을 정할 때 제공 Smart Tagging을 참조한다. 새 비교·공·시계 도구는 생성된 정렬표를 재사용하며 GT나 Smart Tagging을 새로 읽지 않는다. 각 하프 기준점 뒤의 일정 시차가 전체 구간에서 정확한지는 검증되지 않았으므로, 좌표 불일치에 시간 정렬 오차도 포함될 수 있다.

## 생성 순서

```bash
# 1. 선수 비교 및 원래 프레임 축의 수치
python starter/export_tracking_disagreement.py

# 2. 선수 지표와 분리된 공 전용 패널
python starter/export_tracking_disagreement_ball.py

# 3. HTML 플레이어에 경기시계 추가. MP4는 재인코딩하지 않는다.
python starter/add_tracking_match_clock.py
```

생성 후 다음 파일을 로컬 브라우저에서 직접 연다. 서버를 실행할 필요가 없다.

```text
data/derived/tracking_disagreement_ball_20261113/
  tracking_disagreement_ball_clock_player.html
```

HTML과 MP4를 같은 폴더에 유지한다. 공 플레이어는 선수 CSV에 상대 경로로 연결되므로 두 결과 폴더의 위치 관계도 유지한다. 재생/정지, 탐색 바, ±1프레임, 전·후반 시작, 원본 Tactical 프레임 검색을 지원한다. 시계를 포함한 전체화면은 플레이어의 `시계와 함께 전체화면` 버튼을 사용한다.

## 화면과 수치 해석

- 채워진 원=Tactical, X 모양=Broadcast, 파랑=Home, 빨강=Away. X 모양은 점유 라벨 X와 무관하다.
- 선수 수는 양 좌표가 유한한 슬롯 관측 수다. 실제 고유 선수 수를 보장하지 않는다. `B−T count`는 Broadcast 관측 수에서 Tactical 관측 수를 뺀 값이다.
- 전체 대응은 같은 팀 안에서 관측 수가 작은 쪽을 큰 쪽에 최소 제곱 거리로 일대일 대응한다. 잘못된 강제 대응도 포함될 수 있다.
- 제한 대응은 기본 5m 이내의 대응 수를 최대화한 다음 거리 제곱 합을 최소화한다. 제한 RMSE가 낮아도 대응 비율이 낮으면 일부 관측만 일치하는 것이다. 노란 선은 이 대응이며 실제 선수 동일성을 보장하지 않는다.
- 공은 선수 RMSE·인원·대응에 합산하지 않는다. 공 패널의 네 상태는 양쪽 관측, Tactical만, Broadcast만, 양쪽 결측이다. 관측 여부는 ATD 좌표 존재 기준으로 실제 영상의 육안 판정이 아니다.
- 공의 현재 거리와 dx/dy는 양쪽 모두 관측될 때만 계산하고, 나머지는 N/A다. 공에는 선수의 5m 제한을 적용하지 않는다.
- 최근 2초 공 RMSE는 같은 정렬 run의 현재 포함 과거 50프레임 중 공통 관측만 사용한다. 현재 결측이어도 과거 관측으로 계산될 수 있으므로 지원 프레임 수를 함께 확인한다. 결측을 0으로 채우거나 경계를 넘어 보간하지 않는다.
- 경기시계는 하프별 기존 정렬 시작점을 전반 00:00, 후반 45:00으로 놓은 추정값이다. 추가시간은 `45+MM:SS`, `90+MM:SS`로 표시한다. 영상 재생 시간 및 실제 방송 전광판 시각과 구분한다.

## 산출물과 보존

선수 비교는 `data/derived/tracking_disagreement_20261113/`, 공과 경기시계는 `data/derived/tracking_disagreement_ball_20261113/`에 저장한다. 각 폴더의 `REPORT.md`, 요약 JSON 및 CSV에서 계산식과 결과를 확인한다. 전체 Tactical 프레임 축을 유지하고, 매핑 없는 프레임의 대응 거리와 Broadcast 상태는 미정으로 남긴다. 영상은 대응된 프레임만 재생하며 원본 번호는 타임라인 CSV로 확인한다.

기존 계산 캐시와 입력 해시를 재사용한다. 입력·설정이 다르면 다른 `--output` 폴더를 지정해야 한다. 파생 캐시를 원본처럼 취급하거나 다른 경기에서 그대로 재사용하지 않는다.

**원시 트래킹, 영상, 좌표 추출물, 생성 MP4/HTML/CSV/NPY/JSON은 로컬에만 보관하고 GitHub나 외부 서비스에 업로드하지 않는다.** `data/`는 기존 `.gitignore`로 제외된다. 저장소에는 실행 코드·합성 테스트·안내 문서만 포함한다.

## 합성 테스트

```bash
python -m pytest -q tests/test_tracking_disagreement.py tests/test_tracking_disagreement_ball.py tests/test_tracking_match_clock.py
```

추가로 기존 공유 함수와 함께 확인하려면 다음을 실행한다. 실제 경기 데이터는 필요하지 않다.

```bash
python -m pytest -q tests/test_tracking_disagreement.py tests/test_tracking_disagreement_ball.py tests/test_tracking_match_clock.py tests/test_tracking_utils.py tests/test_compare_tracking_player.py tests/test_final_tracking_alignment.py tests/test_all_movement_vectors.py
```
