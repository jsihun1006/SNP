# 프레임 정렬과 전체 non-X 이동 분석

이 문서는 커밋에 포함된 실행 경로를 설명한다. 프레임 정렬, 초기 X 라벨,
전체 이동 벡터 영상, 팀별 수렴·압축 진단을 순서대로 재생성할 수 있다.
기존 이벤트·속력·폭 실험 스크립트는 이 실행 경로에 필요하지 않다.

## 환경과 데이터

검증 환경은 macOS Apple Silicon, CPython 3.11.17이며 직접 의존성은
`requirements-tracking.txt`에 고정했다. 이미 사용하는 Conda 환경이 있다면 그대로 활성화한다.

```bash
conda activate mtah2027
python -m pip install -r requirements-tracking.txt
python -m unittest discover -s tests -v
```

새 환경이 필요하면 활성화 전에 `conda create -n mtah2027 python=3.11.17 pip`로 만든다.
합성 테스트는 실제 경기 데이터 없이 실행되며, 영상 생성에 필요한 H.264(avc1) 인코더도 검사한다.
코덱 지원은 OpenCV 빌드에 따라 다르므로 인코딩 테스트가 통과한 환경을 사용한다.

이미 받은 로컬 데이터 묶음의 ATD TXT/XML과 tactical Smart Tagging XML을 사용한다.
기본 폴더는 `data/tactical/`, `data/broadcast/`다. 다른 위치는 공식 `MTAH_DATA_DIR`로 지정한다.
영상·원시 데이터·추출 CSV/HTML/PNG/MP4는 GitHub나 클라우드 LLM에 업로드하지 않는다.
모든 결과는 Git에서 제외된 로컬 `data/derived/`에 저장된다.

## 실행 순서

저장소 루트에서 다음을 실행한다. 실제 데이터 읽기·연산은 로컬에서만 수행한다.

```bash
python starter/final_tracking_alignment.py
python starter/view_all_movement_vectors.py
python starter/view_team_motion_structure.py
```

| 파일 | 역할 |
|---|---|
| `final_tracking_alignment.py` | 공 궤적·선수 배치로 전·후반별 기준점을 찾고 프레임 대응·Half 태그·초기 X 라벨 생성 |
| `view_all_movement_vectors.py` | 이벤트 목록을 사용하지 않고 전체 non-X 프레임의 속도 벡터를 한 영상으로 렌더링 |
| `team_motion_metrics.py` | 평균 방향·수렴점·수렴 강도·관측 집합의 압축/확산 계산 |
| `view_team_motion_structure.py` | 팀 지표를 추가한 전체 영상·그래프·프레임별 수치 생성 |
| `tracking_utils.py` | ATD 입출력, 메타데이터 팀 매핑, 라벨 마스크, 좌표 속도와 공통 배열 함수 |
| `find_joint_tracking_match.py`, `find_layout_match.py` | 공 궤적 검색과 슬롯 번호를 공유하지 않는 팀별 위치 대응 |
| `compare_tracking_player.py` | 경기장 렌더링과 프레임 대응 플레이어 |

`align_half_starts.py`는 기존 선수 배치 기반 보조 도구와 합성 테스트를 위해 포함한다.
최종 프레임 대응은 위의 `final_tracking_alignment.py`로 생성한다.

## 프레임 정렬과 Half 태그

모든 `frame`/`frame0`는 0-based ATD 인덱스이며 원본 `atd_frame_id = frame + 1`이다.

- 전반 검색 기준은 `games.half_video_blocks()`의 공식 tactical 시작 프레임이다.
- 후반 검색 기준은 Smart Tagging에서 `Half=2nd Half`가 처음 나타나는 구간의 시작이다.
- 각 기준 이후 30초 공 궤적과 팀별 선수 배치를 방송 데이터에서 검색한다.
  팀별 일대일 위치 대응을 사용하며 같은 슬롯 번호를 같은 선수로 가정하지 않는다.
- 이어지는 30초의 공 궤적을 별도로 확인한다. 관측이나 일치 근거가 부족하면 최종 CSV 생성을 중단한다.
- 후반 태그 시작은 매핑된 방송 구간에서 공 x/y가 모두 유한값인 최초 프레임으로 옮긴다.
  tactical 태그 시작도 동일한 프레임 수만큼 옮겨 기존 시차와 공식 종료점을 유지한다.
- 이후에는 각 반의 공식 종료 프레임까지 같은 시차를 적용한다.
  전체 구간의 컷·리플레이 또는 broadcast MP4와 ATD의 대응을 검증한 결과는 아니다.

### 정렬 출력

`data/derived/sync/`에 다음을 만든다.

| 파일 | 내용 |
|---|---|
| `final_tracking_alignment.csv` | tactical 전체 행을 보존한 두 소스 프레임 대응과 Half 태그 |
| `final_tracking_tactical_labels.csv` | tactical 전체 프레임의 `frame,atd_frame_id,half_tag,label` |
| `final_tracking_broadcast_labels.csv` | broadcast 전체 프레임의 동일한 초기 라벨 표 |
| `final_tracking_alignment_report.json` | 기준점 검색·검증 오차, 태그 범위, 관측 여부와 프레임 수 |

대응 CSV의 `tactical_half_tag`, `broadcast_half_tag`는 `1st Half`/`2nd Half`다.
두 소스 전체 라벨 표에서는 태그 밖을 `X`로, 태그 안은 H/A/D 판정 전 빈칸으로 둔다.
이는 추론 준비용 파일이며 아직 완성된 점유권 제출물이 아니다.
대응 CSV에서 방송 번호가 없는 칸은 실제 방송 행을 나타내지 않으므로 방송 라벨도 빈칸이다.
미대응 broadcast 프레임의 X는 전체 방송 라벨 표에서 확인한다.

`half` 숫자 열은 기존 플레이어 호환을 위한 공식 평가 구간 번호다.
공식 후반 평가 시작과 실제 태그 시작 사이에는 `half=2`여도 명시적 Half 태그와 방송 번호가 없다.
이 과정은 공식 평가 구간을 변경하거나 점유권 정답을 수작업으로 만드는 작업이 아니다.
공개·비공개 점유권 GT는 읽지 않는다.

정렬만 시각적으로 비교하려면 다음을 실행한다.

```bash
python starter/compare_tracking_player.py --half-map data/derived/sync/final_tracking_alignment.csv
```

## 전체 non-X 이동 벡터 영상

`data/derived/full_movement_vectors/`에 H.264 영상 한 편, 로컬 HTML 플레이어,
원래 두 소스 프레임과 영상 프레임의 대응 CSV, 검증 보고서를 만든다.
HTML과 MP4를 같은 폴더에 두고 `full_movement_vector_player.html`을 브라우저에서 연다.
탐색 바, ±1프레임, 전·후반 시작 이동, 원래 tactical 프레임 번호로 탐색할 수 있다.

좌표 차분 속도는 원래 소스의 인접 프레임으로 계산한다.
두 프레임 모두 non-X·같은 Half이며 좌표가 유효할 때만 벡터를 유지한다.
하프타임이나 제외된 X 구간을 이어 붙인 재생 순번으로 미분하지 않는다.
결측은 보간하지 않고, 같은 슬롯의 재등장 지점에는 벡터를 만들지 않는다.
화살표는 속도(m/s)×표시 초이며 미래 위치 예측이 아니다.
슬롯 번호는 안정된 개인 ID가 아니다. 공은 위치만 표시한다.

## 팀별 이동 구조

`data/derived/team_motion_structure/`에 팀 평균 방향·수렴점·압축/확산 지표를 추가한
영상, 로컬 플레이어, 시계열 PNG, 프레임별 압축 CSV와 설정 보고서를 만든다.
`team_motion_structure_player.html`을 브라우저에서 연다.

계산식·고정 진단 기준·C/Q의 의미와 한계는 [TEAM_MOTION_STRUCTURE.md](TEAM_MOTION_STRUCTURE.md)에 설명했다.
C/Q는 검증된 확률이 아니며, 관측 부족이나 불안정한 수렴점은 빈값과 상태로 기록한다.
방송 지표는 카메라에 보이는 선수 부분집합의 값이고 원본 팀/슬롯 오류는 남아 있다.
