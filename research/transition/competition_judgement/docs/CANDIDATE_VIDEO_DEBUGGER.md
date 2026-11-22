# v2-3 후보와 실제 영상의 로컬 대조

Broadcast non-X 중 트래킹이 전혀 없는 프레임과 tactical 영상을 나란히 비교하려면
[누락 프레임 비교 안내](../../starter/BROADCAST_MISSING_TRACKING.md)를 사용한다.
`python starter/view_broadcast_missing_tracking.py`는 누락 구간 앞뒤 2초를 포함한 좌우 비교 MP4를 만들고 실제 QuickTime Player로 연다.

Tactical 원본 영상에 동결된 v2-3 상태·판단 근거·분석 Anchor 영역을 표시한다. 원본과 기존 판정 파일은 수정하지 않는다. 서버를 켜거나 데이터를 업로드하지 않는다. Broadcast는 별도 시간축이라 이번 뷰어에 사용하지 않는다.

## 실행

프로젝트 루트에서:

```bash
conda activate mtah2027
python competition_judgement/view_candidate_video.py
```

기본적으로 알고리즘이 선택한 첫 Validation 후보의2초 전부터 정지 상태로 연다. 사용자가 본 장면으로 시작점을 선정하지 않았다. 전체199,319프레임을 탐색할 수 있고, 초기 위치는 `--start-frame 70000`처럼0-based 번호로 바꿀 수 있다. 이는 재생 위치 설정이며 라벨이나 모델 상수 변경이 아니다.

| 조작 | 기능 |
|---|---|
| 아래 Frame 바 | 임의 프레임으로 이동·일시정지 |
| Space | 재생/일시정지 |
| A / D | 이전/다음1프레임 |
| J / L | 이전/다음1초 |
| N / P | 선택한 상태의 다음/이전 연속 표시 구간2초 전으로 이동 |
| 0 | N/P 대상: PATTERN 또는 LOCAL 후보 |
| 1 / 2 | N/P 대상: PATTERN / LOCAL |
| 3 / 4 / 5 | N/P 대상: REVIEW / NO_DETECTED_SIGNAL / INSUFFICIENT |
| 6 | N/P 대상: 모든 상태 |
| V | 대표 Anchor만 / 대표+모든 후보 Anchor 표시 |
| R | 현재 표시 영역을 양쪽radius / 3m / 5m로 필터 |
| Q / Esc | 종료 |

구간 이동은 재생 편의 기능이다. 영구 cluster ID나 실제 경합 Episode를 연결한 결과가 아니다. 전후반·split·기록 경계는 이어 붙이지 않는다. Radius 필터는 현재 표시 대상으로 고른 Anchor를 제한하며, 숨겨진 다른 대표 Anchor를 새로 선정하지 않는다.

## 화면 해석

주황은 PATTERN, 분홍은 LOCAL, 청록은 REVIEW, 회색은 NO_DETECTED_SIGNAL이다. 상단에는 현재 frame·영상 시간·QC·split·대표 Anchor의radius/ID·근거 A/B/RF-short/RF-near·미래 정보 의존 범위를 표시한다. OpenCV 글꼴 때문에 화면 문구는 영어다.

선은 고정 Spatial Anchor `max(6m,현재 군집 지름+3m)`를 원래 tactical homography로 영상에 투영한 것이다. 선수 좌표(m)를105.3×68m로 나눠 정규화한 뒤 행렬을 적용한다. **공 위치, 실제 경합 위치 또는 선수 Bounding Box가 아니다.** 같은 Anchor 안에서 최근접 상대 쌍이 바뀔 수 있다. Homography가 유효하지 않거나 투영이 불안정하면 영역을 표시하지 않는다. 현재 공간 gate가 실패한 Anchor도 그리지 않는다.

같은 frame의3m/5m 원이 겹칠 수 있다. 서로 다른 사건 수로 세거나 하나의 확정 경합으로 합치지 않는다. 후보가 없는 REVIEW/NO 구간도 비교할 수 있지만 **NO는 실제 비경합 정답이 아니고, REVIEW/INSUFFICIENT는 관측 부족을 포함한다.**

## 허용되는 대조와 금지되는 정답 생성

공식 [RULES.md](../../RULES.md)는 제공 경기의 수작업 라벨을 학습·검증·튜닝 등 어떤 목적으로도 금지한다. 반면 영상을 보며 데이터를 이해하거나 정렬·투영·파이프라인을 디버깅하는 것은 허용한다.

이 뷰어는 후자의 용도다. 실제 경합/비경합을 체크하는 입력란, 평가용 수작업 라벨 저장, 예측 수정, 장면별 튜닝, 정확도/F1 계산 기능은 없다. 프레임 일치, 표시 출처, QC/누락으로 인한 보류 동작을 확인할 수 있으나 제공 경기 영상에서 정답을 만들어 경합 분류 성능을 측정했다고 주장할 수는 없다. 의미적 정확도 검증은 허용된 외부 경기의 라벨 또는 주최 측 평가 등 별도 근거가 필요하다.

## 로컬 MP4 미리보기와 검사

```bash
python competition_judgement/view_candidate_video.py --check
python competition_judgement/view_candidate_video.py --export-preview --seconds 12
```

기본 출력은 `data/derived/competition_judgement/candidate_video_debug/candidate_preview.mp4`와 같은 이름의JSON이다. 기존 파일이 있으면 덮어쓰기를 거부한다. 다른 미리보기를 만들려면 같은 로컬 폴더 아래 새 `--output` 경로를 지정한다. 영상은25fps H.264이며, 원래 tactical frame과 출력 frame의 관계는JSON의start/end에 있다. 이 영상도 비공개 경기 파생물이므로 외부 업로드나 Git에 포함하지 않는다.

`--video`와 `--homography`로 로컬 파일 경로를 지정할 수 있지만 tactical 시간축과 같은 파일이어야 한다.25fps·199,319프레임 및 CSV/homography 축을 검증한다. 같은 길이만으로 임의의 다른 영상이 정렬됐다고 보증하지는 않는다.

실제 로컬 검사는 첫/기본 시작/마지막 frame의 디코딩과 오버레이를 완료했고,12초/300프레임 미리보기를 생성했다. 기존522개를 포함한 **534개 테스트가 통과**했다(신규12개,33.12초). 합성 검증은 탐색 경계, 프레임 이동, 투영 방향·단위/실패 처리, 원본 불변성, H.264 인코딩과 출력 프레임 수를 포함한다. OpenCV의 COCOA GUI 지원도 확인했다. 이 실행에서는 GUI창의 직접 클릭 조작이나 실제 경합 정답 판별을 수행하지 않았다. 검증 결과는 출력 폴더의 `video_debug_verification.json`에 저장했다.
