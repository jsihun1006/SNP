# Tryolabs Broadcast 유니폼·심판 분류 V2 실행 보고서

실행일: 2026-11-10. 요청한 기능을 별도 모듈로 구현하고 실제 broadcast의 동일750프레임에 실행했다.
30초/750프레임의 전후 비교 H.264 영상과 객체별 CSV/JSONL을 생성했으며 출력 전체 디코딩을 확인했다.
**심판 클래스의 후보 제외는 동작하지만, 색상만으로 실제 심판과 노란 조끼 인력을 완전히 구별하지는 못한다.**
UNKNOWN은 객체 관측의49.27%이며, 현재 결과로 분류 정확도나 점유 정확도가 향상됐다고 주장하지 않는다.

## 1. 추가한 코드와 보존 범위

기존 파일을 수정하지 않고 다음5개 구현·설정·테스트 파일과 이 보고서를 추가했다.

| 파일 | 역할 |
|---|---|
| [starter/tryolabs_uniform_v2.py](../starter/tryolabs_uniform_v2.py) | HSV 근거, UNKNOWN, 역할/팀 분리, 시간적 안정화, 후보 선택 |
| [starter/run_tryolabs_classification_v2.py](../starter/run_tryolabs_classification_v2.py) | 추적 재실행, 픽셀 수집·군집·범위 산출, 전후 영상·로그·지표·보존 검사 |
| [starter/configs/tryolabs_classification_v2_policy.json](../starter/configs/tryolabs_classification_v2_policy.json) | 품질·시간적 안정화·무라벨 산출 정책 및 근거 |
| [starter/configs/tryolabs_classification_v2_hsv.json](../starter/configs/tryolabs_classification_v2_hsv.json) | 실제 픽셀에서 산출해 고정한 기본 HSV 범위와 판정 기준 |
| [tests/test_tryolabs_classification_v2.py](../tests/test_tryolabs_classification_v2.py) | 합성 테스트19개 |

`starter/run_tryolabs_broadcast.py`, 기존 보고서, upstream 코드, 공/사람 모델 가중치,
기존 `preview/` 산출물, competition_judgement 및 MTAH_Relational_Starter 기존 파일을 보존했다.
검사 대상으로 기록한228개 파일의 SHA256이 작업 전후 같았다. 원본 영상의 size/mtime도 같았다.
Smart Tagging, tactical 좌표, homography, 정답이나 수작업 객체 라벨을 입력으로 사용하지 않았다.

새 비공개 파생물은 전부 `data/derived/tryolabs_broadcast/classification_v2/` 아래에 있다.
이 경로는 기존 `.gitignore`가 제외한다. 외부 업로드·공개 커밋은 하지 않았다.

## 2. 실제 데이터 흐름과 공 검출 보존

기존 실험의 `preview/detections.jsonl`에는 원래 모델이 생성한 사람·공 박스와 confidence가 들어 있다.
이 **동일 캐시**를 사용해 Norfair 운동 추정과 추적, 기존 HSV/InertiaClassifier/Match를 재실행했다.
기존 점유팀·BLUE 누적·LIGHT 누적·근접 카운터가750프레임 모두 일치했다.
기존 비교 화면의 왼쪽 패널은 기존 MP4의 오버레이 패널을 그대로 가져왔다.

새 분류는 이 추적 결과와 실제1920×1080 영상의 몸통·하체 픽셀을 사용한다.
공 모델을 다시 학습하거나 교체하지 않았고, 공 검출·공 추적·원본 main ball 선택을 변경하지 않았다.
새 실행기는 이번750프레임의 저장된 검출 결과를 재생하는 실험용 실행기이며, 임의의 새 영상에 대한
전체 detector CLI를 새로 구현한 것은 아니다.

- 파일10:00부터10초: source frame0 `15000..15249`
- 파일30:00부터10초: `45000..45249`
- 파일60:00부터10초: `90000..90249`

각 구간은250프레임이며 구간 간 상태를 초기화한다. source frame0은 경기 시계가 아닌 파일의0-based frame이다.
원본45px 거리 계산은1920×1080에서 하고, 화면 표시만960×540으로 줄인다.

## 3. HSV 설정을 결정한 근거

사람 박스의 가로15–85%, 세로15–50%를 몸통으로 사용했다. 이는 원본 torso crop과 같다.
새로 검출된 track 관측에서만 픽셀을 수집하고, 관측당 최대96픽셀로 제한해 큰 클로즈업의 면적 편향을 줄였다.
7,838개 객체 관측, 구간별 ID를 구분한70개 track에서 **752,402개 HSV 픽셀**을 얻었다.
동일 track의 여러 프레임별 분포는 `track_color_distributions.json`에 기록했다.
이70개 ID는70명의 실제 인물이라는 뜻이 아니다. 컷·가림·재초기화에 따라 ID가 나뉠 수 있다.

그중 고정 seed `20261110`으로120,000픽셀을 표본화해16개 색상 군집을 만들었다.
Hue의 원형 구조를 반영한 `(S*cos(H), S*sin(H), V)` 공간에서 OpenCV k-means를 실행했다.
사용자가 제시한 남색/자주색/연보라색/형광 노란색 가설은 **색상 군집의 의미를 정하는 넓은 seed 조건**에만 썼다.
최종 분류용 HSV 박스는 실제 군집의2–98% 분위수로 정했고, 그림자 성분에는 혼합을 줄이기 위해10–90%를 썼다.
군집 선택이 모호하거나 필요한 색상 anchor가 부족하면 임의의 범위를 만들지 않고 오류로 중단한다.

OpenCV 기준 H는0..179, S/V는0..255다. 다음 값은 눈으로 임의 지정한 최종 범위가 아니다.

| 성분 | 실제 군집 중앙값 HSV | 최종 lower → upper | 용도 |
|---|---|---|---|
| 남색 | `(122,143,123)` | `(110,92,19)` → `(134,196,179)` | TEAM_A |
| 자주색 | `(165,203,181)` | `(157,146,100)` → `(175,255,243)` | TEAM_A |
| 어두운 자주색 | `(147,143,111)` | `(137,101,77)` → `(157,192,146)` | TEAM_A 보조 |
| 밝은 연보라색 | `(137,77,239)` | `(125,51,203)` → `(151,118,255)` | TEAM_B |
| 그림자 연보라색 | `(140,70,176)` | `(127,36,140)` → `(156,105,198)` | TEAM_B 보조 |
| 형광 노란색 | `(31,242,255)` | `(30,182,182)` → `(42,255,255)` | REFEREE 상의 근거 |

`pixel_clusters.json`에는16개 군집의 분위수와 픽셀 수가 있고,
[픽셀 분포 그림](../data/derived/tryolabs_broadcast/classification_v2/hsv_pixel_distribution.png)에도 범위와 중앙값을 표시했다.

TEAM_A는 남색·자주색 마스크의 합집합 점수를 사용하되 **두 성분이 각각 몸통의2.5% 이상**이어야 한다.
단일 남색 상의나 검은색을 무조건 TEAM_A로 만들지 않는다.
TEAM_B는 밝은/그림자 연보라색 마스크의 합집합을 사용하며, 무채색 흰색 전체를 팀 색상으로 인정하지 않는다.
과노출로 거의 흰색이 된 유니폼은 따라서 UNKNOWN이 될 수 있다.

REFEREE는 형광 노란 상의와 어두운 하체를 함께 요구한다.
하체는 박스 세로60–95%이며, 어두움 기준 `V<=110`은 실제 어두운 픽셀 군집의90% 분위수다.
어두운 하체 비율은 최소10%를 요구한다. 자동 yellow anchor에는 잔디·다른 노란 인력이 섞일 수 있어
하체 비율의10% 분위수만으로는 기준이0에 가까워진다. 이를 허용하지 않기 위한 명시적 공학적 floor가10%다.
이 조건도 심판의 실제 역할을 증명하지는 않는다.

### 판정 점수·UNKNOWN 기준

점수는 **해당 색상 박스에 들어오는 몸통 픽셀 비율**이다. 정답 확률이 아니다.
최소15% 지지율과1위/2위 비율2배 조건의 자동 appearance anchor에서,
지지율과1·2위 점수 차이의10% 분위수를 각각 기본 판정 기준으로 선택했다.
TEAM_A1,707개, TEAM_B2,827개, yellow902개의 anchor가 사용됐으며 수작업 정답이 아니다.

| 클래스 | 최소 지지율 | 최소1·2위 점수 차이 |
|---|---:|---:|
| TEAM_A | 0.19937 | 0.18095 |
| TEAM_B | 0.19207 | 0.18968 |
| REFEREE | 0.25094 | 0.24688 |

기준 미달, 전 점수0, 작은/빈 torso, 가림, 시간적 충돌에는 UNKNOWN을 반환한다.
몸통 최소 크기는6×6, 최소64픽셀이고, 다른 fresh 박스가 몸통 면적의35%를 넘게 가리면 보류한다.
이 크기·가림·복수 색상 조건은 정답에 맞춘 HSV 수치가 아닌 공개된 공학적 보류 기준이다.
설정 JSON에서 조정할 수 있지만, 이번 실행에서 장면별로 수정하거나 수작업 정답에 맞춰 조정하지 않았다.

기본 설정과 실제 산출한 `hsv_config.json`의 내용이 완전히 같은 것도 검사했다.
**범위 산출과 시연이 같은750프레임을 사용하므로 독립 평가나 일반화 검증은 아니다.**

## 4. 팀·역할과 시간적 안정화

객체 로그에는 다음 정보를 따로 저장한다.

```python
{
    "track_id": 15,
    "team": None,                 # REFEREE는 팀 소속 없음
    "role": "REFEREE",
    "label": "REFEREE",
    "classification_confidence": None,
    "classification_source": "HSV_TEMPORAL",
    "eligible": False,
}
```

team은 TEAM_A/TEAM_B/UNKNOWN이며 심판의 team은None이다.
role은 PLAYER/REFEREE/GOALKEEPER/UNKNOWN을 수용할 구조이고,
현재 GOALKEEPER를 강제로 예측하지 않는다. 이번 두 TEAM은 appearance 이름이며 공식 H/A ID를 확정한 값은 아니다.

원본 InertiaClassifier의20관측 이력을 참고해 최근20개 fresh 관측과3표/65% 다수결을 사용한다.
단발성 반대 팀 분류는 기존 안정 상태의 기억을 바꾸지 않고 현재 출력은 UNKNOWN으로 보류한다.
현재 UNKNOWN을 과거 팀으로 채워 점유 후보에 넣지 않는다. stale track도 UNKNOWN이며 투표하지 않는다.
10프레임을 넘는 관측 공백이면 ID의 분류 기억을 초기화한다.
심판 appearance 근거는 다수결을 기다리지 않고 즉시 REFEREE로 표시하고 후보에서 제외한다.

## 5. 점유 후보와 원본 Match 연결

시각화·로그에는 모든 track을 남기되, Match에 넘기는 목록은
**fresh + eligible + role=PLAYER + team=TEAM_A/TEAM_B**만 포함한다.
role과 team을 모두 검사하므로 잘못된 eligible 플래그 하나만으로 심판이 후보에 들어가지 않는다.
선수 목록이 비면 `Ball(None)`을 전달해 원본 `min([])` 예외를 방지한다.

별도 V2 Match 인스턴스에서 원본 `Match.update`를 호출한다.45px,20프레임 기준과 upstream 코드는 그대로다.
V2의 historical 누적은 초기 소유팀을None으로 두며, 기존 BLUE/LIGHT 누적값과 합치지 않는다.
원본 main ball의 새 관측이 있고 공 후보가1개일 때만 V2 Match에 공 근거를 전달한다.
이는 입력 관측의 유효성 gate이며 공 detector/tracker의 변경은 아니다.
원본 Match의 누락 시 카운터 미초기화 등 동작도 보존하므로 historical 값은 여전히 현재 관측 근거와 다르다.

현재 프레임 근거는 별도 `select_candidate`에서 계산한다.
공 후보가 정확히1개이고, 제외 후 유효 선수가45px 이내에 있어야 `PROXIMITY_PENDING`이 된다.
조건을20프레임 연속 만족하면 `PROXIMITY_20_FRAME_EVIDENCE`이고,
근거가 끊기면 `INSUFFICIENT_EVIDENCE`와 구체적인 사유를 기록한다.
심판을 제외한 뒤 선수가45px보다 멀면 소유자를 만들지 않는다.
근접 evidence도 controlled possession 정답은 아니다.

## 6. 실제750프레임 비교 지표

객체 수는 **track-frame 관측 수**이며 고유 선수 수가 아니다.
원래 raw 사람 검출은8,126건, 추적 후 비교 대상 객체 관측은8,132건으로 양쪽이 같은 박스를 사용했다.
초기화 지연과 과거 track 잔존 때문에 raw 검출 수와 track 관측 수는 다를 수 있다.

| 지표 | 파일10분 구간 | 파일30분 구간 | 파일60분 구간 | 합계 |
|---|---:|---:|---:|---:|
| 객체 관측 | 2,880 | 398 | 4,854 | 8,132 |
| V1 BLUE | 905 | 0 | 1,651 | 2,556 |
| V1 LIGHT | 1,493 | 38 | 2,107 | 3,638 |
| V1 OTHER | 482 | 360 | 1,096 | 1,938 |
| V2 TEAM_A | 444 | 0 | 861 | 1,305 |
| V2 TEAM_B | 622 | 40 | 1,600 | 2,262 |
| V2 REFEREE | 7 | 177 | 374 | 558 |
| V2 UNKNOWN | 1,807 | 181 | 2,019 | 4,007 |
| V2 UNKNOWN 비율 | 62.74% | 45.48% | 41.59% | 49.27% |
| V1 동일track 분류 변경 | 28 | 2 | 50 | 80 |
| V2 raw 분류 변경 | 239 | 7 | 358 | 604 |
| V2 시간 안정화 후 분류 변경 | 152 | 7 | 342 | 501 |
| REFEREE 클래스 후보 제외 프레임 | 7 | 177 | 226 | 410 |
| 최근접 REFEREE 클래스 제외 후 보류 | 0 | 0 | 2 | 2 |
| V1 순간 후보 변경 | 10 | 0 | 27 | 37 |
| V2 순간 후보 변경 | 10 | 0 | 26 | 36 |
| V1/V2 순간 후보가 다른 프레임 | 15 | 0 | 4 | 19 |
| 공 근거 부족 프레임 | 170 | 240 | 117 | 527 |
| 공 근거 부족 비율 | 68.0% | 96.0% | 46.8% | 70.27% |

V1에는 UNKNOWN 클래스가 없으므로 OTHER 비율23.83%를 V2 UNKNOWN의 직접 대응 정답으로 취급하지 않는다.
분류 변경은 같은 구간·track ID의 인접 frame만 비교했고 UNKNOWN 진입/복귀를 포함했다.
V2의501회 변경은 모두 UNKNOWN 경계 변경이었다. TEAM_A↔TEAM_B/REFEREE 같은 알려진 클래스 사이의 직접 전환은0회다.
안정화 전 raw 분류의604회보다는 줄었지만, 기존80회보다 많다. 보류로 인한 화면 깜빡임은 남아 있다.

순간 후보 비교는 양쪽에 같은 fresh 관측·공 후보1개·45px 조건을 사용했다.
V1은 전체 사람 중 최근접을 먼저 고른 뒤 BLUE/LIGHT 소속 여부를 검사하고,
V2는 REFEREE/UNKNOWN 등을 제외한 뒤 최근접 유효 선수를 고른다.
ID↔없음 변경도 횟수에 포함하고 서로 다른 구간은 이어 붙이지 않았다.
이는 기존 누적 점유팀 전환 횟수나 정확도 지표가 아니다.

공 근거 부족은 **공 후보0개 또는2개 이상**인 frame이다. 공 후보가1개여도 실제 공이라고 보증하지 않는다.
공 결과를 변경하지 않았으므로 이 비율은 개선 전후 동일하다.
V2에서687프레임은 INSUFFICIENT_EVIDENCE,63프레임은 PROXIMITY_PENDING이었다.
20연속 fresh 근접 근거를 만족한 프레임은0개이고, 새 historical owner도 초기화되지 않았다.
이것은 실제 점유가 없다는 뜻이 아니다.

UNKNOWN 주요 사유는 색상 지지 부족2,816건, stale track294건, 시간 warmup285건,
노란 상의 대비 어두운 하체 부족248건, 가림218건, 색상 모호106건, 시간적 충돌39건, 작은 영역1건이다.

## 7. 비교 영상·로그와 재실행

- [전후 비교 영상](../data/derived/tryolabs_broadcast/classification_v2/comparison.mp4): 1920×930,H.264,25fps,750프레임.
- [심판 클로즈업 비교 화면](../data/derived/tryolabs_broadcast/classification_v2/segment_2_frame_45125.jpg).
- [객체별 JSONL](../data/derived/tryolabs_broadcast/classification_v2/objects.jsonl): raw/최종 분류, team/role, 성분별 점수,
  점수 차이, 하체 비율, 보류 사유, temporal votes, 과거 박스 age, 후보 허용 여부. confidence는 전부null이다.
- [객체별 CSV](../data/derived/tryolabs_broadcast/classification_v2/objects.csv): 주요 열을 평탄화한8,132행.
- [프레임별 CSV](../data/derived/tryolabs_broadcast/classification_v2/frames.csv):750행, V1/V2 후보·근거·각 누적값.
- [실행 요약](../data/derived/tryolabs_broadcast/classification_v2/summary.json),
  [보존·후보 검증](../data/derived/tryolabs_broadcast/classification_v2/verification.json),
  [테스트 로그](../data/derived/tryolabs_broadcast/classification_v2/tests.log).

동일 캐시를 사용한 분류·인코딩·출력 디코딩 검사는12.34초가 걸렸다. 새 detector 추론 속도가 아니다.
별도 픽셀 추출·범위 산출 단계의 시간도 포함하지 않는다.

기존 가상환경에서 테스트:

```bash
data/derived/tryolabs_broadcast/.venv/bin/python -m pytest -q tests/test_tryolabs_classification_v2.py
```

고정 설정과 추적 관측을 재사용해 새 폴더에 전후 영상을 생성:

```bash
data/derived/tryolabs_broadcast/.venv/bin/python starter/run_tryolabs_classification_v2.py \
  --run --output data/derived/tryolabs_broadcast/classification_v2/replay_new
```

원본 캐시부터 픽셀 추출·범위 산출까지 다시 실행하려면:

```bash
PYTHONHASHSEED=0 data/derived/tryolabs_broadcast/.venv/bin/python starter/run_tryolabs_classification_v2.py \
  --extract --cluster --calibrate --run \
  --output data/derived/tryolabs_broadcast/classification_v2/reproduction_new
```

이 조합은 새로 산출한 해당 폴더의 `hsv_config.json`을 사용한다.
`--config PATH`로 판정 설정을 명시할 수도 있다. 이미 있는 결과는 덮어쓰기를 거부한다.
보존 검사는 `--verify`로 실행하며 기준 snapshot은 작업 시작에 만든 `preservation_before.json`이다.
현재 최종 `verification.json`은 이미 있으므로 같은 경로에 다시 쓰지 않으며 재실행 시 새 출력 폴더를 사용한다.

## 8. 테스트 결과와 미해결 문제

**합성 테스트19개 통과(2.72초)**. 심판이 가장 가까워도 제외, 심판만 있는/빈 목록,
제외 후 선수가 멀 때 보류, HSV 전 점수0, 점수 모호함, 작은/가려진 몸통,
두 줄무늬 색상, 연보라색, 노란 상의+하체 조건, 골키퍼 색상 강제 분류 방지,
일시적 팀 분류 흔들림·UNKNOWN·심판 즉시 veto·긴 관측 공백,
기존 Match 기준45px/20과 별도 누적값 불변성을 검사했다.

실제 출력750프레임 전체 디코딩과25fps·해상도 검사를 통과했다.
객체8,132건에서 REFEREE/UNKNOWN/stale가 잘못 eligible로 설정된 경우0건,
현재 후보로 비선수가 선택된 frame0건, 기존 누적값750프레임 일치를 확인했다.
이는 프로그램의 필터 동작 검증이지 실제 팀·심판 정답 검증이 아니다.

남아 있는 문제:

1. **노란 조끼 인력의 REFEREE 오인 가능성.** 몸통 노랑+어두운 하체가 심판의 전용 특징은 아니다.
   로그에서 최근접 REFEREE를 제외한2프레임도 사이드라인 인력 appearance가 포함된 사례다.
   이를 실제 심판 인식 성공2건으로 표현하지 않는다. 색상으로 역할을 확정하지 못하는 근본 한계다.
2. **와이드 화면 심판 보류.** 하체 crop에 피부·잔디가 많이 들어가거나 다리가 가려지면 검은 하의 조건이 실패해
   실제 심판처럼 보이는 객체도 UNKNOWN이 될 수 있다. 조건을 장면별로 낮춰 보정하지 않았다.
3. **UNKNOWN과 표시 깜빡임 증가.** 그림자·과노출·줄무늬의 한 색상 소실·가림·초기화에서 보류가 많다.
   49.27% UNKNOWN과 기존보다 많은 분류 변경을 해결했다고 주장하지 않는다.
4. **골키퍼 소속 미확정.** 청록색 등 다른 복장은 UNKNOWN으로 남긴다. 실제 골키퍼라는 수작업 정답을 만들거나
   임의의 팀에 붙이지 않았다. 향후 role/team을 별도로 확장할 구조만 마련했다.
5. **공 누락·복수 후보·오검출이 그대로다.** 이번에는 모델·공 위치 추정 변경을 하지 않았다.
   기존 심판 손 주변 오검출도 새 분류만으로 제거되지 않는다. 현재 점유 제어를 단정할 수 없다.
6. **원본 tracking 한계.** ID switch와 컷을 완전히 해결하지 않았으며 원본 converter의 last_detection 박스를 사용한다.
   stale는 UNKNOWN으로 제외하지만, 매칭 자체가 잘못된 ID를 알고리즘이 모두 알아내지는 못한다.

향후에는 허용된 외부 경기의 라벨된 팀/심판/골키퍼 평가셋에서 역할·소속 분리와 UNKNOWN 기준을 검증한다.
상의·하의 고정 crop보다 pose/인물 segmentation과 경기장 안팎 문맥을 쓰면 노란 조끼 인력 혼동을 줄일 여지가 있다.
골키퍼에는 별도 복장 appearance와 역할 근거를 추가하되 팀 소속 근거가 부족하면 UNKNOWN을 유지한다.
공 검출은 외부 경기 기반 학습·검증, 작은 물체용 고해상도/tiled 추론, 후보 간 시간적 일관성 검증을 별도 실험으로 다룬다.
이 단계에서 이러한 모델을 새로 학습하거나 성능을 측정했다고 주장하지 않는다.
