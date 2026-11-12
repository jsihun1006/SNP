# Tryolabs Broadcast Uniform Classification V3

2026-11-12, 구현 버전 `3.0.1`. 실제 원본 영상에서 **1,000프레임, 12,513개 객체 관측**을 처리했다. 동일한 검출·Norfair Track을 V2/V3가 공유하며 검출기와 트래커 호출은 0회다. UNKNOWN은 46.50%에서 33.67%, 동일 Track의 연속 프레임 간 분류 변경은 746회에서 515회로 줄었다. 이것은 분류 범위와 안정성의 진단 결과다. 수동 정답이 없어 정확도 향상이나 오분류 감소는 입증하지 않았다.

전체 164,487프레임을 재처리하지 않았다. Mac CPU에서 실행했으며 Windows 네이티브 실행, GTX 1660 SUPER 벤치마크, GPU 추론 결과가 아니다.

추가한 코드는 다음과 같다. 기존 V1/V2 코드·설정·모델은 수정하지 않았다.

| 파일 | 기능 |
|---|---|
| [tryolabs_uniform_v3.py](../starter/tryolabs_uniform_v3.py) | 색상 특징, 팀·역할 분리, 시간 안정화 |
| [tryolabs_classification_v3_hsv.json](../starter/configs/tryolabs_classification_v3_hsv.json) | 실측 HSV 범위와 픽셀 출처·분위수 |
| [tryolabs_classification_v3_policy.json](../starter/configs/tryolabs_classification_v3_policy.json) | 품질·점수·역할·시간 판정 기준과 근거 |
| [calibrate_tryolabs_v3.py](../starter/calibrate_tryolabs_v3.py) | 기존 상의 픽셀 샘플에서 설정 재생성 |
| [tryolabs_v3_data.py](../starter/tryolabs_v3_data.py) | 기존 캐시 읽기, SQLite 복사본, 원본 해시 보존 검사 |
| [run_tryolabs_classification_v3.py](../starter/run_tryolabs_classification_v3.py) | 같은 입력의 V2/V3 비교 실행·영상·로그·그래프 |
| [evaluate_tryolabs_v3_external.py](../starter/evaluate_tryolabs_v3_external.py) | 다른 경기의 별도 수동 정답이 있을 때만 평가 |
| [test_tryolabs_classification_v3.py](../tests/test_tryolabs_classification_v3.py) | 신규 24개 테스트 |

공식 유니폼 정보와 영상의 측정값은 구분했다. 바르셀로나의 2024/25 홈 유니폼은 파랑과 적갈색을 양분한 디자인이다. 이전의 ‘줄무늬’ 설명을 그대로 가정하지 않았다. 아틀레틱의 서드 유니폼은 lavender blue와 pink, 검정 라운드 칼라가 특징이다. 공식 경기 페이지에는 주심이 Jesús Gil Manzano로 기록돼 있다. 이 정보는 색상 가설과 경기 문맥의 근거이며 HSV 수치나 영상 속 인물의 신원 정답은 아니다. [바르셀로나 공식 디자인 설명](https://www.fcbarcelona.com/en/club/news/4058726/the-story-of-halved-fc-barcelona-jerseys), [아틀레틱 공식 발표](https://www.athletic-club.eus/en/news/2024/08/22/where-the-story-begins-third-kit-2024-25/), [공식 경기 페이지](https://www.fcbarcelona.com/en/matches/116220/fc-barcelona-athletic-club-la-liga-2024-2025).

HSV 설정은 기존 750프레임의 상의 픽셀 752,402개에서 계산했다. 기존 V2 색상 지지율 조건으로 남색+적갈색, 연보라, 형광색의 자동 외관 집단을 골랐다. 각각 객체 관측 1,432개, 2,154개, 776개다. V2의 최종 레이블이나 사람의 팀 정답을 사용하지 않았다. 집단에 배경·피부·다른 객체가 섞일 수 있고 기존 범위로 집단을 선택한 편향도 남는다. 독립된 학습·평가 자료로 해석할 수 없다.

OpenCV HSV: H=0–179, S/V=0–255. 아래 양 끝값은 실제 픽셀의 분위수에서 내림/올림했다. 색상 후보를 고르는 넓은 H/S/V 조건 자체는 공학적 가설이며 JSON과 재생성 코드에 공개했다.

| 성분 | H | S | V | 선택된 픽셀 수 | 분위수 |
|---|---|---|---|---:|---|
| Navy | 111–135 | 82–194 | 61–185 | 21,778 | 1–99% |
| Burgundy, 밝은 쪽 | 155–178 | 131–255 | 88–245 | 29,638 | 1–99% |
| Burgundy, 그림자 쪽 | 137–159 | 106–216 | 76–153 | 9,844 | 10–90% |
| Lavender | 126–159 | 31–120 | 133–255 | 109,412 | 1–99% |
| Pale lavender | 132–159 | 30–59 | 211–255 | 41,075 | 5–99%; S 하한은 25% |
| Pink, 보조 | 152–178 | 33–102 | 122–250 | 16,313 | 5–95% |
| Fluorescent yellow | 29–43 | 175–255 | 175–255 | 40,433 | 1–99% |

성분별 선택 집합은 중복될 수 있다. pale lavender의 S 하한을 무조건 낮추지 않고 해당 집합의 25분위수로 정해 무채색 흰색 전체를 포함하지 않았다. 새 navy의 V 하한 61은 기존 19보다 높아 아주 어두운 유니폼을 놓칠 수 있다. 모든 조명에서 기존 범위보다 우수하다고 주장할 수 없다.

설정 근거는 [calibration_diagnostics.json](../data/derived/tryolabs_broadcast/classification_v3/calibration/calibration_diagnostics.json)과 HSV JSON의 `provenance`에 있다. 원본 픽셀 NPZ SHA256은 `0c0802716727dbce6ff13c96fcb11ff96e3ca86759c20f9e60b7daef6f45c2f7`이다.

기존 V2에는 다음과 같은 구조적 제약이 있었다.

- 밝은 lilac은 V≥203, 어두운 lilac은 V≤198이어서 V=199–202에 공백이 있다. H=125–156과 해당 명도대에 있는 후보 픽셀은 전체 샘플에서 3,038개다. S 조건까지 모두 충족하는 팀 정답 픽셀 수를 뜻하지 않는다.
- H=125–160, S=8–50, V≥203인 저채도 후보 중 기존 lilac이 덮지 못한 픽셀은 53,774개다. 밝은 배경도 포함될 수 있다.
- 기존 burgundy와 lilac 마스크에 동시에 들어가는 픽셀은 99개였다. 단순한 HSV 범위의 교집합만으로 모든 혼동을 설명할 수 없다.
- A는 매 관측에서 navy와 burgundy가 각각 2.5% 이상이어야 했다. V2 자동 Track 집단의 A 23개에서 색상 존재 비율의 Track 간 중앙값은 navy 81.77%, burgundy 97.87%, 동시 존재 80.00%였다. 방향·가림·조명에 따라 한쪽이 사라지는 문제를 시사하지만, 사람이 확인한 선수 집단은 아니다.
- V2에는 별도 pink 특징이 없었다. V3 B 자동 Track 집단의 pink 지지율 중앙값은 약 5.16%였다. B의 핵심 증거는 계속 lavender다.

V3는 상의 ROI를 기존과 같은 가로 15–85%, 세로 15–50%로 두고 navy/burgundy/lavender/pink/yellow 비율, HSV 10·50·90분위수, S/V 분포, 가림, ROI 크기, 선명도를 저장한다. 하의 48–72%, 기존 아래쪽 60–95%, 양말 80–98%, 칼라 8–20%도 별도로 측정한다. 검정 칼라와 양말은 작은 ROI·가림 때문에 필수 조건으로 쓰지 않았다.

A의 강한 현재 근거는 navy와 독립적인 burgundy의 조합이다. burgundy와 lavender/pink가 같은 픽셀을 공유하면 그 부분을 독립적인 A 증거로 중복 계산하지 않는다. B도 burgundy와 공유하는 lavender 픽셀을 핵심 lavender 지지에서 제외한다. Pink는 가중치 0.25의 보조 특징이며 단독으로 B를 만들 수 없다. 어두운 무채색, 밝은 흰색, 전 성분 0점은 UNKNOWN이다.

품질 기준은 V2의 상의 최소 폭/높이 6px, 최소 64픽셀, 가림 최대 35%, 보조 색상 최소 2.5%를 유지했다. 지지율/점수 차이도 V2 값(A 약 0.1994/0.1810, B 0.1921/0.1897, REFEREE 0.2509/0.2469)을 유지했다. 점수는 보정된 확률이 아니며 `classification_confidence=None`이다.

시간 정책은 20관측 창·최소 3표·65%·10프레임 공백 초기화를 기준선으로 쓴다. 강한 A 관측이 이력을 만든 뒤 현재 한 색상만 충분하고 충돌하는 B/REF 근거가 없을 때, 마지막 강한 관측으로부터 최대 10프레임만 보조한다. 품질 실패나 무증거를 과거 레이블로 채우지 않는다. stale Track은 UNKNOWN이고 투표하지 않는다. 현재 관측·안정화·최종 출력은 `observed_label`, `stabilized_label`, `final_label`로 구분한다. 모든 시간 상태와 근접 근거는 구간/scene 변경 또는 프레임 불연속에서 초기화한다. 원본 Track ID는 바꾸지 않는다.

REFEREE는 형광색 상의 + 어두운 하의 근거 + 주변 잔디색 + 작은 화면 가장자리 객체가 아님 + 3회 연속 현재 관측을 요구한다. 잔디색 H=25–85/S≥40, 주변 비율 15%, 이미지 가장자리 3%, 작은 객체 높이 25%는 넓은 문맥 방어 조건이다. 실측으로 보정된 주심 식별 확률이 아니다. 형광 조끼를 입은 스태프나 노란 골키퍼를 의미적으로 완벽히 구분하는 기능은 없다.

첫 실행에서 30분 클로즈업의 REFEREE 관측이 V2 177개→V3 18개로 줄었다. 원인은 화면 하단에서 잘린 큰 박스에 48–72% 고정 하의 ROI를 적용하면 상의가 들어가는 것이었다. 자동 진단상 이 구간의 `YELLOW_WITHOUT_DARK_SHORTS` 215개 중 154개는 큰 하단 잘림 박스였고 기존 V2의 60–95% ROI에 어두운 근거가 있었다. 최종 3.0.1에서는 **높이≥화면 25%이고 하단 3% 이내에서 잘린 경우에만** 기존 아래쪽 ROI를 보조한다. 어두운 비율 기준 10%는 그대로다. 이 보완 후 30분 구간은 171개가 됐다. 해당 조건을 별도 테스트했고 보조 사용 여부를 로그에 남겼다. 아래쪽 검정 영역이 실제 반바지라는 해부학적 검증까지 수행한 것은 아니다.

팀과 역할은 분리한다. `team=TEAM_A/TEAM_B`여도 잔디·위치 근거가 약하면 `role=UNKNOWN, eligible=False`다. 이번 실행에 434개가 있었다. `GOALKEEPER/STAFF/SUBSTITUTE`는 구조와 가설 목록만 제공하며 확인되지 않은 객체를 강제로 배정하지 않았다. `REFEREE`는 `team=None`이고 항상 제외한다. 초기 2회 REF 보류도 UNKNOWN이므로 점유 후보가 되지 않는다. `PLAYER`이면서 팀을 알고 fresh이며 eligible인 경우만 기존 `select_candidate()`에 전달한다. 45px과 같은 팀의 20연속 프레임 조건은 유지했다. 원래의 점유 누적 카운터를 수정하거나 V3 근거와 합치지 않았다.

실제 검증한 파일 시간과 0 기반 프레임은 다음과 같다. 750프레임은 설정 작성에 사용한 구간이다. 추가 250프레임은 이번 HSV 범위 산출에는 사용하지 않은 다른 시간대이지만, 같은 경기이고 정답도 없어서 독립적인 정확도 평가가 아니다.

| 구간 | 프레임, 양 끝 포함 | 프레임 수 | 객체 관측 수 |
|---|---|---:|---:|
| 10:00–10:10 | 15000–15249 | 250 | 2,880 |
| 30:00–30:10 | 45000–45249 | 250 | 398 |
| 60:00–60:10 | 90000–90249 | 250 | 4,854 |
| 추가 05:20–05:30 | 8000–8249 | 250 | 4,381 |

원본 `data/broadcast/BAR-ATH_broadcast-001.mp4`는 1920×1080/25fps다. 기존 `classification_v2`의 `track_observations.jsonl`과 `preview/detections.jsonl`, 추가 구간은 `full_match_v2/main`의 기존 tracking/object 청크와 검출 SQLite를 읽었다. 객체 관측 수에는 같은 사람의 반복 관측과 stale Track이 포함된다. 고유 선수 수가 아니다.

최종 결과는 [classification_v3/validated](../data/derived/tryolabs_broadcast/classification_v3/validated/)다. `preview/final/release` 등 중간 실험도 삭제하거나 덮어쓰지 않았으며 이 보고서 수치는 **validated**만 사용한다. 영상은 네 구간을 위 순서로 연결한 40초이며 경기 전체가 아니다.

| 분류 | V2 관측 수 / 비율 | V3 관측 수 / 비율 |
|---|---:|---:|
| TEAM_A | 2,555 / 20.42% | 3,205 / 25.61% |
| TEAM_B | 3,450 / 27.57% | 4,317 / 34.50% |
| REFEREE | 689 / 5.51% | 778 / 6.22% |
| UNKNOWN | 5,819 / 46.50% | 4,213 / 33.67% |

기존 750프레임만 보면 UNKNOWN은 4,007/8,132(49.27%)→2,901/8,132(35.67%)다. 추가 250프레임은 1,812/4,381(41.36%)→1,312/4,381(29.95%)다. 알려진 클래스 coverage는 전체에서 53.50%→66.33%, 점유 후보 자격 관측 비율은 47.99%→56.65%다. coverage 증가가 오분류 증가를 동반했는지는 정답 없이 판단할 수 없다.

| 진단 | V2 | V3 |
|---|---:|---:|
| 동일 Track의 인접 프레임 분류 변경 | 746 | 515 |
| 직접 TEAM_A↔TEAM_B 변경 | 0 | 0 |
| REFEREE→UNKNOWN 변경 | 37 | 22 |
| REFEREE를 한 개 이상 제외한 프레임 | 541 | 557 |
| INSUFFICIENT_EVIDENCE | 866 | 826 |
| PROXIMITY_PENDING | 120 | 160 |
| PROXIMITY_20_FRAME_EVIDENCE | 14 | 14 |

V3의 시간 보조 A는 263관측이다. V2/V3의 점유 후보 ID가 다른 프레임은 40개다. 20프레임 근거는 양쪽 모두 14프레임이며 실제 소유·제어의 정답이나 점유 정확도 개선을 뜻하지 않는다. REFEREE 제외 프레임은 ‘해당 역할 객체가 한 개 이상 있어 제외된 프레임’이며 실제로 심판이 공에 가장 가까웠던 횟수와 다르다.

다음은 **모델 간 불일치 표**다. V2는 정답이 아니므로 Confusion Matrix라고 부르지 않는다.

| V2 → V3 | A | B | REF | UNKNOWN |
|---|---:|---:|---:|---:|
| A | 2,555 | 0 | 0 | 0 |
| B | 0 | 3,448 | 0 | 2 |
| REF | 0 | 0 | 597 | 92 |
| UNKNOWN | 650 | 869 | 181 | 4,119 |

UNKNOWN 4,213개의 주된 사유는 색상 부족 2,577, 시간 초기화 346, stale 329, A 두 번째 색상 부족 310, 가림 224, 어두운 하의 부족 110, 색상 점수 혼동 105, 화면 가장자리 노란색 101, REF 연속 관측 대기 65, 현재/과거 충돌 45, 작은 ROI 1개다. 서로 배타적인 최종 보류 사유이며 분류 정확도의 오류 유형 정답이 아니다.

심판 외관 관측 778개는 구간+Track 기준 8개 ID에서 나왔다. 같은 주심이 여러 ID로 나뉘거나 다른 사람이 포함될 수 있다. 주심 신원은 검증하지 않았다. REFEREE인데 eligible인 관측은 0개다. 작은 가장자리 노란색 101개를 보류했고, 큰 하단 잘림의 아래쪽 어두운 ROI 보조는 REF 관측 171개에 사용됐다.

| 구간 | REF 관측 V2→V3 | REF 연속 출력 최대 길이 V2→V3 | 설명 |
|---|---:|---:|---|
| 05:20 | 131→138 | 38→55프레임 | 다른 시간대의 진단 |
| 10:00 | 7→3 | 2→2프레임 | 현재 근거가 불안정해 여전히 실패/보류가 많음 |
| 30:00 | 177→171 | 158→156프레임 | 잘린 큰 박스의 ROI 보완 필요 |
| 60:00 | 374→466 | 74→180프레임 | 여러 Track ID; 한 명이라고 단정 불가 |

3회 연속 현재 REF 근거 후 출력하더라도 출력 연속 길이는 1–2프레임일 수 있다. 바로 다음 관측이 가림·stale·약한 색상이면 다시 UNKNOWN으로 보류하기 때문이다. 전체적인 전환 감소와 달리 30분 구간 전환은 7→8회다. 모든 구간에서 개선된 것은 아니다.

실제 산출물은 다음 링크로 확인할 수 있다. 대용량 영상·ROI·원본 캐시는 로컬 `data/`에만 두며 Git에 포함하지 않는다.

- [V2/V3 비교 영상](../data/derived/tryolabs_broadcast/classification_v3/validated/comparison_v2_v3.mp4): H.264, 1920×800, 25fps. 생성 후 1,000프레임 전체 디코딩 확인. 팀/ID/REF 제외/동일 공 후보/현재 근접 근거 표시. 음성 없음.
- [상의 HSV 분포](../data/derived/tryolabs_broadcast/classification_v3/validated/hsv_distributions.png): V3 자동 분류 집단별 분포. 정답 집단의 분포가 아니며 배경 혼입이 보인다.
- [Burgundy/Lavender 동시 지지율](../data/derived/tryolabs_broadcast/classification_v3/validated/burgundy_lavender_overlap.png): 성분 공존 분포. 동일 픽셀의 범위 교집합과는 구분한다.
- [첫 Contact Sheet](../data/derived/tryolabs_broadcast/classification_v3/validated/contact_sheet_01.jpg), [Track별 이미지](../data/derived/tryolabs_broadcast/classification_v3/validated/contact_sheets/): 98개 구간+Track의 처음/중간/마지막 전체 프레임과 상의 ROI, 17개 합본 이미지.
- [A 시간 보조 사례](../data/derived/tryolabs_broadcast/classification_v3/validated/diagnostics/HISTORY_SUPPORTED_A.jpg), [연보라 범위 변경 사례](../data/derived/tryolabs_broadcast/classification_v3/validated/diagnostics/V2_UNKNOWN_V3_KNOWN.jpg), [REF 외관 사례](../data/derived/tryolabs_broadcast/classification_v3/validated/diagnostics/REFEREE_APPEARANCE.jpg).
- [색상 혼동 보류](../data/derived/tryolabs_broadcast/classification_v3/validated/diagnostics/AMBIGUOUS_TEAM_COLORS.jpg), [가림](../data/derived/tryolabs_broadcast/classification_v3/validated/diagnostics/TORSO_OCCLUDED.jpg), [가장자리 노란색](../data/derived/tryolabs_broadcast/classification_v3/validated/diagnostics/SIDELINE_OR_CLIPPED_YELLOW.jpg), [하의 근거 부족](../data/derived/tryolabs_broadcast/classification_v3/validated/diagnostics/YELLOW_WITHOUT_DARK_SHORTS.jpg). 자동 선택한 알고리즘 진단 이미지이며 사람이 확인한 오분류 정답이 아니다.
- [objects.jsonl](../data/derived/tryolabs_broadcast/classification_v3/validated/objects.jsonl), [objects.csv](../data/derived/tryolabs_broadcast/classification_v3/validated/objects.csv), [frames.csv](../data/derived/tryolabs_broadcast/classification_v3/validated/frames.csv), [Track 색상 안정성](../data/derived/tryolabs_broadcast/classification_v3/validated/track_color_stability.csv).
- [summary.json](../data/derived/tryolabs_broadcast/classification_v3/validated/summary.json), [구간별 추가 진단](../data/derived/tryolabs_broadcast/classification_v3/validated/supplementary_diagnostics.json), [manifest](../data/derived/tryolabs_broadcast/classification_v3/validated/manifest.json), [보존 검사](../data/derived/tryolabs_broadcast/classification_v3/validated/verification.json), [재현 검사](../data/derived/tryolabs_broadcast/classification_v3/validated/reproducibility.json), [테스트 기록](../data/derived/tryolabs_broadcast/classification_v3/validated/test_results.json).

보존 검사는 기존 코드·설정·upstream·가중치·750프레임 결과·전체 실행 캐시 450개 파일의 SHA256이 변경되지 않았음을 확인했다. 원본 13,962,360,320바이트 영상은 size/mtime이 유지됐다. 영상 전체 SHA256까지 다시 계산한 검사는 아니다. V3의 SQLite 접근은 DB/WAL/SHM을 새 출력으로 복사한 뒤 복사본에만 연결한다. 소스 변경이 감지되면 중단한다. 캐시 작성 프로세스를 종료하거나 일시 중지하고 사용한다.

기존 750프레임의 frozen V2 레이블은 8,132/8,132개가 재실행 V2와 일치했다. 추가 구간은 4,286/4,381개가 일치했다. 추가 구간을 8000프레임에서 새로 시작하면서 이전 전체 실행의 분류 이력이 없으므로 95개 차이가 난다. V2 코드나 기존 결과를 고친 것이 아니다. 공정 비교를 위해 V2/V3 양쪽에 같은 구간 초기화를 적용했고 frozen V2도 로그에 별도로 보존했다.

최종 설정을 고정한 별도 프로세스 재생에서 `input_frames.jsonl`, `objects.jsonl`, `objects.csv`, `frames.csv`, `track_color_stability.csv`가 바이트 단위로 일치했다. 요약은 영상 미생성 재생의 `decoded_output_frames` 필드만 제외하면 같다. 입력 SHA256은 `e38379428f443ee7e875f546453767f832ca592bbd15c98ae1d175016b26304a`다. 설정 SHA256은 `e8d82626da017dc445ecb1ac3e2f8ebe053cb5105e1a7ac617c1019d8bb544e6`, 정책 SHA256은 `2f274e597a227c20190eff610dff015280c3b49ed517e05c0047eb3e57103a5a`이며 코드 해시는 manifest에 있다.

실행한 회귀 테스트는 **70개 통과, 실패 0개, 3.88초**다. 신규 24개와 기존 V2·전체 처리·Windows 지원 46개다. 색상/혼동/작은 ROI/가림/심판 제외/밝은 흰색·검정·분홍 단독 보류/골키퍼 색상 보류/팀·역할 분리/현재 충돌·무증거·이력 만료/잘린 박스/누락 입력/SQLite 원본 보존/동일 캐시 재생/외부 정답 가드를 검사했다. 스태프·골키퍼의 모든 실제 복장에 대한 정확도를 검증한 테스트는 아니다.

```text
python -m pytest -q tests/test_tryolabs_classification_v2.py tests/test_tryolabs_full_match_v2.py tests/test_tryolabs_windows.py tests/test_tryolabs_classification_v3.py
...................................................................... [100%]
70 passed in 3.88s
```

수동 정답 평가 자료는 만들지 않았다. [RULES.md](../RULES.md)의 제공 경기 수동 라벨 금지 규칙이 해당한다: “No hand labels on the provided game, in any amount and for any purpose”. `diagnostic_review.csv`는 자동 예측·사유·이미지 위치를 보는 목록이며 정답 열이 없다. `external_evaluation_template.csv`는 **다른 경기**에서 허용되는 수동 정답을 준비하기 위한 빈 헤더다. `evaluate_tryolabs_v3_external.py`는 외부 경기/사람 라벨 선언만 받아 P/R/F1/Macro F1/Confusion Matrix를 계산하고 빈 자료·자동 의사 정답·제공 경기 식별자는 거부한다. 식별자 검사는 허위 출처를 검증하는 장치가 아니므로 실제 외부 경기 출처를 별도로 확인해야 한다. 이번 경기의 실제 성능 수치는 계산하지 않았다.

Windows CMD에서 프로젝트 루트는 `C:\Users\kjkim\mtah_gpu\SNP`다. 그 아래에 `more-than-a-hack-2027` 폴더를 추가로 만들 필요는 없다. 기존 가상환경과 Windows requirements를 사용한다. V3는 torch를 import하지 않고 새 GPU 모델을 로드하지 않으므로 `--device`나 `--model-residency` 옵션을 받지 않는다. 기존 `both/single` GPU 상주 정책은 변경하지 않았다.

```text
SNP\
  starter\run_tryolabs_classification_v3.py
  starter\configs\tryolabs_classification_v3_*.json
  data\broadcast\BAR-ATH_broadcast-001.mp4
  data\derived\tryolabs_broadcast\full_match_v2\windows_gtx1660\
    run.sqlite
    chunks\...\chunk.json
    chunks\...\tracking.jsonl.gz
    chunks\...\objects.jsonl.gz
  data\derived\tryolabs_broadcast\classification_v3\  (신규 결과)
```

`windows_gtx1660`은 예시이며 실제 V2 실행 폴더명으로 바꾼다. `run.sqlite`만으로는 Track이 없어서 부족하다. 요청한 시간대의 추적·분류 청크까지 완료돼 있어야 한다. V3는 누락 캐시를 발견하면 오류로 중단하며 YOLO를 자동 실행하지 않는다. 기존 원본 영상도 위 경로에 있어야 한다.

```bat
cd /d C:\Users\kjkim\mtah_gpu\SNP
call C:\Users\kjkim\mtah_gpu\.venv\Scripts\activate.bat
python -m pip install -r requirements-tryolabs-windows.txt
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=2
python -m pytest -q tests\test_tryolabs_classification_v2.py tests\test_tryolabs_full_match_v2.py tests\test_tryolabs_windows.py tests\test_tryolabs_classification_v3.py
python starter\run_tryolabs_classification_v3.py --source full --cache data\derived\tryolabs_broadcast\full_match_v2\windows_gtx1660 --starts 600 1800 3600 --seconds 10 --output data\derived\tryolabs_broadcast\classification_v3\windows_samples_01
```

위 명령은 750프레임 **후처리**이며 새 GPU 검출 벤치마크가 아니다. requirements는 기존 CUDA torch를 설치/교체하지 않는다. 처음 실행한 뒤 재검증할 때는 입력을 동결한 다음과 같은 명령을 쓴다. 출력 이름은 항상 새 것으로 지정한다. 이미 존재하면 덮어쓰지 않고 중단한다.

```bat
python starter\run_tryolabs_classification_v3.py --source prepared --cache data\derived\tryolabs_broadcast\classification_v3\windows_samples_01\input_frames.jsonl --output data\derived\tryolabs_broadcast\classification_v3\windows_replay_01 --no-video --no-visuals
fc /b data\derived\tryolabs_broadcast\classification_v3\windows_samples_01\objects.jsonl data\derived\tryolabs_broadcast\classification_v3\windows_replay_01\objects.jsonl
fc /b data\derived\tryolabs_broadcast\classification_v3\windows_samples_01\frames.csv data\derived\tryolabs_broadcast\classification_v3\windows_replay_01\frames.csv
```

Windows에서 초기 GPU 벤치마크의 `segment_1..3` 폴더만 있는 경우 각 segment의 `run.sqlite/chunks`를 `--cache`로 지정하고 각각 `--starts 600`, `1800`, `3600`으로 나눠 실행한다. 완료되지 않은 전체 캐시의 빈 시간대를 V3로 채울 수는 없다.

로컬 1,000프레임 입력을 캐시에서 다시 구성하거나 HSV 설정 산출을 재현하는 명령은 다음과 같다. 생성될 폴더명은 예시이며 새 폴더만 사용한다. 아래 Python 경로는 Mac에서 실제 사용한 기존 Python 3.11 가상환경이다.

```sh
data/derived/tryolabs_broadcast/.venv/bin/python starter/calibrate_tryolabs_v3.py --output data/derived/tryolabs_broadcast/classification_v3/calibration_rebuild_01
data/derived/tryolabs_broadcast/.venv/bin/python starter/run_tryolabs_classification_v3.py --source legacy --extra-cache data/derived/tryolabs_broadcast/full_match_v2/main --extra-start 8000 --extra-frames 250 --output data/derived/tryolabs_broadcast/classification_v3/local_rebuild_01
```

최종 1,000프레임 영상에 사용한 명령은 다음이다. 동결 입력은 앞선 실제 캐시 읽기에서 얻었으며 검출·Track·frozen V2를 함께 저장했다.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 data/derived/tryolabs_broadcast/.venv/bin/python starter/run_tryolabs_classification_v3.py --source prepared --cache data/derived/tryolabs_broadcast/classification_v3/release/input_frames.jsonl --output data/derived/tryolabs_broadcast/classification_v3/validated
```

오류와 미해결 사항도 남아 있다. 고정 사각형 ROI는 몸의 실제 부위와 일치하지 않을 수 있다. 큰 박스 하단 잘림 보완도 그 제한의 일부만 다룬다. 잔디색·화면 가장자리는 실제 피치/벤치 위치를 보장하지 않는다. 특히 유니폼을 입은 스태프·교체 선수, 노란 골키퍼는 현재 특징만으로 확실히 구분할 수 없다. 가까운 화면에서 경기 진행 중인지, 리플레이인지도 판단하지 않는다. 기존 추적 ID 오류를 V3가 복구하지 않는다. 10분 구간의 심판 외관 근거는 여전히 매우 짧다.

검증 도중 처음 발생한 Matplotlib 사용자 캐시 쓰기 경고는 OS 임시 디렉터리 사용으로 해결했다. 최종 영상 인코딩·프레임 디코딩·원본 보존·재현 검사에서 미처리 프레임은 없었다. 요청한 이번 1,000프레임 밖의 성능과 전체 처리 완료 여부를 대신 검증한 것은 아니다.

마지막 질문에 대한 답은 다음과 같다.

1. **V2 HSV 범위에 잘못 설정된 부분이 있는가?** 명도 공백, 저채도 연보라 누락, pink 부재는 구현·픽셀 진단에서 확인했다. 정확한 팀 정답에 대한 ‘잘못된 임계값’이라고 통계적으로 입증한 것은 아니다. 복합색 강제와 잘린 ROI도 주요 원인이다.
2. **두 팀을 혼동하는 구체적 원인은?** Burgundy/pink/lavender의 일부 H/S/V 공유, 조명에 따른 S/V 변화, 반으로 나뉜 A 상의에서 한 색상만 보이는 방향, 가림·배경 혼입이다. 실제 직접 A↔B 전환은 이번 표본 양쪽 모두 0회이므로 혼동의 대부분을 팀 간 오분류로 단정하지 않는다. 관측상 큰 문제는 UNKNOWN이다.
3. **심판의 가장 신뢰할 특징은?** 이 구현에서 가장 강한 조합은 형광 상의와 어두운 하의, 현재 영상의 위치 문맥 및 연속 관측이다. 노란색 단독이나 공식 주심 이름은 인물 식별 증거가 아니다. 어떤 조합이 실제로 가장 정확한지는 외부 정답 평가가 필요하다.
4. **V3는 실제로 개선됐는가?** 이번 실행에서 coverage·전체 시간 안정성은 개선됐다. 30분 구간 전환은 오히려 1회 늘고 10분 심판 근거도 약해 모든 면의 개선은 아니다. 실제 분류 정확도 향상은 아직 입증하지 못했다.
5. **수동 정답인가, 진단 지표인가?** 전부 자동 진단 지표다. 제공 경기의 수동 정답을 만들거나 자동 예측을 정답으로 사용하지 않았다. 따라서 실제 Accuracy/Precision/Recall/F1을 보고하지 않는다.
6. **전체 적용 전에 남은 필수 문제는?** 별도 경기의 독립 정답으로 coverage 증가에 따른 오분류를 평가하고, 골키퍼/벤치/피치 문맥과 잘린 박스를 보강해야 한다. 이후 다양한 조명·줌·카메라 컷에서 기존 scene 경계 품질을 확인해야 한다. 현재 V3 실행기는 소규모 캐시 진단용으로 입력·객체를 메모리에 모으며 자체 청크 재개 기능이 없다. RAM 16GB 환경에서 전체 164,487프레임을 이 명령 한 번에 넣지 말고, 스트리밍·청크 저장·V3 전용 체크포인트를 구현·검증한 다음 확대해야 한다. 공 검출 개선은 별도 실험으로 다루며 이번에는 변경하지 않았다.
