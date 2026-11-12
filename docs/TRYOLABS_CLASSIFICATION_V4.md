# Tryolabs Uniform Classification V4

2026-11-12, `4.0.1`. 기존 1,000프레임과 추가 캐시 250프레임을 실제 원본 영상에서 처리했다. 총 **1,250프레임·14,471개 객체 관측**, 검출·트래커 호출 0회다. **GK_B 167개, OTHER 297개 관측**이 나왔다. GK_A는 경기복 색상 미확인으로 비활성화했고, STAFF는 신뢰할 위치·동선 정보가 없어 확정하지 않았다. 이를 검출 실패율이나 해당 인원이 없다는 뜻으로 해석하지 않는다.

전체 UNKNOWN 비율은 V3 **42.37%→V4 43.38%로 증가**했다. 이번 작업은 골키퍼 역할과 관측 가능한 비일치 외관을 분리하고 근거가 약한 선수 역할을 보류하는 확장이다. 정확도 개선을 입증한 결과가 아니다. 전체 경기 분석·YOLO 재검출·학습·GPU 벤치마크는 실행하지 않았다.

추가한 파일은 다음과 같다. V1/V2/V3 코드·정책·보고서는 변경하지 않았다.

| 파일 | 기능 |
|---|---|
| [tryolabs_uniform_v4.py](../starter/tryolabs_uniform_v4.py) | 8개 클래스, 팀/역할/외관/최종 분리, 역할 시간 상태, GK 점유 자격 |
| [V4 HSV JSON](../starter/configs/tryolabs_classification_v4_hsv.json) | 영상에서 측정한 청록 범위, GK_A 비활성, 원본 출처 |
| [V4 정책 JSON](../starter/configs/tryolabs_classification_v4_policy.json) | GK/OTHER/STAFF/화면 전환 조건과 근거 |
| [tryolabs_v4_context.py](../starter/tryolabs_v4_context.py) | 약한 골문 구조·필드 화면 단서, 컷·디졸브 재확인 |
| [diagnose_tryolabs_v4_colors.py](../starter/diagnose_tryolabs_v4_colors.py) | 기존 Track에서 청록·주황·녹색 후보 탐색 |
| [calibrate_tryolabs_v4.py](../starter/calibrate_tryolabs_v4.py) | 자동 외관 집단의 실제 픽셀 분위수 산출 |
| [tryolabs_v4_data.py](../starter/tryolabs_v4_data.py) | V1/V2/V3 파일 보존 감사 |
| [run_tryolabs_classification_v4.py](../starter/run_tryolabs_classification_v4.py) | 같은 프레임·검출·Track의 V3/V4 비교 실행 |
| [analyze_tryolabs_classification_v4.py](../starter/analyze_tryolabs_classification_v4.py) | 구간·골키퍼 색상/움직임·V3 재현 분석 |
| [test_tryolabs_classification_v4.py](../tests/test_tryolabs_classification_v4.py) | 신규 20개 테스트 |

공식 경기 페이지는 Ter Stegen과 Padilla의 선발 출전을 확인하는 참고 자료로 사용했다. 선수 명단이나 경기 이벤트를 실행기의 입력·정답으로 넣지 않았다. 경기 사진의 RGB를 HSV 설정에 복사하지도 않았다. GK_B의 팀 연결은 사용자가 제공한 ‘아틀레틱 청록색’ 가설이며, 영상 속 얼굴·등번호를 통한 Álex Padilla 신원 식별이 아니다. [아틀레틱 공식 경기 페이지](https://www.athletic-club.eus/equipos/athletic-club/2024-25/liga/partidos/fc-barcelona-vs-athletic-club/1_8432/).

먼저 완료된 캐시 0–8,999프레임을 25프레임 간격으로, 기존 세 표본 구간을 5프레임 간격으로 조사했다. 총 510개 프레임에서 1,324개 색상 후보 관측을 얻었다. 청록 H75–110/S≥55/V≥45, 주황 H0–25/S≥100/V≥90, 녹색 H35–74/S≥70/V≥45는 탐색 가설이며 확정된 유니폼 범위가 아니다. [탐색 기록](../data/derived/tryolabs_broadcast/classification_v4/discovery/discovery.json)과 [Track 후보 목록](../data/derived/tryolabs_broadcast/classification_v4/discovery/track_candidates.json)에 남겼다.

청록 탐색에는 관중석의 청록색 옷과 짙은 정장도 들어왔다. 주황 탐색에는 피부·다른 복장·배경이, 녹색 탐색에는 잔디와 형광 심판복 가장자리가 섞였다. 따라서 어느 탐색 색상도 곧바로 GK 역할이나 팀으로 매핑하지 않았다. [청록 후보 이미지](../data/derived/tryolabs_broadcast/classification_v4/discovery/teal_candidates.jpg), [주황 후보 이미지](../data/derived/tryolabs_broadcast/classification_v4/discovery/orange_candidates.jpg), [녹색 후보 이미지](../data/derived/tryolabs_broadcast/classification_v4/discovery/green_candidates.jpg).

GK_B용 색상 집단은 상의 청록 지지≥0.5, 하의≥0.30인 같은 구간+Track 후보가 10회 이상이고 화면 잔디색 비율≥0.35인 관측으로 자동 선택했다. 특정 프레임·Track ID를 허용 목록으로 하드코딩하지 않았다. 최종 집단은 **25관측·6,400개 샘플 픽셀**이다. 사람이 만든 팀/역할 정답은 아니다. 1·99분위수의 양 끝값을 내림/올림해 아래 범위를 얻었다.

| 설정 | H | S | V | 상태 |
|---|---|---|---|---|
| GK_B 청록 외관 | 76–100 | 58–251 | 124–255 | 영상 실측, 역할은 별도 검증 |
| GK_A | 미확정 | 미확정 | 미확정 | `enabled=false`, `bounds=[]` |
| A/B/REF 성분 | 기존 V3와 동일 | 기존 V3와 동일 | 기존 V3와 동일 | 범위 재조정 없음 |

청록 집단의 HSV 중앙값은 `[91,150,196]`이다. [측정 설정](../data/derived/tryolabs_broadcast/classification_v4/calibration/hsv_config.json), [자동 선택 관측 출처](../data/derived/tryolabs_broadcast/classification_v4/calibration/cohort_sources.json), [측정 픽셀](../data/derived/tryolabs_broadcast/classification_v4/calibration/teal_pixels.npz)에 재생성 근거가 있다. 이 집단이 실행 결과에도 포함되므로 독립 성능 평가 자료가 아니다.

GK_A는 주황·녹색 후보가 있어도 실제 경기복인지, 워밍업 복장인지 확인할 충분한 자동 근거를 얻지 못했다. 시즌 자료에 그런 색상이 존재한다는 이유로 활성화하지 않았다. 현재 GK_A 출력 0은 기능을 끈 결과다. 향후 실제 경기복을 확인하고 영상 기반 범위·출처를 추가하면 같은 골키퍼 역할 경로를 사용할 수 있다.

V4는 `team`, `role`, `appearance_label`, `final_label`을 분리한다. 점수와 실제 정답 확률을 혼동하지 않도록 confidence는 항상 `None`이다.

| 최종 클래스 | team | role | 점유 후보 자격 |
|---|---|---|---|
| TEAM_A / TEAM_B | 해당 팀 | PLAYER | fresh·역할 근거 통과 시 |
| GK_A / GK_B | TEAM_A / TEAM_B | GOALKEEPER | fresh·역할 근거 통과 시 |
| REFEREE | None | REFEREE | 제외 |
| STAFF | None | STAFF | 제외 |
| OTHER | UNKNOWN | OTHER | 제외 |
| UNKNOWN | UNKNOWN | UNKNOWN | 제외 |

상의가 청록이어도 `appearance_label=TEAL_KIT_CANDIDATE, final_label=UNKNOWN`일 수 있다. 팀 유니폼 외관이 있더라도 V3의 선수 역할 근거 또는 V4의 필드 화면 문맥이 약하면 최종 UNKNOWN으로 보류한다. 잔디색 주변만으로 벤치의 같은 유니폼을 선수라고 단정하지 않기 위한 추가 조건이다. 이 조건 역시 실제 피치 좌표를 보증하지 않는다.

골키퍼는 상의 지지≥0.35, 하의 지지≥0.15, 최근 30관측 중 같은 후보 최소 10회·65%, 약한 골문 단서 최소 3회를 함께 요구한다. 골문 단서는 주변 흰 선에서 두 기둥과 연결되는 가로 구조를 찾는 Hough 휴리스틱이다. 필드 라인·광고판도 비슷하게 보일 수 있어 약한 근거로만 사용한다. 화면 잔디 비율과 작은 사람 수 역시 ‘필드처럼 보이는 화면’ 단서일 뿐이다.

역할 성립 뒤에는 fresh한 같은 Track의 현재 유니폼 색상 근거가 유지되면 새 골문 단서나 하의 일치가 일시적으로 없어져도 GK를 유지한다. 박스 밖으로 나온 GK를 즉시 잃지 않기 위한 설계다. 현재 색상 근거가 없거나 가림·stale이면 UNKNOWN이며 과거 레이블로 채우지 않는다. ID 변경·10프레임 초과 공백·구간/scene 변경·화면 전환 재확인은 역할을 다시 성립시켜야 한다. 기존에 필드/심판 유니폼이 확인된 Track의 GK 변경도 보류한다. ID 혼합을 확정된 역할 변경으로 해석하지 않는다.

STAFF는 현재 캐시에 없는 **신뢰할 테크니컬 에어리어·피치 밖 위치·카메라 움직임을 보정한 비선수 동선**과 공식 유니폼 비일치, 교체 선수 근거가 아님, 50연속 관측을 요구하는 경로를 구현했다. 실행기에는 이 문맥을 신뢰할 수 있게 공급하는 생산자가 없어 기본값은 미확인이다. 따라서 실제 실행의 STAFF는 0이다. 벤치처럼 보이는 장면이나 정장을 STAFF 정답으로 만들지 않았다. 조건을 만족하는 합성 문맥에서 STAFF 분리·제외만 테스트했다.

OTHER는 상의 최소 10×12px·256픽셀, Laplacian 선명도 분산≥20, 잔디색 비율≤25%, 모든 정의된 색상 지지<10%, 주황/녹색 GK 미확정 탐색 지지<30%, 원래 V3의 현재 사유가 `LOW_COLOR_SUPPORT`일 때만 후보가 된다. 같은 Track에서 10연속 유효 관측이 필요하다. 작은 ROI·가림·흐림·색상 혼동·노란색 역할 보류·청록 GK 후보는 OTHER로 바꾸지 않는다.

기존 유니폼 관측 이력이 있는 Track은 일시적 비일치 외관만으로 OTHER가 되지 않는다. OTHER가 성립한 Track은 같은 연속 구간에서 팀/GK로 자동 승격하지 않고 충돌 시 UNKNOWN으로 보류한다. 이 방어가 오히려 올바른 후속 팀 근거를 막을 위험도 있다. 실제로 기존 팀 레이블 중 131개가 이런 역할/외관 충돌로 보류됐다. 정답이 없어 그것이 필요한 보류인지 잘못된 과보수 판정인지는 알 수 없다.

추가 캐시에서 기존 scene ID가 관중석→벤치의 디졸브까지 이어지는 사례가 있었다. V4는 64×36 영상의 H/S 히스토그램과 평균 픽셀 차이를 직전 및 10프레임 전과 비교해 재확인한다. 기준은 설정 JSON의 공학적 휴리스틱이며 실제 컷 정답으로 보정한 값이 아니다. 감지 시 V4 역할 상태와 20프레임 근거를 초기화하고 해당 프레임을 UNKNOWN으로 출력한다. V3 비교 기준선은 그대로 재생하며, 원래 Track·scene ID를 고치지 않는다. V4 내부 epoch만 새로 부여한다.

재확인된 프레임은 **31개**다: `legacy:1`의 15195–15204, `full:19`의 5127–5137과 5209–5218. 해당 프레임 객체 355개는 모두 UNKNOWN이었다. 빠른 팬·줌이나 조명 변화도 경보를 낼 수 있고, 다른 컷을 놓칠 수도 있다. 이 숫자를 카메라 컷 검출 정확도로 해석하지 않는다. 워밍업/실제 플레이/리플레이 상태는 전부 `UNVERIFIED`로 남긴다. 그 상태를 확정하는 OCR·리플레이 인식 기능은 구현하지 않았다.

실행 구간은 다음과 같다. 파일 시간은 경기 시계와 다르다. 비교 영상은 이 다섯 구간을 순서대로 붙인 **50초**다.

| 입력 | 0 기반 프레임, 양 끝 포함 | 프레임 수 | 객체 관측 |
|---|---|---:|---:|
| 기존 10:00 표본 | 15000–15249 | 250 | 2,880 |
| 기존 30:00 표본 | 45000–45249 | 250 | 398 |
| 기존 60:00 표본 | 90000–90249 | 250 | 4,854 |
| 기존 추가 05:20 표본 | 8000–8249 | 250 | 4,381 |
| 이번 추가 03:20 표본 | 5000–5249 | 250 | 1,958 |

최종 결과는 [classification_v4/final](../data/derived/tryolabs_broadcast/classification_v4/final/)이다. `preview`, `validated`, 이전 재생 등 중간 결과도 보존했으며 아래 수치는 **final**만 사용한다. 같은 사람의 여러 프레임·stale 관측이 포함돼 고유 인원 수가 아니다.

| 최종 클래스 | V3 관측 수 | V4 관측 수 |
|---|---:|---:|
| TEAM_A | 3,244 | 3,018 |
| TEAM_B | 4,317 | 3,933 |
| GK_A | 지원 없음 | 0, 비활성 |
| GK_B | 지원 없음 | 167 |
| REFEREE | 778 | 778 |
| STAFF | 지원 없음 | 0, 문맥 부족 |
| OTHER | 지원 없음 | 297 |
| UNKNOWN | 6,132 (42.37%) | 6,278 (43.38%) |

기존 **1,000프레임만** 비교하면 V3 UNKNOWN 4,213/12,513(33.67%)→V4 4,613/12,513(36.87%)다. 여기서 OTHER는 4개, GK_B는 167개다. 추가 250프레임에서는 OTHER 293개가 나왔으며 대부분 비필드 화면의 외관이다. 사람의 직업·소속이나 관중 여부를 정답으로 붙인 결과는 아니다.

아래는 모델 간 변경 표다. V3를 정답으로 사용하는 Confusion Matrix가 아니다.

| V3 → V4 | A | B | GK_B | REF | OTHER | UNKNOWN |
|---|---:|---:|---:|---:|---:|---:|
| A | 3,018 | 0 | 0 | 0 | 0 | 226 |
| B | 0 | 3,933 | 0 | 0 | 0 | 384 |
| REF | 0 | 0 | 0 | 778 | 0 | 0 |
| UNKNOWN | 0 | 0 | 167 | 0 | 297 | 5,668 |

A/B/REF 유지율은 각각 **93.03% / 91.10% / 100%**다. V3의 A/B에서 UNKNOWN으로 바뀐 이유는 팀·역할 문맥 부족, 이전 OTHER와 역할 충돌, 화면 전환 재확인이다. 클래스 유지율은 분류 정확도·Recall이 아니다.

전환을 공정하게 비교하도록 같은 V4 epoch 안의 인접 프레임만 양쪽에서 집계하면 클래스 변경 V3 513→V4 454회, 역할 변경 435→454회다. 역할 종류가 추가됐고 역할 변경은 오히려 늘었다. 원래 V3의 모든 인접 프레임을 집계한 525회와 V4 epoch를 구분한 454회를 바로 안정성 개선으로 비교해서는 안 된다. 직접 알려진 클래스끼리의 A↔B, REF↔GK, OTHER↔GK 전환은 이번 출력에서 없었다. UNKNOWN을 거치는 전환은 원시 로그에 별도로 있다.

골키퍼 후보가 OTHER가 된 관측은 0, V3 REFEREE가 V4 GK가 된 관측도 0이다. 이는 정책상 충돌을 보류한 결과이며 실제 골키퍼·심판 오류가 없다는 증명이 아니다. 실제 STAFF→PLAYER 오류는 정답/신뢰할 STAFF 문맥이 없어 계산하지 않았고 `actual_staff_player_errors=null`로 저장했다. 문맥이 없는 상태에서 0으로 보고하지 않았다.

청록 후보는 `legacy:3 / ID 2`에서 176개 fresh 관측으로 이어졌고 9개는 역할 대기, 167개는 GK_B였다. 이 ID는 결과를 설명하기 위한 출처이며 분류 조건에 하드코딩하지 않았다. 상의 지지율 10/50/90분위수는 **0.436/0.568/0.659**, 하의는 **0.251/0.386/0.522**, 마스크 픽셀 Hue 중앙값의 분위수는 **91/91/92**였다. 골문 구조 단서는 69.32% 관측에 있었다. 상의와 하의의 같은 청록 범위 지지가 함께 유지되는 것을 확인했으며, 이것이 해부학적 하의 분할 정답이라는 뜻은 아니다.

청록 후보 샘플 12,143픽셀에서 V3 navy/burgundy/lavender/pink/yellow와 겹치는 마스크 픽셀은 모두 0이었다. 해당 표본의 측정 결과이며 다른 조명에서의 일반화 보증이 아니다. 후보 박스 폭 중앙값은 43.5px로 장갑을 신뢰할 수 있게 식별하지 못했다. 장갑 검출기를 추가하거나 장갑처럼 보이는 픽셀을 역할 정답으로 사용하지 않았다.

후보 박스 중심의 프레임 간 픽셀 이동량 10/50/90분위수는 1.21/4.20/8.51px다. 카메라 팬·줌을 보정하지 않았으므로 실제 이동 속도나 비선수 동선으로 해석할 수 없다. 골문·피치 좌표, 손·발의 자세를 신뢰할 수 있게 추정하지 못한 한계가 남는다.

점유는 같은 공 후보와 기존 박스 하단 양 모서리의 **45px 거리**를 유지하고 허용 역할에 GOALKEEPER만 추가했다. 같은 팀의 20연속 프레임 조건과 V3 누적 결과는 변경하지 않았다. 이번 V3/V4의 후보 ID 차이는 **0프레임**이다. 양쪽 모두 INSUFFICIENT_EVIDENCE 1,076, PROXIMITY_PENDING 160, PROXIMITY_20_FRAME_EVIDENCE 14프레임이다. GK 추가가 이번 표본의 점유 판정을 바꾸지는 않았다.

공이 GK 박스 안에 있지만 발 모서리에서 45px를 넘는 `gk_body_near_foot_far` 진단은 이번 표본에서 0프레임이다. 공을 손으로 잡았다는 정답이나 손 위치 추정은 아니다. 합성 테스트에서는 몸통 근처 공이 기존 발 거리 기준을 통과하지 못하는 경우를 확인했다. 이 기준의 한계가 해결됐다는 뜻은 아니다.

산출물은 아래 링크에서 확인할 수 있다. 모두 로컬 `data/`에 있으며 Git에는 포함하지 않는다.

- [V3/V4 비교 영상](../data/derived/tryolabs_broadcast/classification_v4/final/comparison_v3_v4.mp4): 1920×800, H.264, 25fps, 음성 없음. 1,250프레임 전체 디코딩 확인.
- [클래스별 HSV 히스토그램](../data/derived/tryolabs_broadcast/classification_v4/final/hsv_distributions.png), [GK 색상 지속성 그래프](../data/derived/tryolabs_broadcast/classification_v4/final/analysis/goalkeeper_color_stability.png), [GK 프레임별 색상 CSV](../data/derived/tryolabs_broadcast/classification_v4/final/analysis/goalkeeper_color_series.csv).
- [대표 Contact Sheet](../data/derived/tryolabs_broadcast/classification_v4/final/contact_sheet_01.jpg), [GK 후보 Track 이미지](../data/derived/tryolabs_broadcast/classification_v4/final/contact_sheets/legacy_3_id2.jpg), [OTHER/화면 전환 후보 이미지](../data/derived/tryolabs_broadcast/classification_v4/final/contact_sheets/full_19_id22.jpg). 총 71개 Track 이미지·12개 합본으로 전체 프레임과 상의 ROI를 함께 제공한다.
- [객체 JSONL](../data/derived/tryolabs_broadcast/classification_v4/final/objects.jsonl), [객체 CSV](../data/derived/tryolabs_broadcast/classification_v4/final/objects.csv), [프레임 CSV](../data/derived/tryolabs_broadcast/classification_v4/final/frames.csv), [요약](../data/derived/tryolabs_broadcast/classification_v4/final/summary.json), [추가 진단](../data/derived/tryolabs_broadcast/classification_v4/final/analysis/diagnostics.json).
- [원본 보존](../data/derived/tryolabs_broadcast/classification_v4/final/verification.json), [재현성](../data/derived/tryolabs_broadcast/classification_v4/final/reproducibility.json), [공통 비교 검사](../data/derived/tryolabs_broadcast/classification_v4/final/comparison_checks.json), [테스트 기록](../data/derived/tryolabs_broadcast/classification_v4/final/test_results.json), [코드·설정 해시 manifest](../data/derived/tryolabs_broadcast/classification_v4/final/manifest.json).

보존 검사에서 기존 V1/V2/V3 코드·설정·보고서·모델·검출/Track/SQLite·파생 결과 **1,084개 파일의 SHA256이 모두 유지**됐다. 원본 13,962,360,320바이트 영상은 size/mtime 불변을 확인했다. 영상 전체 SHA256을 재계산한 검사는 아니다. SQLite는 기존 V3의 복사 방식으로 DB/WAL/SHM을 V4 폴더에 복사하고 복사본에만 연결했다. 캐시 작성 프로세스를 멈춘 상태로 실행해야 한다.

기존 V3 1,000프레임 분류는 **12,513/12,513개 일치**했다. 최종 1,250프레임을 별도 프로세스에서 재생한 `input_frames.jsonl`, `objects.jsonl`, `objects.csv`, `frames.csv`도 바이트 단위로 일치했다. 요약은 영상 미생성 재생의 디코딩 수 필드를 제외하면 같다. 이는 동일 환경·동일 입력의 재현 검사이며 서로 다른 OS/디코더 간 비트 일치 보증은 아니다.

입력 SHA256은 `2e993a07b94e46ee018133bfb4997c7b25ba07564dbf30fb3efbe7469f8f218c`, V4 HSV는 `b087dfe548ad934271482737240cbf28c69e1a1dffc37cf3cc3ac1bb53cefb95`, 정책은 `5e9f89393148cde5e5303f48f19b8fe23003ac39870476e806f1bf7fd8c5f63f`다. 실제 코드·라이브러리 버전은 manifest에 기록했다.

Mac Python3.11 CPU에서 **90개 테스트 통과·실패 0개, 3.33초**였다. 기존 70개와 신규 20개다. 색상 단독 GK 금지, 상·하의/시간/골문 조건, 골문 밖 역할 유지, ID/공백/scene 초기화, OTHER/UNKNOWN, 기존 유니폼 이력, STAFF 문맥/교체 선수 제외, 심판 제외, GK 점유 허용, 발 거리 한계, 컷/디졸브, 누락 입력, 합성 영상 캐시 재현을 검사했다. 실제 경기의 Precision/Recall/F1이 아니다.

```text
python -m pytest -q tests/test_tryolabs_classification_v2.py tests/test_tryolabs_full_match_v2.py tests/test_tryolabs_windows.py tests/test_tryolabs_classification_v3.py tests/test_tryolabs_classification_v4.py
90 passed in 3.33s
```

제공 경기의 수동 정답은 생성하지 않았다. [RULES.md](../RULES.md)의 “No hand labels on the provided game, in any amount and for any purpose”를 준수했다. 진단 이미지 관찰을 정답 CSV·Track 허용 목록·특정 프레임 보정 규칙으로 바꾸지 않았다. 따라서 Accuracy/Precision/Recall/F1은 보고하지 않는다.

Windows CMD에서는 기존 CUDA 가상환경을 그대로 사용한다. V4는 torch import·GPU 모델 로딩·YOLO 재추론을 하지 않는다. 새 모델이나 requirements를 추가하지 않았고 기존 Windows requirements가 충분하다. 아래 명령은 저장소 루트 `SNP` 기준이다. 그 아래에 `more-than-a-hack-2027`을 다시 만들 필요는 없다.

```bat
cd /d C:\Users\kjkim\mtah_gpu\SNP
git switch KKJ
git pull --ff-only origin KKJ
call C:\Users\kjkim\mtah_gpu\.venv\Scripts\activate.bat
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=2
python -m pytest -q tests\test_tryolabs_classification_v2.py tests\test_tryolabs_full_match_v2.py tests\test_tryolabs_windows.py tests\test_tryolabs_classification_v3.py tests\test_tryolabs_classification_v4.py
```

기존 Windows V3의 동결 입력이 있으면 그것을 재사용한다. 폴더명은 실제 V3 실행 폴더로 바꾼다. V4 결과는 새 폴더에만 쓴다.

```bat
python starter\run_tryolabs_classification_v4.py --source prepared --cache data\derived\tryolabs_broadcast\classification_v3\windows_samples_01\input_frames.jsonl --output data\derived\tryolabs_broadcast\classification_v4\windows_from_v3_01
python starter\analyze_tryolabs_classification_v4.py --run data\derived\tryolabs_broadcast\classification_v4\windows_from_v3_01 --frozen-v3 data\derived\tryolabs_broadcast\classification_v3\windows_samples_01\objects.jsonl
```

V3 동결 입력이 없고 전체 처리 캐시의 추적 청크가 완료돼 있다면 다음 750프레임 후처리를 대신 실행한다. `run.sqlite`만 있고 `chunks/.../tracking.jsonl.gz`, `objects.jsonl.gz`, `chunk.json`이 없으면 실행할 수 없다. 원본은 `data\broadcast\BAR-ATH_broadcast-001.mp4`에 있어야 한다.

```bat
python starter\run_tryolabs_classification_v4.py --source full --cache data\derived\tryolabs_broadcast\full_match_v2\windows_gtx1660 --starts 600 1800 3600 --seconds 10 --output data\derived\tryolabs_broadcast\classification_v4\windows_750_01
```

`windows_gtx1660`은 실제 V2 실행 폴더명으로 바꾼다. 이 명령은 GPU 벤치마크가 아니다. 누락 캐시는 오류로 중단하며 YOLO를 자동 실행하지 않는다. Windows의 Track 캐시가 Mac의 과거 구간별 Track과 다르면 같은 영상 시간이라도 로컬 표와 관측 수/ID가 달라질 수 있다. 한 실행 안에서는 항상 같은 입력을 V3/V4가 공유한다.

재현 검사는 다음과 같다. 같은 output을 재사용하면 덮어쓰지 않고 중단하므로 폴더명은 항상 새 것으로 지정한다.

```bat
python starter\run_tryolabs_classification_v4.py --source prepared --cache data\derived\tryolabs_broadcast\classification_v4\windows_from_v3_01\input_frames.jsonl --output data\derived\tryolabs_broadcast\classification_v4\windows_replay_01 --no-video --no-visuals
fc /b data\derived\tryolabs_broadcast\classification_v4\windows_from_v3_01\objects.jsonl data\derived\tryolabs_broadcast\classification_v4\windows_replay_01\objects.jsonl
fc /b data\derived\tryolabs_broadcast\classification_v4\windows_from_v3_01\frames.csv data\derived\tryolabs_broadcast\classification_v4\windows_replay_01\frames.csv
```

로컬 실제 실행과 설정 재생성 명령은 다음과 같다. 최초 캐시 연결 실행은 `preview`였고 최종 설정으로 다시 실행한 결과는 `final`이다. 이 이름들은 이미 존재하므로 다시 실행할 때 새 이름을 사용한다.

```sh
data/derived/tryolabs_broadcast/.venv/bin/python starter/diagnose_tryolabs_v4_colors.py
data/derived/tryolabs_broadcast/.venv/bin/python starter/calibrate_tryolabs_v4.py
data/derived/tryolabs_broadcast/.venv/bin/python starter/run_tryolabs_classification_v4.py --extra-cache data/derived/tryolabs_broadcast/full_match_v2/main --extra-start 5000 --extra-frames 250 --output data/derived/tryolabs_broadcast/classification_v4/preview --no-video --no-visuals
data/derived/tryolabs_broadcast/.venv/bin/python starter/run_tryolabs_classification_v4.py --source prepared --cache data/derived/tryolabs_broadcast/classification_v4/preview/input_frames.jsonl --output data/derived/tryolabs_broadcast/classification_v4/final
data/derived/tryolabs_broadcast/.venv/bin/python starter/analyze_tryolabs_classification_v4.py --run data/derived/tryolabs_broadcast/classification_v4/final
```

오류·미해결 위험은 다음과 같다.

- 미완성 `.work-000009000`은 gzip 스트림 종료가 완전하지 않은 캐시였다. 9000프레임부터의 이 청크는 복구·재추론하지 않고 탐색/실행에서 제외했다. 숫자로 시작하는 완료 청크 0–8,999와 기존 표본만 사용했다. 최종 요청 1,250프레임에는 누락이 없었다.
- GK_A는 실제 경기복 근거가 없고, STAFF 문맥 생산자도 없다. 두 클래스를 실제 영상에서 검증 완료했다고 주장하지 않는다. GK_B도 청록+복합 문맥의 역할 가설이며 Padilla 본인인지 확정하지 않았다.
- 자동 청록 집단은 한 구간·한 Track 중심이라 일반화 편향이 크다. 워밍업/리플레이를 자동 배제하지 못하므로 시즌 다른 복장이나 골키퍼 교체에 그대로 일반화할 수 없다.
- Hough 구조와 잔디색은 실제 골문/피치 정답이 아니다. 노란 GK·형광 스태프·같은 유니폼의 교체 선수, 그림자·채도 손실, bbox 혼합을 완전히 해결하지 못했다. OTHER와 UNKNOWN 경계 및 OTHER 이력 고정이 실제 선수 근거를 과도하게 막을 수 있다.
- 화면 전환 경보는 빠른 팬·줌에도 반응할 수 있다. 전환을 놓치면 기존 잘못 연결된 ID의 역할이 이어질 위험이 남는다. 기존 Track 자체는 고치지 않았다.
- 손으로 공을 잡은 GK의 발 거리 문제는 진단만 추가했다. 실제 공·손 제어 판정이나 점유 알고리즘 개선은 별도 실험이다.
- Windows 네이티브·GTX1660 SUPER 실행은 이번에 수행하지 않았다. 로컬 CPU 결과를 GPU 실측으로 표현하지 않는다.
- V4는 **최대 5,000프레임의 소규모 실행**으로 제한한다. 준비 입력은 상한을 넘기기 전에 읽기를 중단하며, 기존 출력은 덮어쓰지 않는다. 자체 청크 저장/중단 재개는 아직 없으므로 전체 164,487프레임을 실행하려면 스트리밍·V4 전용 체크포인트·재개 동일성 검증부터 구현해야 한다. 현재 명령으로 전체 처리를 시작하지 않는다.

후속 우선순위는 GK_A 실제 경기복의 자동 근거 확보, 허용되는 별도 경기의 정답 평가, 신뢰할 피치·골문/벤치 문맥과 카메라 움직임 보정, OTHER/역할 이력 정책의 외부 평가, 전체용 스트리밍·재개다. 공 모델과 손 제어 기준의 개선은 별도 버전에서 비교해야 한다.
