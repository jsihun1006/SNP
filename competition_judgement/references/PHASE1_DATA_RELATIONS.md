# 1단계 구현: 선수 데이터 검증·공간 관계·경합 후보 기준 모델

작성일: 2026-11-09. 신경망·학습·군집·H/A/D 판별은 구현하거나 실행하지 않았다. 이 모듈은 최종 점유 판별 시스템의 **독립적인 선수 관계/구조 분석 모듈**이다. Smart Tagging, 볼 좌표, 볼 정답, 공개 GT, broadcast를 사용하지 않는다.

## 1. 구현 위치와 실행

실제 패키지는 `MTAH_Relational_Starter/mtah_relational/`에 유지했다. 루트로 복사할 필요가 없다. 기존 `starter/`, `evaluation/`, `games.py`, 원본 데이터와 기존 제출 baseline은 변경하지 않았다. `games.py`의 상수만 읽는다. GitHub push는 하지 않았다.

프로젝트 루트에서:

```bash
conda activate mtah2027
export PYTHONPATH="$PWD/MTAH_Relational_Starter:$PWD"

# 프레임 ID 전체·팀 매핑·단위·메타데이터 25fps 검사
python -m mtah_relational.cli check

# 전체 프레임 관계 및 지수. 새 로컬 폴더만 허용한다.
python -m mtah_relational.cli phase1 \
  --out-dir data/derived/relational_phase1 \
  --zone-stride 25

# 선택: 후속 모델용 source-slot 배열도 로컬에 저장
python -m mtah_relational.cli phase1 \
  --out-dir data/derived/relational_phase1_with_nodes \
  --zone-stride 25 --save-nodes
```

같은 출력 폴더가 이미 있으면 오류를 내며 덮어쓰지 않는다. 중간 실패 시 임시 디렉터리를 정리하고 최종 폴더를 공개하지 않는다. `--zone-stride 1`은 모든 평가 프레임의 48구역을 출력해 파일이 매우 커진다. 기본25는 **보조 구역 지표만 1fps 샘플링**한다. 전역 관계·지수는 여전히 모든 프레임에서 계산한다. `--zone-stride 0`은 구역 출력을 비활성화한다.

기존 nested CLI의 `extract/plot`도 남겨 두었다. `extract`의 ±1초 next delta는 미래를 포함하는 **기술 통계**이므로 자기지도 모델 입력에 그대로 넣지 않는다. 새 `phase1`과 `preprocessing.spatial_input`은 그 사전 계산 delta를 사용하지 않는다.

## 2. 입력 무결성 정책

| 항목 | 구현 |
|---|---|
| frame | raw의 모든 record를 순회하며 `frame_id == 1-based line number` 검사. count·양끝 번호만 비교하지 않음 |
| 시간 | `frame=frame_id−1`, `time_s=frame/25`; XML FPS와 공식25 일치 확인 |
| 홈/원정 | XML `Score.idLocalTeam/idVisitingTeam`를 사용. Team 나열 순서에 의존하지 않음 |
| 슬롯 순서 | `PlayerChannelRef→PlayerChannel→Player` 참조 순서와 xy 채널 순서·단위 검증 |
| 좌표 | normalized xy→공식105.3×68m. XML105×68m와 의도적 차이를 report에 기록 |
| 볼·속도 | 같은 record의 볼 field는 수치 파싱하지 않음. 선수의 세 번째 speed 값도 미사용 |
| 누락 | xy 모두 NaN, 부분 NaN, infinity/비숫자·잘못된 field 수를 구분 |
| bounds | 피치 밖 관측을 flag. 기본2m margin을 넘는 좌표만 분석 view에서 NaN 처리하며 clip/원본 수정 없음 |
| duplicate | 같은 팀 관측 사이 거리≤0.2m를 의심 쌍으로 기록. 자동 병합·삭제하지 않음 |
| 전후반 | 공식 `games.half_video_blocks()`만 사용. SmartTag/sync/X-mask에 의존하지 않음 |

다른 팀 선수의 동일 위치를 곧바로 중복이라고 하지 않는다. 근접한 두 선수일 수 있다. 같은 팀0.2m 의심도 실제 중복의 정답은 아니며 보수적 품질 정책이다. 슬롯은 인물이 아니므로 개인 속도·가속도·지속적 ID 연결을 만들지 않았다. 메타데이터 매핑을 검증하는 것은 각 관측의 실제 팀 분류가 정확하다고 검증하는 것과 다르다.

현재 `quality_valid`는 양 팀 분석 인원이 각각8–11명이고, 같은 팀 duplicate 의심·피치 밖 관측·부분 NaN·invalid coordinate가 없을 때 True다. 두 좌표 모두 NaN인 비활성 슬롯은 인원 기준 외에는 자동 실패 이유가 아니다. bounds margin 안의 바깥 관측도 보존하지만 strict outside flag 때문에 지수 eligibility는 False다. 이 기준은 GT로 튜닝하지 않은 초기 전역 정책이다.

`quality_valid`와 `index_valid`는 다르다. 지수는 추가로 공식 반 구간과 양 팀 관계의 유효성이 필요하다. 나쁜 품질의 프레임도 CSV 행은 보존하고 지수는 NaN으로 둔다. NaN은 비경합0·D·X 예측을 뜻하지 않는다.

## 3. 구현 모듈

| 파일 | 기능 |
|---|---|
| [data_quality.py](../MTAH_Relational_Starter/mtah_relational/data_quality.py) | `read_metadata`, `parse_players`, `iter_player_frames`, `assess_observations`; strict frame/팀/단위/관측 QA |
| [features.py](../MTAH_Relational_Starter/mtah_relational/features.py) | 팀별 중심/폭/길이/응집도/밀집도, 상대 거리·최근접·pair/component; legacy 차분의 경계 검사 보강 |
| [spatial_schema.py](../MTAH_Relational_Starter/mtah_relational/spatial_schema.py) | H/A 대칭24개 feature와 valid mask, ordered H/A 구조 `[2,10]` |
| [candidate_baseline.py](../MTAH_Relational_Starter/mtah_relational/candidate_baseline.py) | `CausalContestedIndex`: fit 없는 현재 접촉·과거1초 접근 변화 지수 |
| [zones.py](../MTAH_Relational_Starter/mtah_relational/zones.py) | 보조8×6 구역 count·혼합·밀집도·구역 내부 pair |
| [preprocessing.py](../MTAH_Relational_Starter/mtah_relational/preprocessing.py) | `SpatialInput`, `spatial_input(s)`; raw부터 숨기는 frame-local 입력 계약 |
| [phase1.py](../MTAH_Relational_Starter/mtah_relational/phase1.py) | streaming 추출, 집계 report, optional memmap, 새 폴더 원자적 저장 |
| [cli.py](../MTAH_Relational_Starter/mtah_relational/cli.py) | `phase1` 추가; strict `check`; 기존 nested 명령 유지 |

## 4. 지표 정의

팀별 `home_*`, `away_*`를 각각 저장한다. 후속 owner 모델이 어느 팀의 구조가 변했는지 볼 수 있도록 대칭 지표만 남기지 않았다.

* `cx,cy`: 현재 관측 집합 평균. `length=x_max−x_min`, `width=y_max−y_min`.
* `length_p90_p10,width_p90_p10`: 각 축 P90−P10. 기존 min/max도 보존한다. GK 제거를 보장하는 값은 아니다.
* `radius=compactness_m`: 팀 중심까지 평균 유클리드 거리. `bbox_area=length×width`.
* `density_per_m2=n_players/bbox_area`: 면적0이면 NaN. convex hull 밀집도가 아니다.
* `nearest_teammate_m`: 팀별 각 관측의 최근접 동료거리 평균. 두 명 미만이면 NaN.
* `cross_min_m`, `cross_mean_m`: 전체 H×A 거리의 최소/평균.
* `cross_nearest_mean_m/median_m`: H 각 선수·A 각 선수의 최근접 상대거리를 합친 평균/중앙값. 전체 pair 평균과 다르다.
* `cross_pairs_3m/5m/10m`: 전역 거리행렬에서 inclusive threshold의 상대 쌍 수.
* `cross_density_rm`: 쌍 수/(관측 H수×관측 A수). MNAR 누락·중복을 완전히 보정하는 수치는 아니다.
* `max_cross_cluster_3m`: H-A 3m edge만으로 연결된 최대 component 인원. 고립 노드는 제외하며 연결 사슬을 포함한다.

전역 거리 계산에는 구역 경계를 적용하지 않는다. 보조 `within_zone_pairs_*`는 이름 그대로 **같은 구역 안의 쌍**만 센다. 따라서 이를 합쳐도 전역 쌍 수와 같지 않을 수 있다. 48구역 혼합도는 `2min(nH,nA)/(nH+nA)`, 빈 구역은 NaN이며 밀집도는 인원/고정 구역면적이다. `density_change_per_m2_s`는 같은 반의 연속 샘플 양끝 품질이 유효할 때의 차분이다. 샘플 사이 모든 프레임의 품질을 검증한 변화량은 아니다.

## 5. Rule-based Contested Situation Index

학습·분포 적합·공개 GT·미래 프레임 없이 고정 수식으로 작동한다.

```text
contact = (3m 내 상대가 있는 H 인원 + 해당 A 인원)/(nH+nA)
cluster = max_cross_cluster_3m/(nH+nA)
proximity = exp(-cross_min_m/3)
spatial_contact_index = 0.5*contact + 0.3*cluster + 0.2*proximity

# 과거25프레임을 포함한 총26개 관측 모두가 유효한 경우
delta_contact = contact[t] - contact[t-25]
nearest_approach_m_per_s = cross_nearest_mean_m[t-25] - cross_nearest_mean_m[t]
approach_change_index = 0.5*clip(delta_contact,0,1)
                     + 0.5*clip(nearest_approach_m_per_s/5,0,1)
contested_index = 0.8*spatial_contact_index + 0.2*approach_change_index
```

과거 문맥이 부족하거나 사이에 invalid frame이 있으면 `temporal_available=False`, 변화 지표는 NaN이며 현재 spatial index만 사용한다. 현재 프레임도 invalid이면 모든 index는 NaN이다. 프레임 gap·반 변경에서 history를 끊는다. 후반으로 전반 history를 넘기지 않는다.

지수 범위는0–1이지만 **경합 확률이 아니다.** 근접·혼합 관계와 집합 수준 접근 변화의 정의된 조합이다. approaching 값도 개인 속도가 아니며 관측 구성 변화의 영향을 받는다. 특정 경기 프레임에 맞춰 weight나 threshold를 수정하지 않았다. 좁은 팀 구조·압박·세트피스 대기와 실제 볼 경합을 확정 구분하지 못한다.

## 6. 출력 형식

| 파일 | 행/shape | 의미 |
|---|---|---|
| `features.csv` | 199,319행 | `frame,period,time_s`, ordered H/A 구조, 전역 관계·품질·지수·변화 가용성 |
| `symmetric_features.csv` | 199,319행 | `frame,period`, 대칭24개 및 각 `valid_*`24개 |
| `zones_48.csv` | 기본1fps 보조 출력 | sampled `frame,period,zone`, 인원·혼합·밀집·구역 내부쌍·차분. stride0이면 비어 있음 |
| `quality_report.json` | 집계 JSON | strict frame/FPS/metadata 검증, 전체·평가구간·각 반·그 외 품질통계와 정책 |
| `schema.json` | JSON | schema version, feature24·미래 복원 target13·ordered team10개 정의 |
| `LOCAL_ONLY.txt` | 안내 | 경기 파생물의 로컬 보관 조건 |
| `source_slot_xy_m.npy` (선택) | float32 `[T,46,2]` | bounds 처리 후 좌표. row i=frame i; 인물 ID tensor가 아님 |
| `observed.npy`, `team_codes.npy` (선택) | bool `[T,46]`, int8 `[46]` | 분석 view의 finite mask와 소스 팀 코드 |

파일의 NaN과 `valid_*`를 함께 읽는다. symmetric mask는 수학적 관측 가능성이고 strict index quality와 같지 않다. 복원 학습 시 clean target-valid와 별도 품질 조건을 함께 고려해야 한다.

예시 로컬 로딩:

```python
import pandas as pd
df = pd.read_csv("data/derived/relational_phase1/features.csv")
reliable = df.loc[df["index_valid"], ["frame", "contested_index"]]
home = df.filter(regex=r"^home_")
away = df.filter(regex=r"^away_")
# 위 데이터프레임을 클라우드 LLM/공개 저장소에 전송하지 않는다.
```

## 7. 테스트와 환경

기존6개와 신규60개, 합계 **pytest 66개 통과**. 모든 테스트는 합성 데이터만 사용한다.

* 정확한3/5/10m 포함 경계·최근접 상대 평균·compactness·밀집도·bipartite 연결 사슬.
* 10개 seed의 선수 순열 불변성과 H/A 교환 대칭, ordered team 표현의 교환.
* 완전/부분 NaN·infinity·malformed field·bounds·동일 팀 duplicate 의심·실제 상대 근접의 구분.
* Team 순서와 반대인 명시적 H/A mapping, channel ref 순서, FPS·단위·slot mismatch 거부.
* 내부 누락+중복이 상쇄된 ID·역순·0-based raw·record 수 오류 거부.
* period0/half/gap 경계, invalid history를 가로지르지 않는 causal 지수.
* hidden raw의 좌표·인원·팀 구성·bounds/duplicate 상태를 바꿔도 **모든 model input이 동일**한지 검증. hidden 시점의 품질·팀 구조·node 수까지 차단.
* 미래 관측 변경이 과거 causal index에 영향을 주지 않음.
* 볼·속도 값을 바꿔도 추출 결과가 동일함. SmartTag decoy 파일 없이 동작.
* 합성 end-to-end CSV/memmap 정렬, output overwrite 방지와 실패 시 partial output 정리.

이번 실제 환경은 `mtah2027` Python3.11.17, NumPy2.4.6, Pandas3.1.0이었다. `pytest`, torch, PyG는 이 Conda 환경에 설치되어 있지 않았다. torch/PyG는 이번 단계에 필요하지 않다. 기존 requirement의 Pandas 상한을 `<4`로 넓혀 실제 검증한 환경과 일치시켰다.

**설치할 경우 사용자가 로컬에서 실행할 명령:**

```bash
conda activate mtah2027
python -m pip install "pytest>=8,<10"
export PYTHONPATH="$PWD/MTAH_Relational_Starter:$PWD"
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  MTAH_Relational_Starter/tests
```

이번에는 Conda 환경을 변경하지 않았다. base interpreter의 pytest9.0.3으로66개를 실행한 뒤, **mtah2027 interpreter에서도 같은66개를 실행**했다. 후자는 base의 pure-Python pytest 패키지 경로를 해당 프로세스의 `sys.path` 뒤에만 추가하고 plugin 자동 로딩을 껐다. NumPy/Pandas와 검사 대상 코드는 mtah2027 환경에서 로딩했다. 표준 설치가 완료된 것처럼 보고하지 않는다.

재현에 필요한 임시 실행 방식(이 Mac에 이미 있는 base pytest를 이용하는 경우):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/MTAH_Relational_Starter:$PWD" \
/opt/anaconda3/envs/mtah2027/bin/python - <<'PY'
import sys
sys.path.append('/opt/anaconda3/lib/python3.14/site-packages')
import pytest
raise SystemExit(pytest.main(['-q', '-p', 'no:cacheprovider',
                             'MTAH_Relational_Starter/tests']))
PY
```

## 8. 실제 경기에서 실행한 결과

**합성 결과와 별도로**, mtah2027 interpreter로 실제 tactical ATD 전체를 실행했다. 결과는 로컬 `data/derived/relational_phase1_v1_20261109/`에 있다. 좌표·프레임별 추출물을 대화나 외부 서비스로 전송하지 않았으며 아래는 재구성 불가능한 집계 통계다.

| 검사/통계 | 전체 | 공식 평가 구간 |
|---|---:|---:|
| 프레임 수 | 199,319 | 158,857 |
| 평균 관측 선수 수 (bounds 전) | 18.9767 | **20.4874** |
| source slot 양좌표 NaN 비율 | 58.7463% | 55.4621% |
| 같은 팀0.2m duplicate 의심이 있는 프레임 | 1,356 | **952 (0.5993%)** |
| 최소 한 선수가 피치 밖인 프레임 | 31,484 | **22,876 (14.4004%)** |
| 2m margin 초과로 분석에서 제외한 slot-frame | 14,139 | 5,391 |
| 부분 NaN / invalid coordinate slot-frame | 0 / 0 | 0 / 0 |
| index 유효 프레임 | 111,733 | **111,733 (70.3356%)** |
| 과거1초 관계 변화까지 유효한 프레임 | 101,558 | 101,558 |

NaN 비율의 분모는 **46 source slots×프레임**이다. 22명의 실제 선수 중55%가 누락됐다는 뜻이 아니다. 비활성 slot이 많다는 정상적인 구조도 포함된다. 인원 카운트에는 중복 의심·필드 밖 관측이 포함될 수 있어 물리적 선수 수의 정답은 아니다.

전반79,467프레임 평균 관측20.1211명, 지수 유효57,085프레임. 후반79,390프레임 평균20.8541명, 지수 유효54,648프레임이었다. quality 기준 때문에 평가 구간47,124프레임에서 지수를 보류했다. 그 안의 진짜 경합 비율이나 미검출률은 알 수 없다.

엄격한 ID 순회는 `frame_id=1..199319`, `frame=0..199318` 모두 통과했다. 명시적 Local/Visiting mapping과 channel order를 검증했으며 source slot은 H24개/A22개였다. **이는 실제 홈24명/원정22명이라는 뜻이 아니다.** normalized 좌표 채널과 XML25fps는 확인했지만 영상 FPS를 디코더로 별도 측정하거나 프레임의 시각적 위치를 검증하지는 않았다.

기본25 stride에서 구역 출력은 **304,992행**이다. 후속 로컬 output 검사에서 전역·대칭 CSV 각199,319행의 frame 정렬,3≤5≤10m pair count, index 범위/가용 mask, symmetric finite/valid mask, zone 샘플/period/번호 계약 위반이 모두0이었다. 이 별도 검사 결과는 같은 로컬 폴더의 `output_validation.json`에 저장했다. 이는 실제 경합 정답 평가가 아니다.

## 9. 다음 모델에 전달할 인터페이스

`iter_player_frames`의 `PlayerFrame`은 raw source slot view다. `spatial_input`은 관측을 정제하고 다음을 반환한다.

```text
SpatialInput
  frame, period               routing/boundary only, 절대 경기시각 임베딩 금지
  xy_m:             [N_t,2]    finite observed node set
  team:             [N_t]     0=H,1=A
  values:           [24]      symmetric, invalid values는0으로 sanitize
  feature_valid:    [24]      관측/정의 가능성
  quality:          [4]       min/max 팀인원, duplicate/OOB 비율
  team_structure:   [2,10]    H/A 각각의 구조 표현, 학습된 embedding 아님
  team_structure_valid:[2,10]
  hidden:           bool      artificial span mask, 자연 결측과 다름
```

홈/원정 각각10개 구조 값은 `cx,cy,width,length,width_p90_p10,length_p90_p10,radius,bbox_area,density_per_m2,nearest_teammate_m`이다. 이후 GNN의 `z_H,z_A`도 별도로 유지해 owner evidence 모듈에 연결해야 한다.

다음 단계는 이 계약 위에 half/split-aware151프레임 window와 provenance-aware corruption을 먼저 추가하는 것이다. 그래프 `edge_index/edge_attr`, Graph Encoder·두 Bi-LSTM·fusion·loss·clustering은 **아직 구현하지 않았다**. 현재는 frame-local 입력이라 derivative halo가 없으며 hidden-sentinel 테스트가 통과한다. 향후 차분·보간·스무딩을 추가하면 dependency halo까지 다시 검사해야 한다.

## 10. 미해결 한계

* 팀 소속 오분류·ID swap·실제 duplicate를 메타데이터만으로 확정 교정하지 못한다.
* 강한 quality 제외는 경합처럼 tracking이 어려운 장면을 더 많이 제외할 수 있다. coverage가 경합 recall은 아니다.
* 과거1초 집합 변화는 선수 구성 변화와 실제 움직임을 완전히 분리하지 못한다.
* bounds 밖 위치는 합법적인 선수 위치일 수 있다. 현재 지수 정책은 보수적이며 실제 오류 확률을 학습한 정책이 아니다.
* 폭/길이·bbox density는 GK·시야·누락에 민감하다. quantile도 완전한 해결책이 아니다.
* 정답 없이 실제 경합 F1이나 H/A/D spellF1을 산출하지 않았다. 지수의 정답성·확률 보정·점유 개선은 미검증이다.

기존 보호 대상 소스·baseline과 원본 ATD의 보존을 별도 확인했다. 생성된 경기 파생 파일은 `data/`의 Git ignore 적용을 확인했으며 push하지 않았다.
