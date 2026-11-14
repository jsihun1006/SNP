# TranSPORTmer 특징 재사용 계약

`contest_detection/features.py`는 다음 기존 함수 객체를 그대로 re-export한다. 계산 분기, threshold, 단위, dtype, 출력은 변경하지 않는다. 외부 라이브러리는 NumPy/pandas이며 기본 feature import는 torch가 필요 없다. `mtah_relational`과 `competition_judgement` import 경로를 설정한다.

```powershell
$root = (Get-Location).Path
$paths = @((Join-Path $root 'competition_judgement/src'), (Join-Path $root 'MTAH_Relational_Starter'), $root)
$env:PYTHONPATH = $paths -join [IO.Path]::PathSeparator
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -c "from contest_detection.features import extract_local_frame, local_features; print(extract_local_frame.__module__, local_features.__module__)"
```

Python에서는 같은 함수 이름을 그대로 가져온다.

```python
from contest_detection.features import extract_local_frame, profiles, local_features
# frame: mtah_relational.data_quality.PlayerFrame (xy_m in metres; teams 0/1)
# pitch: (length_m, width_m); config: original local_clusters_v2_1.json
# summary, cluster_rows, graph = extract_local_frame(frame, pitch, config)
```

| 요청 특징 | 실제 기존 함수/모듈 | 계약·제약 |
|---|---|---|
| Local Density | `local_features.component_features`, `extract_local_frame` | cross_pair_density=E/(H×A), edge_density=2E/[n(n−1)], contact strength. 사람/m²와 구분. 원래 `mtah_relational.features.geometry_features`, `zones.zone_metrics`의 면적 밀도는 별도 기능 |
| Cross-team Proximity | `local_clusters.build_mixed_graph`, `component_features`, `TemporalSource.anchor_series` | 3m/5m edge, nearest distance, kernel/contact. 현재 군집과 고정 Anchor의 soft 지표를 구분 |
| Approach Velocity | `temporal_features.summarize_anchor_series`, `pattern_values.equations`, `local_temporal_features.local_features` | 기존 것은 거리/soft 변화 및 lag 방향. **m/s 속도 함수 없음**. 향후 Δt·same-pair/identity·결측 미분 계약을 정의해야 함; 이번에 새로 구현하지 않음 |
| Cluster Persistence | `temporal_features.contiguous_duration`, `temporal_matching.reciprocal_unique`, B.2 `local_features` | 관측된 근접의 연속 prefix/하한·검열 및 기하 대응만 존재. **영구 선수/군집 identity와 episode persistence 없음** |
| Rise→Fall | `local_temporal_features.local_features`; Full Pattern은 `pattern_profiles.profiles` | 같은 Anchor의 같은 Short 또는 Near scope. 완전 reliable+구성 조건. Short의 Rise와 Near의 Fall을 섞지 않음 |
| Evidence Profile | `pattern_values.feature_profiles`, `pattern_profiles.profiles` | P1–P4/N1–N3 값·상태/이유와 A–D obs/seq/outcome. `without_sequence`는 순서 정보 없으면 UNKNOWN. 최초 버전은 `evidence_features.compute_evidence` 별도 보존 |
| Missingness / Observation Confidence | `quality.quality_fields`, `temporal_features.summarize_anchor_series`, `local_temporal_features.support_summary`, `scope_reason` | 관측 mask·strict QC·gap·인원 변화·COMPLETE/PARTIAL/UNKNOWN. 통계적으로 보정된 confidence 확률은 아님 |

실제 함수 이름은 source를 기준으로 하며 독립 velocity/persistence 함수를 존재하는 것처럼 만들지 않는다. `anchor_series`는 TemporalSource 메서드이며 `features.py`에서 re-export하지 않는다. 호출 전에 source/cache 초기화가 필요하다.

## 입력/출력 경계

- 공간 함수: `(N,2)` metre 좌표와 0/1 팀 코드 또는 원래 PlayerFrame. 원시 46개 slot은 영구 선수 ID가 아니다. 결측/부분 NaN/Inf/bounds/duplicate QC를 통과시켜 기존 policy를 유지한다.
- 시간 함수: 원래 Source가 만든 dict의 `[batch,101]` 배열. `soft`, `nearest_m`, `observed`, `reliable`, `record_support`, `strict`, `global_count_same`, `anchor_n_home/away`, radius별 edge/contact, component/path/matching metadata 등이 필요하다. `TemporalSource` → `EvidenceSource` → `PatternSource` → `LocalTemporalSource`가 공급하는 서로 다른 계약을 source와 합성 fixtures에서 확인한다. B.2는 local path 배열도 요구한다.
- 설정: 원래 JSON을 기능별로 사용한다. `local_features(s,radii,config)`의 `radii`는 batch별 radius 배열이며 frame 전체의 단일 값으로 임의 대체하지 않는다.
- frame key는 0..199318, 군집 key는 `(frame,radius_m,cluster_id)`다. 다른 radius/frame/Anchor의 Evidence를 합해 새 정답을 만들지 않는다. NaN/UNKNOWN은 zero/negative로 치환하지 않는다.
- temporal functions는 좌표만 주면 완성되는 단일 encoder가 아니다. 입력 준비 계층과 학습용 batch/tensor 변환은 아직 추가 개발이 필요하다. NumPy→Torch 변환 시 key·mask·dtype·관측 지원을 함께 전달해야 한다.

## 향후 학습 설계에 남기는 항목

v2-3 판정/score를 TranSPORTmer target label로 사용하지 않는다. 미래를 포함하는 offline feature는 prediction 시점·타깃과의 누출 여부를 먼저 검토해야 한다. 향후 DFL 이벤트 기반 label 정의, 좌표/시간 정렬, identity, 학습 split과 leakage 차단, sampling, feature normalization, unknown mask, label provenance를 별도로 구현한다.

Ablation Study는 Local Density / Proximity / 시간 변화 / 관측 지원 / Rise→Fall / Evidence를 분리해 평가하도록 추후 설계한다. 아직 실험·성능 결과는 없다. 근접/마킹/동반 이동과 실제 볼 경합은 현재 선수 좌표만으로 확정되지 않으며, DFL 이벤트와 현재 경기의 입력 규격도 자동 호환되지 않는다.
