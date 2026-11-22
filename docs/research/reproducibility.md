# SNP 모델별 재현 보고서

2026-11-23 로컬 JSH `29d6666`와 원격 브랜치 끝점을 기준으로 조사했다. 아래의 경로는 공모전 저장소 루트 기준이다. 로컬 `artifacts/organization/final-snp-plan/RUN_FILES.csv`에 확인한 파일의 실행별 위치와 유형이 있다. 모델이 같아도 학습 분할, D 마스크, 입력 타임라인, 관측/결측/복원 조건이 다른 결과는 따로 보관한다.

| 표준 파일 | 들어갈 내용 |
|---|---|
| `README.md` | 실행 단위, 준비·학습·추론·평가 명령, 입력/출력 스키마, 검증한 범위 |
| `parameters.json` 또는 실행기가 실제 읽는 설정 | 기본값과 적용값, seed, 하이퍼파라미터, FPS, 좌표 단위, 후처리와 선택 기준 |
| `split.json` 또는 manifest 내부 분할 | 경기·하프별 train/validation/test, 마스크 출처, 초기 가중치가 본 경기 |
| `run_manifest.json` 또는 기존 `manifest.json` | 소스 커밋, 외부 코드 버전, 패키지 버전, 플랫폼, 입력과 출력의 상대 경로·해시 |
| `summary.csv`, `by_match.csv`, `by_fold.csv` 또는 `results.json` | 기록된 집계 지표, 평가 범위, 조건·폴드별 결과, 단계별 실측 시간 |
| 로컬 `weights/`, `predictions/`, `cache/` | 실제 모델·프레임 예측·입력 배열. 프레임 번호, FPS/하프, 라벨 스키마, SHA-256 기록 |

기존 실행 기록에 들어 있는 경로나 해시는 수정하지 않는다. 현재 파일 위치를 새 자산 인덱스나 README에 연결한다. 가중치와 결과가 모두 있는 경우에도 같은 환경에서 추론 또는 재학습까지 확인한 범위를 따로 표시한다.

## JSH · 데드볼

| 모델 / 실험 | 파라미터·실행 소스 | 결과·가중치와 상태 |
|---|---|---|
| **LightGBM v2 / VIF 370** | `src/deadball/train_dfl.py`, `src/common/config.py`, `experiments/deadball/lightgbm/dfl_7fold/`의 `dfl_half_7fold.json`, `lightgbm_vif50.json`, `feature_columns_370.json`, `requirements-lock.txt`, `reproduce.py`. 7경기 14하프 분할을 유지한다. 이벤트 실험은 추론 이벤트 입력과 학습만 사용하는 조건을 나눈다. | `reference/{dfl_half_7fold_comparison,dfl_event_7fold_comparison,dfl_event_training_only_7fold}/`에 `by_half.csv`, `by_fold.csv`, `summary.csv`, 실행 명세. 같은 하프 7폴드에서 v2/370의 D Spell F1 **0.8908/0.8963**, 인플레이 **0.8852/0.8818**. 대회 추론용 최종 LightGBM 가중치는 로컬 `artifacts/deadball/lightgbm/dfl_final/weights/model_all_v2.txt`. 이것과 VIF 370 폴드 모델을 구별한다. |
| **LightGBM 대회 적용·라인아웃 및 위치 규칙** | `src/deadball/lightgbm/`의 `config.json`, `extract.py`, `validate.py`, `README_lightgbm_corrections.md`, `src/deadball/{apply_competition,final_dead_balls,deadball_rule_signals,rescore_deadball_location_rules}.py`. 스무딩·정지 구간·라인아웃 조건을 모델과 별도로 기록한다. | `artifacts/deadball/lightgbm/bar_ath/`의 `frames.csv`, `segments.csv` 및 `summary.txt`. `dfl_lightgbm_corrections_20261113_01/`의 `comparison.csv`, `dead_spell_comparison.csv`, manifest. 위치·재개 신호는 `deadball_rule_signals/`의 실행별 `manifest.json`·`summary.csv`. 실제 BAR–ATH 출력의 좌표·프레임 CSV는 로컬에서 관리한다. |
| **DFL LOMO 비교 · LightGBM/XGBoost/CatBoost 및 초기 TCN/Transformer** | `notebooks/deadball/dfl_model_comparison.ipynb`와 `artifacts/deadball/comparison/dfl_lomo_20261111_01/run_manifest.json`. 선수 특징, 경기별 seed, Viterbi와 라인아웃 조건 및 패키지 버전 기록. | 같은 실행의 `model_summary.csv`와 경기별 비교표. 인플레이 Spell F1은 LightGBM Viterbi **0.8450**, 라인아웃 포함 **0.8770**. 실험 집계가 있으며 개별 후보 가중치가 모두 있다는 근거는 확인하지 못했다. |
| **데드볼 TCN 후보·그리드** | `notebooks/deadball/dfl_tcn_tuning.ipynb`, `dfl_tcn_grid.ipynb`. `artifacts/deadball/tcn/{dfl_tcn_tuning_20261111_01,dfl_tcn_grid_20261111_01}/run_manifest.json`. 4초/20초, width 24/48, Adam/AdamW, dilation·seed 17·학습률 0.002 및 최대 20에폭 기록. | `model_summary.csv`, `per_match.csv`, `inner_epoch_history.csv`, 그리드의 `nested_selection.csv`. 확인한 실행 폴더에 모델 `.pt`는 없고 `checkpoint.json`은 재개/완료 기록이다. 같은 학습을 재실행하거나 원래 가중치를 확보해야 한다. |
| **데드볼 시간 Transformer** | `notebooks/deadball/dfl_transformer.ipynb`; `artifacts/deadball/transformer/dfl_transformer_20261111_01/run_manifest.json`. 20초, width 48, head 4, FFN 96, 1블록 RoPE, seed 17, 학습률 0.002. | `model_summary.csv`, `per_match.csv`, `comparison.csv`, `inner_epoch_history.csv`. 이 실행의 인플레이 Spell F1 **0.7472/0.8046**(Viterbi/라인아웃). 해당 로컬 실행 폴더에는 저장된 `.pt`가 확인되지 않았다. |
| **TranSPORTmer 데드볼·공 복원 / Joint / oversampling** | `notebooks/deadball/{dfl_transportmer,dfl_transportmer_joint,tranSPORTmer_D,tranSPORTmer_D_oversampling}.ipynb`, `experiments/deadball/transportmer/dfl_7fold/`의 고정 환경과 manifest. 관측·가림·복원, 이벤트 사용 여부와 6보정을 각 조건으로 기록. | `artifacts/deadball/transportmer/` 아래 네 실행의 `run_manifest.json`, `model_summary.csv`, `comparison.csv`, 학습 이력과 조건별 `matches/`. 예를 들어 고정 하프 7폴드의 `observed/none` D Spell F1 **0.6904→0.8190**. 복원·무이벤트의 기존 **0.9097** 참고값을 아래 warm-start 런의 결과로 사용하지 않는다. |
| **Dead Final 1 · 경기 단위 7폴드** | `artifacts/deadball/transportmer/dead_final_1/training/{parameters.json,run_manifest.json}`. 경기 6개 학습/1개 평가, seed 17, width 128, head 16, 입력 6.25Hz·60시점, 배치 16, 최대 60에폭, 학습률 0.001, D 임계값 0.5. 보정 재적용 진입점은 `src/deadball/dead_final/reproduce.py`. | `weights/<경기>.pt` 7개와 `initialization.json`, `predictions/<경기>.csv`의 확률·원시판정 및 평가 라벨, `comparison/predictions/<경기>.csv`의 조건 비교. `experiments/deadball/transportmer/dead_final_1/reference/`에서 요약을 읽는다. D Spell F1 **0.5435** 원시, **0.8308** 6보정+라인아웃 시작 강제. 둘 다 이 경기 단위 실험 기록이다. |
| **Dead Final Model 1 warm-start · 하프 7폴드** | `src/deadball/dead_final/run_model1.py`, `requirements-model1.txt`, Dead Final 1 초기화 7개, 위 LightGBM의 고정 분할 JSON. seed 7, width 128·head 16·FFN 512, 6.25Hz→25fps, 최대 25에폭·patience 4, 복원/데드 학습률 0.001/0.0002. 평가 하프 중 초기 가중치가 본 하프를 표시한다. | `--check` 경로 검증 이력은 있지만 기본 출력 `artifacts/deadball/transportmer/model1_7fold/`의 완료 결과는 확인하지 못했다. 생성해야 할 파일은 `parameters.json`, `comparison/{summary,by_fold,by_half,fit_times}.csv`, 양쪽 모델의 refit 가중치와 프레임 CSV. |
| **Model 1 단일 홀드아웃·라인아웃 재디코딩** | `artifacts/deadball/transportmer/transportmer_model1_holdout_J03WR9_train_loss/`의 `manifest.json`, `training.json`, `fold_1.json`. 별도 IFAB·라인아웃·경계·위치 보정 폴더에 조건 기록. | `ball_restorer.pt`, `dead_best.pt`, `by_fold.csv`, `by_half.csv`, `summary.csv`, `predictions/J03WR9_H*_restored_none.{csv,npz}`. `ifab_line_out_viterbi_comparison/`의 프레임별 차이와 요약은 독립 비교로 보관한다. |
| **HSJ 휘슬 파일럿** | `src/deadball/audio/whistle.py`와 `experiments/deadball/audio/hsj_signal_pilot/README.md`. 2400–4800Hz와 경계 ±2초 기준, 원래 HSJ `segments.csv`가 비교 입력이다. | `summary.json`에 후보 83건과 경계 일치 통계, 로컬 `whistle_candidates.csv`에 시간·주파수. 독립 휘슬 정답 평가가 아닌 경계 일치 진단이다. |

## JSH · H/A 소유 및 전환

| 모델 / 실험 | 파라미터·실행 소스 | 결과·가중치와 상태 |
|---|---|---|
| **공식 naïve와 Viterbi Model 1** | `src/possession/naive/{baseline.py,run.py,run_manifest.json}`의 raw·smooth25/50/125, `src/possession/model1/{config.json,model.py,run.py,run_manifest.json}`의 관측·결측·상태전환 비용. Model 1의 긴 공 결측 기준 3초와 팀/데드 전환 비용 1.2/0.8. | `experiments/possession/{naive,model1}/reference/`의 `scoreboard.csv`, `confusion.csv`, `per_class_metrics.csv`. 전체 프레임 예측·진단은 `artifacts/possession/{naive,model1}/reference/`. Model 1의 0.330435와 smooth50의 0.325581은 공개 구간 진단이다. |
| **Video ball hybrid** | `src/possession/video_ball_hybrid/{possession,evaluate,possession_share}.py`, 같은 폴더 README의 검출 클래스·공 후보·ATD 정렬 규약. football YOLO의 공 클래스 0과 COCO YOLO 클래스 32를 섞지 않는다. | 기본 `artifacts/possession/video_ball_hybrid/reference/`에서 완료 예측/점수는 확인하지 못했다. 재현 시 선택 검출·원본 프레임·공가용 마스크, Model 1 설정, 전체 예측 및 집계·점유율을 남긴다. |
| **PathCRF trial140 5fps / 기하 보정** | `vendor/pathcrf/saved/140/args.json`, `src/possession/pathcrf/run_pathcrf_{dfl,geometry_dfl}.py`; 5fps·10초 창, J03WR9 홀드아웃, geometry·평활화 조건. 외부 체크포인트는 정확한 revision 및 SHA-256 기록. | `artifacts/possession/pathcrf/dfl_runs/{trial140_heldout,trial140_geometry}/`의 `manifest.json`, `results.json`, `J03WR9_H*.parquet`; geometry 선택표 `validation.json`, `smoothing_results.json`. 기준 저장 결과 Spell F1 **0.7250**. 아래 고정 예측 D 사용 런의 5fps **0.7181**과 평가 조건이 다르다. |
| **PathCRF 전환 head / scene 계열** | `src/possession/pathcrf/run_pathcrf_transition_{dfl,warmstart}.py`, `src/possession/scene/run_dfl_scene_transformers.py`, `experiments/possession/pathcrf/dfl/`. frozen/head, warmstart 및 full/inplay 전처리를 분리한다. | 로컬 `dfl_runs/transition_head_frozen`의 `head.pt`·`head_fit.json`, `transition_head_warmstart`의 `finetune.pt`; 두 런은 결과 JSON, validation JSON, 평가 Parquet가 있다. `scene_two_models_v1`, `scene_transportmer_124_v1`에는 `full/`, `inplay/`별 가중치와 결과가 있다. `transition_head_{v1,mps_v1,full_v1}`의 준비·벤치마크·중단 로그를 완료 결과와 구분한다. |
| **5fps→25fps 경합 refiner · DFL 학습·보정** | `experiments/possession/pathcrf/candidate25_dfl/{parameters.csv,parameters_25fps.csv,parameters_candidate25.csv,inference_parameters.json}`, `src/possession/pathcrf_refiner/`. 학습 5경기, 검증 J03WQQ, 평가 J03WR9. 25fps 100프레임, width 128·4 head·2블록·시간 2층, seed 100·배치 64·최대 40·채택 29에폭. 추론의 gate·온도·Viterbi 보정은 JSON 기준. | `artifacts/possession/pathcrf/candidate25_dfl/`의 `weights/{pathcrf_trial140.pt,best.pt,last.pt}`, `inputs/J03WR9{,_base5}.npz`, `historical_run/{manifest.json,training_history.csv}`. `historical_run/correction_epoch29/`의 `holdout_comparison.csv`, `results.json`, `J03WR9_H*.csv`, 오류 진단을 보존한다. 동일 D 고정 조건에서 5fps **71.81%**, 채택 보정 **76.22%** Spell F1. |
| **Refiner 보정 실험군** | `correct_5fps.py`, `correct_5fps_v2.py`, `correct_candidate25.py`, `run_frozen_mlp.py`와 실험별 manifest/선택 파라미터. 최종 inference JSON과 혼용하지 않는다. | 로컬 `correct_5fps_v1/`, `correct_5fps_v2/`, `dfl_5train_1val_1test/`, `dfl_5fps_candidate25_40ep_es5/`, `candidate25_dfl/historical_run/frozen_mlp_boundary_v1/`의 실측표와 기존 프레임별 결과를 보존. 벤치마크는 `transition_dfl_workspace/`에 있다. |
| **채택 PathCRF · BAR–ATH SNP 재실행** | `src/possession/possession/run_csv.py`, 위 `inference_parameters.json`과 `requirements.txt`. 원래 설정 MPS, 배치 32, 스레드 8. 정렬 트래킹, 원래 SNP 입력, tactical ATD 메타데이터, 두 고정 가중치를 지정한다. | `artifacts/possession/pathcrf/candidate25_dfl/inference/SNP_final.csv`는 199,319행 `frame,label` 결과. `SNP_final.json`의 입력 해시·정렬 범위·개입 프레임 수와 NPZ/점수 캐시 보존. 원래 입력 `reproduction_inputs/SNP.input.csv`를 기록된 SHA-256과 일치하게 복구했다. 대회 정확도 점수로 해석하지 않는다. |
| **TranSPORTmer H/A OOF B/C** | `src/possession/transportmer/run_transportmer_ha.py`와 `experiments/possession/transportmer/ha_oof/README.md`. seed 2027, J03WOH 홀드아웃, 개발 6경기 OOF, 공결측 프로필, warm 5/extra 15 epoch, 배치 24. | `artifacts/possession/transportmer/ha_runs/`에서 분할, `ball_profile.json`, `benchmark.json`, J03WOY OOF 1폴드 warm/full 체크포인트와 로그를 확인. 완성된 `oof_smoothing_by_match.csv`, `locked_holdout_recipe.json`, `holdout_comparison.csv`는 확인하지 못했다. 동봉 Colab과 공개 DFL 다운로드 조건은 복구 경로다. |
| **Borghesi TCN · 5/1/1 및 입력 변형** | `src/possession/tcn/run_borghesi_tcn_dfl.py`; `artifacts/possession/tcn/borghesi_tcn_dfl/seed2027*/manifest.json`이 각 split·width·아키텍처·학습률·clean/stress·입력 선택의 기준. full, ball-only, nearest5 및 paper LR을 분리. | 각 실행의 `results.json`에서 variant별 선택 에폭과 global/D/H-A 지표, fixed-D Spell F1을 읽는다. 체크포인트와 예측 배열을 원래 실험 ID에 연결한다. |
| **Borghesi 전환 TCN · 하프 7폴드** | `src/possession/tcn/run_borghesi_transition_dfl.py`; `borghesi_transition_dfl/{seed2027,seed2027_context32}/`의 `manifest.json`. context32 transition 기준 past20 + 현재 1 + future11, width 64, seed 2027, 배치 8, 최대 30에폭, LR 0.001. | `state/`, `transition/`의 `results.json`, `fold_*/{state,transition}.pt`, 프레임 NPZ와 `comparison.json`을 보존. 5/1/1 TCN과 별도 표에서 비교한다. |

### SNP 실행의 입력을 식별하는 방법

`SNP_final.json`에 기록된 트래킹 SHA-256 `bdd017ba…`, ATD XML `327e35e3…`가 현재 로컬 파일과 일치한다. 원래 `~/Desktop/SNP.csv`는 발견되지 않았고, `공모전archive/SNP.csv`의 SHA-256은 `10ef7087…`이다. 추론 캐시의 `fallback_ha`와 최종 출력에 유지된 D 마스크로 원래 라벨을 복구했다. **복구한 원본 형식의 CSV SHA-256은 당시 헤더에 기록된 `978b55698f224c94c1442f5b1a6437bb6a7f390169b15e60837171b47cfcc11d`와 정확히 동일하다.** 아카이브와는 60,105프레임의 라벨이 다르다.

원래 결과와 별도 위치에서 재실행하는 명령이다.

```sh
base=artifacts/possession/pathcrf/candidate25_dfl
python src/possession/possession/run_csv.py \
  --tracking artifacts/tracking/snp/tracking_all_frames.csv \
  --snp "$base/inference/reproduction_inputs/SNP.input.csv" \
  --metadata data/tactical/BAR-ATH_tactical_atd.xml \
  --output "$base/recheck/SNP_final.csv" --device mps --batch 32 --threads 8
python evaluation/validate_submission.py "$base/recheck/SNP_final.csv"
```

출력 SHA-256은 기존 `f19555d6dc1cc42dcddd7d9639d5c87f4ba72013a5849152d75aca5f64a7baf1`와 대조한다. 현재 검증은 **입력 바이트 복구와 기존 출력 검증**까지다.

## JSH · 검출·트래킹

| 모델 / 파이프라인 | 파라미터·실행 소스 | 결과·검증 범위 |
|---|---|---|
| **YOLO11n / video_naive** | `src/tracking/detection/` 공용 진입점과 `src/tracking/video_naive/README.md`. 1fps 실행은 25프레임 간격, 1280px, confidence 0.15, IoU 0.70, 배치 8, COCO 0/32. `weights/yolo11n.pt`와 영상·호모그래피 해시는 `artifacts/tracking/video_naive/run_manifest.json` 기준. | 같은 로컬 폴더 및 `dense_30s/`의 `detections.csv`, `frame_status.csv`, `reference_comparison.csv`, `summary.json`, `preview.mp4`. 검출 커버리지·자동 ATD와의 일치도이며 독립 검출 정확도가 아니다. |
| **football YOLOv8x** | 같은 공용 진입점에 별도 모델 프로필. 전체 1fps 기본 640px, confidence 0.10, IoU 0.70, 배치 4, MPS·half 옵션, 클래스 0/1/2/3. 5분 dense 별도 설정. `weights/football_yolov8x.pt`와 `full_1fps`·`dense_5min`의 manifest. | 로컬 `artifacts/tracking/football_yolov8x/{full_1fps,dense_5min}/`의 검출·샘플링·ATD 비교·검증 결과. 공유 요약은 `experiments/tracking/football_yolov8x/`. |
| **Tracking alignment / ATD 병합** | `src/tracking/tracking_alignment/` 및 `games.py`, `experiments/tracking/atd_merge/bar_ath/merge_summary.json`. 하프별 오프셋, 반사, 영상·ATD 인덱스 규칙을 별도 메타데이터에 남긴다. | 정렬 진단은 `artifacts/tracking/tracking_alignment/`가 지정 저장소다. 현재 이 로컬 디렉터리에 완료 산출물은 확인되지 않았다. 실제 통합 입력은 `artifacts/tracking/snp/tracking_all_frames.csv`를 보관하고 `frame`과 `tactical_frame`의 계약을 적는다. |
| **최종 볼 보정·통합 트래킹 · KKJ 기준** | 원격 `final:submission/final_tracking/`의 `CODE_MANIFEST.json`, `ENVIRONMENT_DETECTOR.json`, `ENVIRONMENT_POSTPROCESS.json`, `VERIFICATION.json`과 `docs/unified_tracking_pipeline/pipeline.json`. 모델 revision·경기 좌표 변환·후보 선택·전/후반 결합의 22개 단계 기록. | `VERIFICATION.json`은 49개 소스 선택 및 일부 산출물 연결 해시를 검증했다. 두 ATD 입력은 당시 LFS 포인터만 있어 전체 재생성을 검증하지 않았다. 원본/보정 후보 좌표 CSV, 프레임 매핑, 전체 트래킹, 체크포인트를 로컬에 보관하고 집계와 실행 정보만 JSH 공유 후보로 둔다. |

## HSJ 및 KKJ · 브랜치별 재현 계약

| 브랜치 / 모델 | 코드·설정 | 결과·보관 기준 |
|---|---|---|
| **HSJ / LightGBM v2 + 라인아웃·재개·골키퍼 보정** | `model/영상추출/모델/dead_ball/config.json`, `src/config.py`, `src/keeper_hold.py`, `src/{apply_competition,final_dead_balls}.py`. 25fps tactical, 특징 v2, `smooth_choice`, `slow_buildup`, `final`, 골키퍼 정책을 실제 run manifest에서 확인한다. | `features.csv`의 frame·graded·dead_score·dead·dead_source, `frames.csv`, `segments.csv`와 `run_manifest.json` 및 검증 결과. HSJ의 최신 보정과 JSH의 7폴드 점수는 별도다. |
| **HSJ / model1_dead_ball·possession·possession_pathcrf** | `model/점유율계산/계산알고리즘/model1_dead_ball/config.json`과 `run.py`. Phase 1 프레임 마스크가 D 경계를 고정하고 기존 Model 1 비용으로 H/A를 고른다. `possession/parameters.json`, `possession_pathcrf/`는 별도 PathCRF 실험 기록이다. | 경로와 코드가 원격에서 확인된다. 동일 D 입력을 증명하는 해시, 전체 H/A/D 결과와 평가 요약을 확보한다. JSH 캔디데이트의 로컬 체크포인트·점수를 HSJ 실측으로 복사하지 않는다. |
| **KKJ / Tryolabs V2·V3·V4** | `starter/configs/tryolabs_classification_v{2,3,4}_{hsv,policy}.json`, `starter/run_tryolabs_v4_streaming.py`, Windows lock/setup 및 `docs/TRYOLABS_V4_FULL_MATCH_STREAMING.md`. 청크 250프레임, 원본 경기 정렬과 버전별 HSV·정책 기록. | 원본 V2 캐시의 `run.sqlite`와 완료 청크의 `chunk.json`, `tracking.jsonl.gz`, `objects.jsonl.gz`, V4 출력 `frames.jsonl.gz`, `state.json`, 상위 `manifest.json`, `checkpoint.json`, `summary.json`, `verification.json`. 1,250프레임 회귀와 전체 164,487프레임 완료 판정은 구분한다. |
| **KKJ / Broadcast GSR** | `broadcast_gsr/configs/mtah_broadcast.json`, `requirements-{core,models}.txt`, `REQUIRED_FILES.md`, `run_windows.cmd`. 기본 750프레임, detection 0.10/NMS 0.70, pitch 105.3×68m, jersey=none, refine=false. YOLOX·OSNet 지정 가중치가 실운영 필수, CLIP·SFR은 옵션 조건. | `data/derived/broadcast_gsr/`의 후보·추적 좌표, 보정/비교 집계와 검증 로그를 로컬 보관. `doctor`, `smoke`, `run`을 분리. 기존 문서는 macOS 합성 15개 통과이며 Windows·CUDA·실경기 정확도는 미검증으로 기록한다. |
| **KKJ / SSS-1** | `sss1_reimplementation/configs/{experiments.json,environment-windows.yml,constraints-windows.txt}`, `checkpoints/README.md`, `scripts/{setup_windows,run_windows}.ps1`. C(full/full), B1/B2(SAHI), A1/A2(BAAT), 학습 도메인·탐지/pose 가중치·`training_choices.json`을 연결. | `outputs/validation/{windows-ci-summary.json,provenance.json,validation_summary.json}` 및 `cli_image/predictions.csv`, `results.json`, `timing.json`. 기록된 Windows CPU 첫 실행은 **42개 통과, 3개 제외**, 공개 yolo26l BAAT 이미지 실행 증거. GPU·공모전 전체 실행 점수는 별도다. |
| **로컬 보관 / 제출 패키지와 transition** | `submission:run_submission.py`, `SNP.md`, Phase 1 ATD/볼/데드볼 모듈과 Phase 2 `model2_pathcrf/`; `transition`의 관계 모델 config와 DFL Colab 전처리·분할. 두 독립 목적에 맞춰 저장한다. | 제출 입력·출력과 선택 모델은 제출 당시 소스/기술 문서와 대조해 고정한다. 실험 결과는 동일 커밋·분할·체크포인트가 확인된 경우에만 연결한다. |

전술/중계 원본 영상·ATD, 프레임별 결과와 가중치의 실제 파일은 각 컴퓨터의 로컬 자산으로 준비한다. 결과를 새로 계산하는 실행은 새 `run_id`에 기록한다.
