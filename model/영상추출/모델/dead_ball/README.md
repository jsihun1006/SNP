# dead_ball: 선수 움직임 기반 데드볼 판정 (Phase 1)

전술 ATD에서 **프레임마다 데드볼 여부**를 만든다. H/A 점유는 판단하지 않는다.
Phase 2 점유 모델이 D 상태를 정할 때 입력으로 쓰라고 만든 특징 파일이다
(예: model1의 "긴 공 누락 → D" 대신 `dead`를 사용). 이렇게 쓰는 Phase 2 모델이
`model/점유율계산/계산알고리즘/model1_dead_ball/`이다.

## 방식

```text
선수 특징(팀 단위 속도·대형·라인까지 거리 등, 공 미사용)
→ LightGBM: 공개 DFL/IDSSE 7경기로 학습(대회 경기 라벨 사용 안 함)
→ 2상태 비터비 평활화(전환 벌점 40) + JSH 보정 ⑤(느린 빌드업: 패스·수신·공 동행 구간은 데드 진입을 어렵게)
  + ①(데드볼 시작 늦음) + ⑥(짧은 데드볼 누락). 라인을 넘는 순간은 데드 진입 벌점 0
→ 확실한 라인 아웃은 공이 안에서 다시 움직이고 모델이 인플레이로 볼 때까지 무조건 데드
→ 데드볼 끝 앞 5초 안에 재개(공이 선수 옆에서 빠르게 출발)가 보이면 끝을 그 재개로 앞당김(구간 수는 그대로)
→ 페널티킥(선수 배치로 감지, 공 불필요) 순간 데드볼을 끊고, 뒤에 킥오프가 오면 골로 보고 킥오프까지 데드
→ 골키퍼가 공을 쥔 구간은 인플레이(공이 골키퍼 옆에서 사라졌다가 재개 자리가 아닌 곳에서 나타남 등)
→ 데드볼 시작을 라인 아웃 프레임에 맞춤
→ 공개 킥오프(FAQ.md): 후반 110808–113741은 D
```

- 공은 라인 아웃 판정에만 쓴다. **공이 안 보인다는 이유만으로 D로 보지 않는다.**
- 학습·검증 데이터의 공 상태(DFL은 데드볼 동안 공 고정, Metrica 샘플은 공 삭제)가 정답을 드러내므로
  모델 입력에는 공 특징을 넣지 않았다.
- 코드는 저장소 `src/`에 있고, 이 폴더는 진입점과 형식 검사만 둔다. 설정과 근거는 `src/config.py`.

## 근거 (정답이 있는 공개 데이터)

| 검증 (2026-11-17 최종 파이프라인) | 인플레이 구간 F1* |
|---|---|
| DFL 7경기, 경기 하나씩 빼고 학습 (`src/eval_dfl_current.py`) | **0.904** |
| DFL, 골키퍼 쥠 구간 공을 지운 조건(대회처럼 공이 안 보임) | 0.900 |
| Metrica 샘플 3경기 (학습에 안 씀) | 0.832 |
| SkillCorner 20경기 (학습에 안 씀, 10 fps를 늘려 씀) | 0.761 |

단계별 근거와 뺀 시도는 저장소 `LESSONS.md`.

\* 공식 채점기의 spell 점수를 인플레이 = H, 데드 = D로 두고 계산한 값. 데드볼이 점유 구간을 얼마나 정확히 나누는지 본다.

## 실행

저장소 루트에서, 대회 데이터가 `data/`에 있는 환경으로 실행한다.

```bash
python -m pip install -r model/영상추출/모델/dead_ball/requirements.txt
python model/영상추출/모델/dead_ball/extract.py                  # src 파이프라인 실행 + 내보내기
python model/영상추출/모델/dead_ball/extract.py --skip-pipeline  # 이미 만든 결과만 내보내기
python model/영상추출/모델/dead_ball/validate.py model/영상추출/추출csv/dead_ball/<실행ID>
```

## 모델 파일 (Git 제외)

| 항목 | 값 |
|---|---|
| 파일 | `data/external/dfl/model_all_v2.txt` (LightGBM 텍스트 모델, 약 2.8 MB) |
| SHA-256 | `18a46eb5dd1e89f1057edc163e04b5f33b178a5c63123d7b4b2078724683737e` |
| 학습 데이터 | Sportec Open DFL(IDSSE) 7경기. Bassek et al. 2025, Scientific Data 12, 195. CC-BY 4.0(DFL 표기) |
| 학습 설정 | `src/config.py` (특징 세트 v2, `lgbm_params`, `sim_seed` 7) |

받는 방법은 둘 중 하나다.

1. **팀 내부 전달**: 위 파일을 받아 `data/external/dfl/`에 넣고 해시를 확인한다.
2. **직접 학습**: DFL 원본 XML 7경기 × 3파일을 figshare(doi 10.6084/m9.figshare.28196177)에서 브라우저로 받아
   `data/external/dfl/raw/`에 둔 뒤 실행한다(스크립트 내려받기는 막혀 있다). 경기 ID는 `src/dfl_load.py`의 `MATCHES`.

   ```bash
   python src/dfl_load.py all
   python src/train_dfl.py --save-final
   ```

   시드는 고정돼 있지만 운영체제가 다르면 LightGBM 결과가 아주 조금 다를 수 있다(추정). 같은 결과가
   필요하면 1번을 쓴다.

## 출력: `model/영상추출/추출csv/dead_ball/<실행ID>/` (Git 제외)

| 파일 | 내용 |
|---|---|
| `features.csv` | 프레임당 1행, 199,319행 |
| `frame_status.csv` | 프레임당 1행: `frame,time_seconds,processed,ball_seen` |
| `run_manifest.json` | 입력 파일·모델 파일 해시, 설정값, 코드 커밋, 패키지 버전 |
| `validation.json` | `validate.py` 검사 결과 |

### `features.csv` 열

| 열 | 의미 |
|---|---|
| `frame` | 전술 영상 0기반 프레임 (ATD `frame_id` = frame + 1) |
| `time_seconds` | frame / 25 |
| `graded` | 채점 구간(전·후반)이면 1 |
| `dead_score` | 모델의 프레임별 데드볼 확률(평활화 전, 0~1) |
| `dead` | 최종 판정. 1 = 데드볼. 채점 구간 밖은 항상 0 |
| `dead_source` | 데드인 구간의 출처: `model`, `model+out_start`(시작을 라인 아웃에 맞춤), `forced_line_out`, `public_kickoff`. 페널티킥 끊기·골키퍼 쥠은 구간 경계를 바꾸므로 따로 표시하지 않는다(`data/dead_balls/final_model/`의 `penalty_kicks.csv`, `keeper_holds.csv`). 인플레이면 빈칸 |

## 한계

- 선수 움직임이 주된 근거라, 공이 돌고 있는 느린 빌드업을 짧은 데드볼로 볼 때가 있다(⑤가 일부를 고친다).
- ⑤는 휘슬 직후 공을 주고받는 것도 빌드업으로 보고 진행 중인 데드볼을 몇 초 끊을 수 있다.
- 공이 안 보이는 일반 골 뒤 세리머니 중에는 데드볼이 잠깐 끊길 수 있다(페널티 골은 킥오프까지 묶는다).
- 2D 트래킹 흔들림으로 라인 근처 공이 잠깐 밖으로 찍히면 가짜 라인 아웃이 될 수 있다(`dead_source = forced_line_out`).
- Metrica 기준으로 데드볼을 실제보다 4~5%p 넓게 잡는다. DFL 숫자는 대회 성능보다 낙관적이다.
- 자세한 시도·폐기 기록은 저장소 루트 `LESSONS.md`.
