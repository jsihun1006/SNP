# model1: 상태 전환 비용을 사용하는 점유 시퀀스 모델

고정된 비학습 모델이다. 원시 최근접 팀 관측을 그대로 점유 정답으로 취급하지 않고,
H/A/D 상태가 시간에 따라 이어지는 경로를 동적 계획법으로 추론한다.

## 현재 입력

`evaluation/results/baseline_raw.csv`는 정답이 아니라 주최 측 알고리즘의 출력이다.
H/A는 최근접 선수 팀 관측, D는 해당 알고리즘이 답을 얻지 못한 **관측 누락**으로 읽는다.
첫 실행은 공개 raw에서 시작했다. 이후 실제 ATD 파일을 `/Users/jsh/Desktop/data/public`에서
찾아 `data/tactical/`에 심볼릭 링크로 연결했고, 현재 실행은 실제 좌표에서 최근접 팀 관측부터 생성한다.
시퀀스 모델에는 그 팀/누락 관측만 전달한다. 속도·태그·homography·MP4는 사용하지 않는다.
최초 raw 입력 실행의 결과는 로컬 `archive/published_raw_initial/`에 보존했다.

`data/tactical/` 또는 `MTAH_DATA_DIR`에 공식 ATD TXT/XML이 있으면 최근접 팀 관측부터 다시
계산한다. 모델의 나머지 부분은 동일하다. 명시한 ATD 경로가 잘못되면 실패하며 CSV로 대체하지 않는다.

## 판단 방식

전체 경로 비용을 최소화한다.

```text
총 비용 = Σ 프레임별 관측 불일치 비용 / fps + Σ 상태 전환 비용
```

- 관측 H인데 상태 A이거나 관측 A인데 상태 H이면 초당 2의 비용.
- 관측 H/A가 있는데 D라고 판단하면 초당 2의 비용.
- 3초 미만 누락은 H/A에 중립이며 D에는 초당 1의 비용.
- 3초 이상 연속 누락은 H/A에 초당 0.5의 비용, D에는 0의 비용.
- H↔A 변경 비용 1.2, H/A↔D 변경 비용 0.8. 상태 유지 비용 0.
- 동적 계획법이 전체 영상의 최소 비용 경로를 찾는다. 앞뒤 문맥을 쓰는 오프라인 모델이다.

이 설정에서는 같은 팀 관측 사이에 끼인 약 1.2초 이하의 상대 팀 관측을 전환 비용으로
억제할 수 있다. 긴 누락도 양쪽 전환 비용을 상쇄해야 D로 바뀐다. 정확한 경계는 앞뒤 관측에 달려 있다.
누락 구간 전체 길이를 사용하므로 누락 시작 시점에는 알 수 없는 미래 정보도 사용한다.
처음 상태 비용은 동일하며 완전 동률이면 H/A/D 순으로 결정한다.

설정은 `config.json`에 명시한 공학적 가정이다. 경기에서 측정한 상수나 확률이 아니다.
공개 정답을 보고 임계값·전환 비용을 맞추거나 여러 설정을 비교해 고르지 않는다.
긴 누락이 실제 데드볼이라는 보장은 없고, 이 모델이 실제 공 컨트롤을 직접 검출하는 것도 아니다.

## 실행

저장소 루트에서 NumPy/pandas 환경을 사용한다.

```bash
python3 model/점유율계산/계산알고리즘/model1/run.py
```

개별 실행:

```bash
python3 model/점유율계산/계산알고리즘/model1/model.py
python3 model/점유율계산/계산알고리즘/model1/validate.py
python3 model/점유율계산/계산알고리즘/model1/evaluation.py
```

합성 관측에 대한 상태 추론과 전수 탐색 대비 최적성 검증:

```bash
python3 model/점유율계산/계산알고리즘/model1/test_model.py
```

다른 위치의 ATD:

```bash
python3 model/점유율계산/계산알고리즘/model1/run.py --input-mode atd \
  --atd-raw /absolute/path/BAR-ATH_tactical_atd.txt \
  --atd-meta /absolute/path/BAR-ATH_tactical_atd.xml
```

현재 Mac의 실행 가능한 Python:

```bash
/Users/jsh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 model/점유율계산/계산알고리즘/model1/run.py
```

`requirements.txt`는 저장소 루트의 기존 것을 사용한다. 검증과 일부 진단 작성은
`model/점유율계산/계산알고리즘/naive`의 기존 함수를 재사용하고, 점수는 `evaluation/`의 공식 구현을 호출한다.

## 산출물

| 파일 | 역할 |
|---|---|
| `model.py`, `config.json` | 관측 읽기·상태 추론·고정 비용 설정 |
| `validate.py`, `evaluation.py`, `run.py`, `common.py` | 형식 검증·공식 평가·실행·공통 도구 |
| `RESULTS.md` | 실제 점수·baseline 비교·해석 |
| `predictions/model1.csv` | 전체 199,319프레임의 제출 형식 예측 |
| `predictions/baseline_smooth50.csv` | 같은 입력에서 만든 비교 기준 |
| `diagnostics/inference_evidence.csv` | 관측 팀·누락 길이·상태별 비용·최종 예측; GT 미사용 |
| `diagnostics/*` | 전체/공개 프레임 비교·오답 구간·혼동행렬·클래스별 지표 |
| `reports/*`, `scoreboard.csv` | 공식 점수와 전환 매칭 결과 |
| `pre_evaluation_manifest.json`, `run_manifest.json` | 평가 전 설정·입력·코드 해시 및 예측 해시 |

공개 평가 구간의 결과는 검증 목적으로만 읽는다. 전체 경기의 정답은 없으므로
전체 경기 성능 향상은 이 결과만으로 주장할 수 없다. 비교 점수로 기본 모델을 자동 교체하지 않는다.
원본 naive와 주최 측 파일은 수정하지 않으며 생성 예측·프레임 진단은 Git에서 제외한다.

정리된 구조에서 `predictions/`, `diagnostics/`, `reports/` 등은 상위 결과 폴더를 가리키는 상대 심볼릭 링크다. 기존 실행 manifest의 절대경로는 생성 당시 경로이므로 현재 경로와 다를 수 있다.
