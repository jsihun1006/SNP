# Unpack the match data here

The competition match assets are gated. See [DATA.md](../DATA.md) for how to
enrol, accept the terms, and get the download link.

Unpack the bundle here, keeping the names it arrives with:

```
data/
├─ tactical/                                  the evaluation reference
│  ├─ BAR-ATH_tactical.mp4                    row i of a submission is frame i of this
│  ├─ BAR-ATH_tactical_atd.xml                tracking metadata
│  ├─ BAR-ATH_tactical_atd.txt                tracking, players and ball
│  ├─ BAR-ATH_tactical_homography.csv         one 3x3 matrix per frame
│  └─ BAR-ATH_tactical_smart_tagging.xml      638 tactical phases and set pieces
└─ broadcast/                                 optional, NOT synchronised to tactical
   ├─ BAR-ATH_broadcast.mp4
   ├─ BAR-ATH_broadcast_atd.xml
   ├─ BAR-ATH_broadcast_atd.txt
   └─ BAR-ATH_broadcast_homography.csv
```

One folder per recording, because the question that matters most about any of these files is which
timeline it is on. `games.py` derives every path, so nothing else in the repository spells one out:

```python
from games import asset_paths
asset_paths()                      # tactical, the default
asset_paths(source="broadcast")
```

If a filename ever changes, `games.py` is the only edit.

Prefer to keep the data somewhere else? Point `MTAH_DATA_DIR` at it and nothing else changes:

```bash
export MTAH_DATA_DIR=/mnt/big-disk/mtah2027       # macOS / Linux
$env:MTAH_DATA_DIR = "D:\mtah2027"                # Windows PowerShell
```

## DFL 학습 데이터와 이전 실험

| 경로 | 내용 |
|---|---|
| `external/dfl/raw/` | 독일 DFL 7경기 원본 XML 21개의 무손실 압축본(`.xml.gz`). 경기별 트래킹·이벤트·경기 정보 |
| `external/dfl/` 및 `external/dfl/features/v2/` | DFL 변환 데이터와 선수 특징 캐시 |
| `external/dfl/model_all_v2.txt` | 공개 DFL 데이터로 재학습한 LightGBM 가중치. Git 추적 |
| `dead_balls/model_v2/`, `dead_balls/final_model/` | BAR–ATH 데드볼 모델 출력과 라인아웃 보정 후 결과 |
| `experiments/hsj-signal-pilot/` | 휘슬 파일럿 스크립트·후보 CSV·집계 JSON |

- **DFL 출처** · Manuel Bassek, Robert Rein, Hendrik Weber, Daniel Memmert (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*. [Springer Nature Figshare](https://springernature.figshare.com/articles/dataset/An_integrated_dataset_of_spatiotemporal_and_event_data_in_elite_soccer/28196177) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 원본 XML은 gzip 압축만 했으며 내용은 동일하다.
- **DFL 불러오기** · 저장소 루트에서 `python src/dfl_load.py all`을 실행하면 압축 파일을 직접 읽어 로컬 Parquet를 생성한다. 이미 내려받은 `.xml` 파일도 그대로 사용할 수 있다.
- **데드볼 재현 결과** · 107구간, 채점 프레임의 48.1%를 데드볼로 예측. 확실한 라인아웃 7건 강제, 시작 2건 보정. 예측 집계이며 대회 정답 기준 점수는 미확인.
- **휘슬 파일럿** · 후보 83개. HSJ 예측 시작 ±2초에 6개, 종료 ±2초에 7개. 독립 휘슬 정답은 미사용.
- **로컬 보관** · DFL 비압축 XML·캐시와 대회 경기 자료·프레임별 실험 출력은 Git 제외. 대회 자료의 재배포 조건은 [RULES.md](../RULES.md) 참고.

## You do not need any of this to run the scorer

The evaluation reads CSVs only. `python evaluation/run_evaluation.py` works on a fresh clone with
this folder empty, and scores the four baselines that ship with the repository.
