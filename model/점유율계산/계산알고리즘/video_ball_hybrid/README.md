# 영상 공 검출 + ATD 기반 점유 모델

사전학습 영상 공 검출기의 결과 CSV와 주최 측 ATD 선수·팀 정보를 함께 사용한다.
선택한 공 검출은 유효한 피치 좌표 중 confidence가 가장 높은 하나다. confidence 0.10은
선택한 검출기 제작자의 권장값을 바탕으로 평가 전에 정했다. 낮은 문턱값은 오검출을 늘릴 수 있다.

1. 검출한 영상 프레임에서 최대 12프레임(0.48초) 동안 그 공 좌표를 유지한다.
2. 영상 공 좌표가 없으면 같은 프레임의 ATD 공 좌표를 쓴다.
3. 공이 있으면 같은 프레임의 ATD 선수 중 가장 가까운 선수의 팀을 선택한다.
4. 기존 `model1`의 고정 설정 H/A/D 시퀀스 디코더를 적용한다.

ATD `frame_id`는 1부터이므로 영상 frame = `frame_id - 1`로 정렬한다. 공 검출 없는
경우를 곧장 데드볼로 확정하지 않지만, model1 디코더의 D는 긴 관측 누락을 이용한
휴리스틱이다. 시간 디코더는 미래 프레임을 활용하는 오프라인 방식이다.

평가에는 **영상 전체에 균일한 1fps 공 검출 결과**만 사용한다. 공개 5분에 대해서만
고밀도 검출 결과를 섞으면 입력 조건이 다른 구간보다 유리해질 수 있으므로 별도
진단 용도로만 사용한다. 공개 정답은 점수·파이프라인 확인용이며 학습·튜닝에 쓰지 않는다.

## 실행

경로를 실제 검출 결과와 결과 폴더로 바꿔 저장소 루트에서 실행한다.

```bash
python3 possession.py --repo /path/to/공모전 \
  --detections /path/to/detections.csv --frame-status /path/to/frame_status.csv \
  --ball-class 0 --out /path/to/new_prediction_dir

python3 evaluate.py --repo /path/to/공모전 \
  --prediction /path/to/new_prediction_dir/prediction.csv \
  --out /path/to/new_score_dir
```

`prediction.csv`는 전체 199,319프레임의 `frame,label` 제출 형식이다.
`frame_diagnostics.csv`는 영상·유지·ATD·결측 출처와 최근접 팀 판단 근거를 기록한다.
`pre_evaluation_manifest.json`과 `run_manifest.json`에 고정 정책과 입력·코드 해시가 남는다.
공식 평가 결과는 `scoreboard.csv`, `RESULTS.md`, `reports/` 및 `evaluation_manifest.json`에 저장한다.
프레임 단위 데이터·좌표·영상은 Git에 공개하지 않는다.
