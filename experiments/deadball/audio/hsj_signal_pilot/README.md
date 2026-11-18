# HSJ 휘슬 후보 파일럿

음성 스펙트럼에서 후보를 뽑아 HSJ 데드볼 예측 경계 전후 2초의 겹침을 집계한 실험이다. 집계는 독립 휘슬 정답으로 평가한 점수가 아니다. 당시 후보 집계는 [summary.json](summary.json)에 있다.

```sh
python src/deadball/audio/whistle.py --hsj-segments /path/to/original/segments.csv --output artifacts/deadball/audio/hsj_signal_pilot_rerun
```

당시와 같은 비교를 하려면 원래 HSJ `segments.csv`를 제공한다. 기본 영상은 `games.py`의 전술 영상이다. 이전 작업 폴더의 고정 경로는 명시적 입력으로 바꿨다. 당시 프레임별 후보는 `artifacts/deadball/audio/hsj_signal_pilot/`에 보관한다.
