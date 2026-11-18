# Dead Final 1 · DFL 경기 단위 홀드아웃

`tranSPORTmer_D`를 6경기로 학습하고 제외한 1경기를 평가하는 실험을 7번 수행했다. 원시 결과와 보정 비교를 확인할 수 있다. [실험 기록](../../../../src/deadball/dead_final/README_DEAD_FINAL_1.md)

- [전체 요약](reference/summary.csv)
- [경기별 점수](reference/by_match.csv)
- [하프별 점수](reference/by_half.csv)
- [초기화 가중치의 학습 경기·해시](reference/initialization.json)

## 로컬 산출물

`artifacts/deadball/transportmer/dead_final_1`에 가중치, 예측, 학습 기록, 보정 비교가 있다. 이 디렉터리의 `weights/<경기>.pt`와 원래 학습 실행 `artifacts/deadball/transportmer/transportmer_d_9067_rerun/matches/<경기>/model.pt`는 SHA-256이 일치한다. 일곱 경기의 파일을 각각 하나의 로컬 데이터로 통합했고, 패키지명과 학습 실행의 경로는 하드링크로 연결했다. 재학습 저장 코드는 임시 파일을 쓴 뒤 교체하므로 기존 학습 기록과의 연결이 보존된다. 기존 체크포인트를 제자리에서 덮어쓰지 않는다.

`reference/`의 집계 수치와 초기화 명세는 로컬 패키지에도 필요한 파일과 하드링크로 연결되어 있다. 모델 가중치와 프레임 단위 예측은 로컬에 보관한다.

`training/notebooks/`의 네 노트북은 현재 연구 노트북과 정리 시점의 SHA-256이 동일하다. 학습 당시 해시를 보존하기 위해 APFS의 쓰기 시 복사 기능으로 같은 디스크 블록을 공유하는 독립 스냅샷을 유지했다. 연구 노트북을 편집해도 해당 스냅샷은 변경되지 않는다.
