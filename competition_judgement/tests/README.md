# Stage1 합성 테스트

`test_stage1.py`는 경기 자료 없이 합성 좌표로 거리 경계·고유 접촉 수·연결 사슬·구역 경계·순열/팀 교환·불충분 관측·노이즈/누락·causal 지속 조건·시간/분할 경계·Train-only threshold·199,319행 정렬·입력 allowlist·산출물 보존을 검사한다.

프로젝트 루트에서:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/competition_judgement/src:$PWD/MTAH_Relational_Starter:$PWD" python -m pytest -q -p no:cacheprovider MTAH_Relational_Starter/tests competition_judgement/tests
```

실제 경기 검사는 합성 pytest와 구분해 실행 보고서와 로컬 verification JSON에 기록한다.
