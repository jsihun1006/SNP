# 전환·경합 연구 기록

`transition`의 `0739a8f`에서 연구 소스와 설정을 이관했다. 실제 프레임별 자료와 이전 작업환경의 생성 산출물은 로컬 보관본에서 찾는다.

| 프로젝트 | 내용 |
|---|---|
| [MTAH_Relational_Starter](MTAH_Relational_Starter/README.md) | 선수 관계 그래프, 공간·시간 특징과 robustness 실험 |
| [competition_judgement](competition_judgement/README.md) | 경합 후보와 전환 전후 근거 검증 |
| [contest_detection](contest_detection/README.md) | 휴대 가능한 특징 계산·입력 생성 어댑터와 버전별 실험 |

레거시 명령은 이 디렉터리를 실행 루트로 가정한다. 이 컴퓨터의 로컬 `data` 연결과 `games.py`는 JSH의 공통 경로를 가리킨다. 새 체크아웃에서는 `MTAH_DATA_DIR`와 실험의 명시적 캐시 경로를 설정한다. DFL H/A 학습의 현재 진입점은 [H/A 실험 안내](../../experiments/possession/transportmer/ha_oof/README.md)에 있다.
