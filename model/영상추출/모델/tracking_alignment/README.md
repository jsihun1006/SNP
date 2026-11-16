# ATD 프레임 정렬·팀 이동 진단

제공된 tactical/broadcast ATD를 서로 정렬하고 선수 이동을 시각화하는 도구다.
영상에서 좌표를 새로 추출하는 검출기나 H/A/D 점유권 예측 모델은 아니다.
`KKJ`의 `a1527a0`에서 추가된 코드를 이 폴더로 옮겼으며 정렬·지표 계산식은 유지했다.

## 위치와 입력·출력

| 구분 | 위치·역할 |
|---|---|
| 코드·환경·설명 | `model/영상추출/모델/tracking_alignment/` |
| 원본 입력 | `data/tactical/`, `data/broadcast/`의 ATD TXT/XML과 tactical Smart Tagging XML. 다른 입력 루트는 `MTAH_DATA_DIR`로 지정 |
| 정렬·초기 라벨 | `model/영상추출/추출csv/tracking_alignment/sync/` |
| 전체 이동 벡터 영상·프레임 대응 | `model/영상추출/추출csv/tracking_alignment/full_movement_vectors/` |
| 팀 이동 지표 CSV.gz·영상·그래프 | `model/영상추출/추출csv/tracking_alignment/team_motion_structure/` |
| 모델별 결과 진입점 | `model/모델별결과csv/tracking_alignment/` → 위 결과 폴더 |
| 합성 테스트 | 저장소 루트 `tests/test_*.py` |

`outputs/`도 같은 결과 폴더로 이어지는 상대 심볼릭 링크다.
이 도구의 기본 출력은 기존 `data/derived/`에서 위 `추출csv` 경로로 변경했다.
결과 폴더는 `.gitignore`만 Git에 포함하며 좌표·라벨·HTML·영상·그래프는 로컬에 보관한다.

## 실행

저장소 루트에서 실행한다. 원본 데이터와 기존 추출 결과는 수정하지 않는다.

```bash
python -m pip install -r model/영상추출/모델/tracking_alignment/requirements.txt
python -m unittest discover -s tests -v
python model/영상추출/모델/tracking_alignment/final_tracking_alignment.py
python model/영상추출/모델/tracking_alignment/view_all_movement_vectors.py
python model/영상추출/모델/tracking_alignment/view_team_motion_structure.py
```

첫 명령은 기존 CPython 3.11.17 검증 환경의 의존성을 고정한 파일을 사용한다.
실제 정렬·영상 생성에는 두 소스의 원본 ATD와 필요한 메타데이터가 있어야 한다.
폴더 정리 작업에서는 합성 테스트와 실행 경로를 검증하며 실제 경기 분석을 재실행하지 않는다.

## 구성

- `final_tracking_alignment.py`: 공 궤적·팀별 배치로 전·후반 기준점 검색, 대응 CSV와 초기 라벨 생성.
- `find_joint_tracking_match.py`, `find_layout_match.py`, `align_half_starts.py`: 정렬 검색과 보조 도구.
- `compare_tracking_player.py`: 두 ATD 소스 비교 플레이어·영상 내보내기.
- `view_all_movement_vectors.py`: 전체 non-X 구간의 좌표 차분 이동 벡터 영상.
- `team_motion_metrics.py`, `view_team_motion_structure.py`: 팀 평균 이동·수렴·압축/확산 진단.
- `tracking_utils.py`: 원본 ATD 입출력·팀 매핑·마스크·공통 수치 함수.

초기 라벨은 태그 밖 `X`, 태그 안 빈칸이다. H/A/D 정답이나 제출용 예측이 아니므로
점유율 성능 비교표에는 모델 점수를 추가하지 않는다. 공개 점유권 GT도 읽지 않는다.
원본 추적 슬롯은 안정된 개인 ID가 아니며 반별 고정 시차는 전체 영상 동기화의 증명이 아니다.

상세 실행·정렬 규칙은 [HALF_ALIGNMENT.md](HALF_ALIGNMENT.md),
이동 지표 정의와 한계는 [TEAM_MOTION_STRUCTURE.md](TEAM_MOTION_STRUCTURE.md)를 참고한다.
