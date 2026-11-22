# transition 작업 보고 / 2026-11-15

## A. Git 정보

- 저장소: https://github.com/jsihun1006/SNP
- 원본 기준: `origin/KKJ`, `17f527eb31c4bc2a8f9dc61d0c08e93cd1173808`.
- 작업 브랜치: `transition`, 별도 `/private/tmp/mtah2027-transition` worktree. 원본 KKJ HEAD와 수정/미추적 상태는 byte 기준으로 동일하다.
- 원본 코드/증거 보존 commit: `d19600b462fdae2530deb65bf3c9ab54c733a723`.
- 후속 commit은 Windows launcher/adapter/의존성/문서 작업이다. 최종 tip은 `git rev-parse HEAD`, 원격은 `git ls-remote origin refs/heads/transition`으로 확인한다. Push 성공 여부와 최종 원격 SHA는 작업 종료 응답에 보고한다.
- Push 대상: origin/transition. upstream·main·KKJ에는 Push하지 않는다. force/reset/clean/amend/rebase를 사용하지 않는다.
- 브랜치: https://github.com/jsihun1006/SNP/tree/transition

## B. 개발 이력 복원

모든 소스 prefix는 `competition_judgement/src/competition_judgement/`다. 당시 Git commit은 없으며 현재 로컬 소스를 보존했다.

| 버전 | 대표 코드 | 복원 상태 | 당시 누적 / 현재 해당 파일 테스트 | 주요 변경·최종 여부 | 근거 |
|---|---|---|---|---|---|
| v1 | cli.py, detector.py | 최종 수정본 모듈 확보, 초기 byte 미복원 | 187 / 31 | 최초 후보·causal 5 frame; 선행 | 최종 source hash 6/6, 실제 9,936/446 확인 |
| v2-1 | local_clusters.py, local_features.py | 모듈 확보 | 213 / 26 | 3/5m 혼합 다중 군집; 선행 | hash 기록 21/21 |
| v2-2 | temporal_source.py, temporal_features.py | 최종 모듈 확보, 초기 byte 미복원 | 233 / 20 | ±50 Anchor·시간 관계; 선행 | 최종 hash 기록 16/16 |
| Stage A | missingness_diagnostics.py | 모듈 확보 | 377 / 25 | 결측 원인·관측 지원; 선행 | hash 기록 12/12 |
| Stage B | evidence_features.py | 핵심 확보, 초기 진단 일부 미복원 | 404 / 27 | Full/Partial Evidence; 선행 | hash 기록 10/14, 불일치 명시 |
| 개편 B/B.1 | pattern_values.py, pattern_profiles.py | 모듈 확보 | 440 / 36 | 지원/순서·A–D 분리; 선행 | hash 기록 7/7 |
| B.2 | local_temporal_features.py | 모듈 확보 | 477 / 37 | 같은 scope Rise→Fall; 선행 | hash 기록 6/6 |
| v2-3 | run_candidate_v23.py, candidate_decision_v23.py | 실제 최종 판정 확보 | 522 / 45 | gate·PATTERN/LOCAL/REVIEW/NO/INSUFFICIENT; **최종** | source hash 5/5, frozen input hash 5/5, 저장 결과 정합 |

현재 기본 검사 251 passed, 관계/영상 포함 419 passed. 저장소 광역 검사는 604 passed / 14 failed이며 방송 도구의 누락 의존성·외부 모듈·기존 OpenCV 오류다. 과거 522개와 현재 suite 범위 차이는 VALIDATION.md에 기록했다. 단계별 전체 당시 tree snapshot은 없고, 모듈별 증거로 복원 가능한 범위를 표시했다.

## C. v2-3 실행

Entry point는 `python -m competition_judgement.run_candidate_v23`, 고정 설정은 `competition_judgement/configs/candidate_v23.json`이다. NumPy/pandas 기반 CPU 처리로 Torch/GPU가 필요 없다. 입력은 동결 B.1/B.2 cluster profiles, v2-1 frame/cluster, v1 frame과 별도 원시/pair/Stage A cache다. 출력은 cluster 789,005×16 / frame 199,319×39 sidecar, metadata 및 Validation 진단이다.

```powershell
$python = Join-Path $PWD '.venv\Scripts\python.exe'
& $python contest_detection/run.py v2_3 --output data/derived/competition_judgement/candidate_v23_windows_new
```

기존 출력이 없는 새 경로를 사용한다. 전체 runner는 원래 CPU 기록 52.89초이며 전처리 비용은 별도다. Windows 실행 시간·peak RAM은 미측정이다. 라이선스 입력은 clone에 포함되지 않는다. 모든 입력 경로·생성 순서·schema·주의점은 REPRODUCTION.md에 있다. 이번에는 전체 경기 추론을 하지 않았다.

## D. TranSPORTmer 통합 준비

`features.py`에 원래 함수 객체를 재사용하도록 연결했다. 공간 밀집/근접, 관측된 시간 변화/근접 지속 하한, Short/Near Rise→Fall, Evidence Profile, Missingness/관측 지원을 사용할 수 있다. 원래 source에서 metre 좌표·mask·key 계약을 유지해야 한다. m/s Approach Velocity, 선수 신원 기반 persistence/episode, 모델용 tensor batching·DFL 정렬/라벨러는 추가 개발 항목이다.

Windows PowerShell 설치/실행, CPU 규칙 의존성, 선택 관계/영상 의존성과 향후 CUDA PyTorch를 분리해 문서화했다. `.gitattributes`로 원본 byte를 보존하고 공통 launcher가 Path/cwd/UTF8/path separator를 처리한다. Windows·RTX5080 실제 인수 검사는 후속 단계다.

v2-3 규칙 판정은 TranSPORTmer 정답 라벨로 취급하지 않는다. 향후 DFL 이벤트 기반 학습·Ablation 설계를 별도로 수행하며, 이번에 모델 구현/학습은 하지 않았다.

## E. 미해결 문제

- 최초 v1/초기 v2-2 및 Stage B 일부 초기 진단 소스의 원래 byte를 찾지 못했다. 가짜 과거 커밋/버전을 만들지 않았다.
- 과거 522개 suite의 당시 root 119개 정확한 node 목록과 전체 tree snapshot은 없다.
- 광역 검사 방송 도구 14개 실패는 그대로 기록했다. 경합/관계/영상 보조 suite는 통과했다.
- 원래 전체 verifier의 >477 테스트 guard·절대경로/mtime 보존 계약은 Windows 복사 환경에 그대로 적용할 수 없다. 원본을 변경하지 않고 문서화했다.
- Windows file mapping/rename·codec·wheel 설치와 CUDA 실행은 해당 환경에서 추가 검증해야 한다.
- 다른 경기/DFL용 입출력 일반화·선수 identity·label provenance·미래 누출/시간 정렬·Ablation은 향후 작업이다.
