# Naive baseline 실행 결과

- 입력 방식: `atd`.
- Fresh nearest-player inference from tactical ATD; no ground truth used.
- 전체 제출: 199,319프레임. 실제 평가: 7,500프레임 (11185–18684, 00:07:27.400–00:12:27.400; 끝 시각 제외).
- 기본 모델: `baseline_smooth50`. 주최 측 고정 기준이며 공개 GT로 선택·튜닝하지 않음.
- bootstrap 1000회; 구간은 공식 scorer의 5–95 백분위(90% 구간).

| 모델 | spellF1 | macroF1 | 정확도 | 점유 구간 수 | 전환 F1 ±25프레임 |
|---|---:|---:|---:|---:|---:|
| baseline_smooth125 | 0.4242 | 0.5812 | 0.7275 | 16 | 0.1667 |
| baseline_smooth50 | 0.3256 | 0.6207 | 0.7132 | 26 | 0.0816 |
| baseline_smooth25 | 0.2868 | 0.6200 | 0.6912 | 36 | 0.2034 |
| baseline_raw | 0.0378 | 0.5972 | 0.6512 | 131 | 0.1410 |

공개 정답 점유 구간은 17개. 기본 모델 오답은 2,151프레임이며 진단용 연속 오답 구간은 28개다.

## 기본 모델의 클래스별 결과

| 라벨 | precision | recall | F1 | 정답 프레임 수 |
|---|---:|---:|---:|---:|
| H | 0.6972 | 0.8804 | 0.7781 | 3454 |
| A | 0.7496 | 0.7738 | 0.7615 | 2577 |
| D | 0.6569 | 0.2138 | 0.3225 | 1469 |

## 해석과 파일

- 이 점수는 공개 5분에만 해당한다. 전체 경기 성능이나 모델 우열로 일반화하지 않는다.
- 기본 모델은 최근접 선수 팀에 H/A를 부여하고 ±50프레임에서 H/A만 투표한다. 추적 답이 없을 때 D로 처리하므로 실제 데드볼과 관측 누락을 구분하지 못한다.
- 공개 정답은 파이프라인 검증에만 사용하며 학습·임계값·윈도 선택에 사용하지 않았다.
- `scoreboard.csv`: 공식 평가 수치. `reports/`: 공식 JSON과 전환 CSV.
- `diagnostics/frame_diagnostics.csv`: 전체 프레임의 GT·4개 예측·시간. `primary_correct=-1`은 GT X로 평가 대상이 아님.
- `diagnostics/public_frame_diagnostics.csv`: 실제 평가 대상 7,500프레임만 추출.
- `diagnostics/error_segments.csv`: 기본 모델의 연속 오답(동일 GT/예측 쌍) 구간.
- `diagnostics/confusion.csv`, `per_class_metrics.csv`, `label_counts.csv`: 혼동·클래스·분포.
- 모든 시간은 25fps 전술 영상 시작 기준이며 경기 시계가 아니다. MP4의 편집·프레임 일치 여부를 이번 실행에서 디코딩으로 확인하지 않았다.
- ATD TXT/XML의 실제 좌표에서 최근접 팀 관측을 새로 생성했다. 속도는 사용하지 않았다.
