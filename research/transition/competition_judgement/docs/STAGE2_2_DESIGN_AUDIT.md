# v2-2 사전 설계 감사

2026-11-09. 요청한 v2-1 설계/실행 보고서, Stage1 기록, 현재 군집·FrameStore·QA API를 확인했다. 기존213개 합성 테스트를 재실행해213 passed in21.20s였다. 보호 소스·문서187개 SHA256과 원본/기존 산출물8,276개 size/mtime을 시작 시 기록했다. v1/v2-1의 코드·설정·결과는 변경하지 않는다.

## 데이터와 범위

기존 Frame CSV199,319행과 Long CSV789,005행의 키·출처 hash를 검증해 사용한다. raw parser나 선수 ID 연결을 새로 만들지 않는다. 기본±50프레임,101개 관측이며25/75 등으로 바꿀 수 있다. 같은 half·split·contiguous-record run 안에서만 참조하고, period0은 시간 분석 범위 밖으로 남긴다. 없는 시점은 center sentinel로 gather한 뒤 support=False로 완전히 제외하므로 다른 블록의 값을 잠깐 참조하는 경로도 없다.

## 기본: 고정 공간 Anchor

현재 군집 중심c에 `R=max(6m, 현재 diameter+3m)`인 고정 원을 둔다. 현재 component의 모든 관측을 포함하지만 큰 사슬에서는 넓은 지역을 포함한다는 한계가 있다. 각 과거/미래 시점에 원 안의 관측을 다시 선택한다. 원 밖으로 이동한 집단과 미관측을 동일 인물 추적으로 구분하지 않는다.

3m/5m cross-edge와 고유 접촉 관측 수는 원래 inclusive hard threshold를 유지한다. 보조 연속 강도는 `max exp(-d_HA/3m)`이며 mean pair kernel도 별도 보존한다. 군집 ID·멤버를 따라가는 강도가 아니라 현재 공간의 최근접 관계다. 원의 경계 선택은 여전히 hard하므로 kernel이 모든 불연속을 없애는 것은 아니다.

각 시점의 mathematical observation-valid는 원 안에 양 팀 관측이 있을 때다. reliable-valid는 여기에 기존 strict QC, 현재 strict QC, 현재와 같은 전역 팀별 관측 수를 요구한다. 현재 인원과 같아도 사람이나 소속 오류가 없다는 보증은 아니다. 원 안 인원 변화도 별도로 기록한다.

observed 평균·표준편차·과거/미래 차분은 실제 관측 가능한 값만 기술한다. 기본 pattern score는 전체101개가 reliable-valid인 경우에만 내보낸다. partial 지원은 별도80% side flag다. 이 기준을 실제 결과에 맞춰 완화하지 않는다.

접근/미래 변화=현재−과거 평균/미래 평균−현재, 단기/장기 접근은 과거 절반별 비교다. 접근→집중→분리 score=`clip(S0−past_mean,0,1)*clip(S0−future_mean,0,1)`. 안정적 근접 score=hard contact occupancy × soft window mean × (1−soft range). 확률·정답 라벨·경합의 필수 조건이 아니다.

현재 주변에 중심이 들어오는 v2-1 component 수와 최대 N의 변화도 별도로 기록한다. 고정 원 안 raw contact 수와 완전히 같은 통계는 아니다. static 이동평균 기준은 원래 frame-level v2 최대 강도의 동일 run 안 평균이며 support를 기록한다.

근접 지속은 현재부터 양방향으로 연속 reliable+hard-contact인 구간의 frame-bin 길이(N/25)로 기록한다. 관측 부족/인원 변경/지역 이탈은 unknown으로 끊고 onset/end 검열을 표시한다. 양 팀이 여전히 보이면서 hard 관계가 없어졌을 때만 관측된 지역 관계 종료라고 부른다. 군집이나 볼 경합의 종료 정답은 아니다. 재접근은 미래 전체가 reliable일 때만 관측 가능한 off→on을 센다.

## 보조: 보수적 인접 군집 대응

한 frame 차이에서 중심 이동≤2m, 관측 수 비율≥0.5, 대리 원판 IoU≥0.1을 후보 연결로 둔다. 원판 반경은 max(1m,diameter/2)이며 실제 멤버의 공간 overlap 보증이 아니다. 양끝 strict QC·전역 인원 일치·same run을 요구하고 서로 후보가 정확히 하나인 reciprocal-unique인 경우만 대응을 기록한다. 병합/분리/경쟁은 ambiguous로 유지하며 one-to-one 최적화를 강제하지 않는다. 대응들을 이어 영구 track을 만들지 않는다.

## 검증·선택 제한

합성의 접근/지속/분리/재접근·dropout·merge/split·경계/미래 sentinel·순열/팀 교환을 먼저 검증한다. 실제 robustness는 기존12×151 Validation support와 seed2027이다. 기본±50의 완전한 context를 가진 각 support 중앙51개 시점만 clean spatial probe로 고정한다. 교란 뒤 cluster ID를 같은 집단으로 강제 매칭하지 않고 공간 anchor를 고정해 비교한다. 별도로 redetection의 adjacent-match 분포를 기록한다.

observed-mask가 같은 표본, full-reliable이 양쪽 모두 성립하는 표본을 구분해 비교한다. dropout으로 해석 가능한 표본이0개가 되어도 임계값/품질을 재선택하지 않는다. Test 시간 블록은 전체 deployment 출력에 포함하지만 특징/설정 선택에는 사용하지 않는다. GT·볼·Smart Tagging·영상 수작업 판단·신경망·최종 후보 튜닝·Push는 사용하지 않는다.
