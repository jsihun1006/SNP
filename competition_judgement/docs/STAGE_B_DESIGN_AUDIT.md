# Stage B 사전 설계 감사

2026-11-09. 실제 문서는 root docs가 아니라 `competition_judgement/docs/`에 있다. Stage A 보고서와 controlled masking/partial policy/read-only cache, v2-1 component features, v2-2 source/features/matching·실행 보고서를 확인했다. Stage A 전체 Full25.72%, Validation reliable-P80 31.13%를 기준 통계로 유지한다. 현재 QC gate, count-change gate, ambiguous 지역 이탈/미관측을 구분한다.

기존 소스·문서210개 SHA256, 원본/기존 산출물8,344개 size/mtime을 구현 전 `/private/tmp/cj_stage_b_start.json`에 기록했다. 새 코드는 별도 `evidence_*` 모듈, 결과는 새 로컬 폴더다. 기존 v2-2와 Stage A 소스·설정·결과는 수정하지 않는다.

## 구현 전 고정한 정책

`competition_judgement/configs/evidence_stage_b.json`을 실제 Validation Evidence 계산 전에 생성했다. 결과를 보고 임계값을 재선택하지 않는다. 모든 시간 결과는 현재 strict+observed를 요구한다. Full은 원래 reliable101개를 전부 요구하며, 추가적인 특징별 정보 조건도 적용한다. Partial은 reliable의 부분 집합만 사용하고 현재 QC를 완화하지 않는다. 두 결과는 별도 namespace이며 공식 v2-2 점수를 바꾸지 않는다.

P1은 past band별20/23개·중심10·최대공백5, P2는 near-past23개·공백2·해당 scope 전역/국소 인원 일정, P3는 양쪽 near24개·공백1·전역 인원 일정, P4는 future23/22개·공백3·전역/국소 인원 일정·관측된 관계 종료 또는 완전 연속 window censor, N1은 중앙51개 완전 연속+양끝20개·공백3·인원 일정, N2는 ±10 reciprocal-unique 기하 경로 완전 지원, N3는 네 band 각각23개·공백2를 요구한다. Stage A는 이 수치의 오류 보장을 제공하지 않는다. 경험적 정책 실험으로 명시하고 통제 삭제로 검증한다.

P1은 거리/soft 접근과 near−far 평균 대비, P2는 고정 지역 내 관측 component의 최대 반경/지름 축소와 raw 접촉수/edge/density 증가, P3는 지역 군집 수/density 변화와 adjacent many-to-one/one-to-many 가능성, P4는 미래 거리/soft 분리·관측된 종료·가능한 component 확장이다. P2/P4의 국소 인원 변화는 실제 유입·이탈과 tracking 구성을 구분할 수 없어 보류한다. P3의 merge/split은 기하 후보이며 실제 멤버 관계의 증명이 아니다.

N1은 연속 근접 하한·지역 count/density 안정·접근/분리 부재, N2는 짧은 임시 기하 경로의 centroid 이동과 내부 shape 안정, N3는 soft 변동/범위/피크와 점유율이다. 영구 선수/군집 ID는 만들지 않는다. N2 대응이 불명확하면 NaN이다. Negative는 N1을 우선하고 N1이 invalid일 때만 N3를 같은 persistence family의 대체값으로 사용한다. N1+N3를 가중 합산하지 않는다. N2는 다른 family다. Positive는 유효 P 중 최소2개 평균이며 사용 항목을 기록한다. 결측은0으로 대체하지 않는다.

Full/Partial raw score는 같은 형식이지만 지원과 사용 항목이 다르므로 동등한 신뢰/고정척도 비교로 주장하지 않는다. 중복·상관과 simultaneous-high는 사전 설정한0.5 reporting band로 기술할 뿐 경합 후보를 만들지 않는다.

## 합성 기대 동작

1. 정적 근접: P1/P2/P3/P4 낮음, N1/N3 높음, N2 이동0.
2. 일정 shape의 이동: 명확한 경로이면 N2 증가; 같은 선수 동반 이동의 증명 아님.
3. 접근 후 분리: P1/P4 증가; 구성 변화/관측 부족이면 보류.
4. 접근 후 유지: P1 증가, P4 낮음. 모든 Positive 동시 발생을 요구하지 않음.
5. 같은 관측 집합의 공간 축소: P2 증가. 관측 구성이 바뀌면 보류.
6. 기하 merge/split: P3 branch 신호 증가, reciprocal ambiguity 때문에 N2 NaN.
7. 중심 인접 누락: 연속 하한 감소·N1/N2 보류; 현재 누락은 모든 시간 점수 NaN.
8. 양끝 누락: 일부 평균 Evidence만 Partial 가능; Full은 보류, feature별 정책 차이 확인.
9. 전역 count 변화: 시점/횟수 기록, 해당 offset은 reliable 제외; 현재 QC 실패와 별도 사유.
10. 대응 ambiguity: N2 NaN+reason. 합성 패턴은 실제 경합/비경합 라벨이 아님.

원래199,319축/Long키를 유지하고 frame별 max에는 항목별 출처키를 기록한다. 서로 다른 군집의 P/N max를 하나의 사건으로 합치지 않는다. 실제 Validation 진단은 기존 Stage A1,778개 기준 Window·Train donor·seed를 재사용한다. Test는 전체 배치 출력만 만들고 분석/선택에는 쓰지 않는다.
