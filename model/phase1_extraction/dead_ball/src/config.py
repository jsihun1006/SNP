"""데드볼 파이프라인의 설정값. 경로·하이퍼파라미터·상수는 여기서만 바꾼다.

값마다 근거를 주석으로 단다: 축구 규칙, 공개 문서, DFL(공개 경기)에서 잰 값, 또는 미리 정한 값.
대회 경기 영상이나 공개 정답 5분에 맞춰 고른 값은 없다.
"""

import os

ROOT = __import__("repo_root").ROOT

CONFIG = {
    # --- 경로 -------------------------------------------------------------------------------------
    # DFL(IDSSE) parquet 캐시와 학습 결과. data/는 커밋되지 않는다.
    "dfl_cache": os.path.join(ROOT, "data", "external", "dfl"),
    # 브라우저로 받은 DFL 원본 XML을 두는 곳(figshare가 스크립트 내려받기를 막는다: HTTP 202/403).
    "dfl_raw": os.path.join(ROOT, "data", "external", "dfl", "raw"),
    "fps": 25,

    # --- 특징 (src/features.py) -------------------------------------------------------------------
    # 속도는 위치의 앞뒤 차분(±2프레임 = 0.16초)으로 구한다. 위치를 그대로 미분하면 흔들림이 커진다.
    "speed_half_window": 2,
    # 사람이 낼 수 없는 속도(트랙 교체로 튄 값)는 버린다.
    "max_player_speed": 12.0,
    "max_ball_speed": 40.0,
    # 1초 미만의 짧은 공 결측만 선형 보간한다(공 특징은 진단용이며 모델 입력에는 쓰지 않는다).
    "ball_interp_max_s": 1.0,
    # 시간 문맥: 특징의 과거(-)·미래(+) 시점 값(초). 반칙 직후 선수들이 멈추는 모습은 몇 초 뒤에 보인다.
    "context_offsets_s": [-10, -4, -2, -1, -0.4, 0.4, 1, 2, 4, 10],
    # 팀 길이·너비는 이 백분위 범위로 잰다. 최댓값−최솟값은 팀이 잘못 붙은 트랙 하나에 크게 흔들린다.
    "formation_quantiles": (10, 90),
    "speed_quantiles": [10, 25, 50, 75, 90],
    # 모델 입력은 선수 특징만 쓴다. DFL은 데드볼 동안 공을 고정하고 Metrica 샘플은 공을 지워서, 공 특징은
    # 학습 데이터에서 정답을 그대로 드러낸다. 공은 최종 단계의 라인 아웃 판정에만 쓴다.
    "player_features": ["home_speed_mean", "home_speed_max", "home_speed_std", "home_count", "home_length",
                        "home_width", "away_speed_mean", "away_speed_max", "away_speed_std", "away_count",
                        "away_length", "away_width", "all_speed_mean", "centroid_distance"],
    # 특징 세트. v1 = player_features만, v2 = v1 + 기준값 없는 분포 요약.
    # v2가 DFL(경기 하나씩 빼고)과 Metrica 2경기 모두에서 더 좋았다(인플레이 구간 F1 DFL 0.757 → 0.845).
    # v1은 비교 재현용으로만 남긴다(train_dfl.py, metrica_eval.py의 --features).
    "default_feature_set": "v2",
    "feature_sets": {
        "v1": [],
        "v2": ["all_speed_p10", "all_speed_p25", "all_speed_p50", "all_speed_p75", "all_speed_p90",
               "team_speed_gap", "touchline_min", "goalline_min", "nn_mean", "nn_min"],
    },

    # --- 트래킹 노이즈 시뮬레이션 (src/simulate.py): DFL을 대회 트래킹처럼 -----------------------
    # 목표값은 대회 데이터의 자동 집계(compare_domains.py)에서 왔다. 사람이 단 라벨은 쓰지 않았다.
    "sim_target_players": 20.5,       # 프레임당 평균 추적 인원(DATA.md, 채점 구간)
    "sim_dropout_mean_s": 10.0,       # 선수 한 명이 안 보이는 구간의 평균 길이(초)
    # 여러 명이 함께 빠지는 누락(대회: 프레임당 인원 p50 22, p10 18). 전체 누락 중 비율, 인원 범위, 평균 길이.
    "sim_group_share": 0.8,
    "sim_group_size": (2, 6),
    "sim_group_mean_s": 10.0,
    # 빈 구간 길이의 로그 표준편차(대회: p50 7.8초, p90 163초로 꼬리가 길다). 0이면 지수 분포.
    "sim_gap_sigma": 1.5,
    # 좌표 흔들림의 표준편차(m)와 상관 시간(초). 부드럽게 이어서 속도 잡음이 부풀지 않게 한다.
    "sim_jitter_m": 0.15,
    "sim_jitter_smooth_s": 1.0,
    "sim_seed": 7,
    # 학습 경기마다 이 범위에서 설정을 무작위로 뽑는다(특정 누락 모양에 기대지 않도록, simulate.simulate의 인자).
    "sim_random_ranges": {
        "target_players": (19.5, 21.5),
        "dropout_mean_s": (4.0, 20.0),
        "group_share": (0.4, 0.95),
        "group_mean_s": (4.0, 20.0),
        "gap_sigma": (0.5, 2.0),
        "jitter_m": (0.05, 0.3),
    },
    # 공 누락 시뮬레이션(simulate.ball_dropout): 라인 아웃 처리를 공이 안 보이는 조건에서 시험할 때 쓴다.
    "sim_ball_missing_share": (0.0, 0.25),
    "sim_ball_gap_mean_s": (0.5, 6.0),

    # --- 학습과 평가 (src/train_dfl.py, src/metrics.py) -------------------------------------------
    "train_stride": 5,  # 이웃 프레임은 거의 같아서 학습에는 5프레임마다 하나만 쓴다
    "lgbm_params": {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 200,
                    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.7, "reg_lambda": 1.0,
                    "random_state": 0, "verbose": -1},
    "cause_max_gap_s": 3.0,  # DFL 데드볼 원인: 구간 끝에서 이 시간 안의 세트피스 이벤트로 정한다
    "min_segment_s": 2.0,    # 구간 정밀도를 셀 때 이보다 짧은 예측 구간은 세지 않는다

    # --- 평활화 (src/smoothing.py, src/compare_smoothing.py) --------------------------------------
    # 비교한 후보.
    "smooth_median_s": [0.5, 1.0, 2.0, 3.0, 5.0],
    "smooth_viterbi_penalty": [5, 10, 20, 40, 80],
    "smooth_viterbi_bias": [-0.5, 0.0, 0.5],
    "smooth_min_s": [0.0, 1.0, 2.0, 3.0],
    # 고른 설정: 2상태 비터비, 전환 벌점 40. DFL에서 다른 6경기로 고르기를 7번 해서 7번 모두 같은 값.
    "smooth_choice": {"method": "viterbi", "penalty": 40.0, "bias": 0.0, "min_s": 0.0},
    # 느린 빌드업 보정(src/slow_buildup.py, JSH 노트북 ⑤): 패스·수신·공 동행 구간은 데드 진입을 어렵게.
    # 값은 JSH 노트북의 COMPETITION_RULES 그대로(DFL 학습 쪽 내부 검증 경기로 선택).
    # + JSH 노트북 ①(데드볼 시작 늦음)·⑥(짧은 데드볼 누락), 2026-11-16 채택(src/eval_jsh_all_corrections.py): ⑤만 대비
    #   DFL +0.0011(2/0), Metrica +0.0042, SkillCorner −0.0007. ②·③·④와 ①~⑥ 전부는 불통과. ①은 팀 정보가 필요하다.
    # + restart_snap(2026-11-17, src/eval_restart_late.py): 데드볼 끝 앞 5초 안의 재개로 끝을 당긴다(새 구간을 만들지 않음).
    #   ⑤+①+⑥ 대비 DFL +0.0015(4/1), Metrica +0.0049, SkillCorner +0.0068(17/1). 노트북 ③ 그대로(비용 조정)는
    #   SkillCorner −0.0076, 양 팀 가속 확인을 붙여도 −0.0028로 불통과, 경계 당기기에 가속 확인을 붙이면 효과 0.
    # 주최 측 답변(2026-11-17: 결과를 대회 영상과 대조 금지, 튜닝은 외부 데이터로만) 뒤 원칙: 외부 데이터에서 미리 정한
    # 기준으로 고른 것만 남긴다. 페널티킥 두 값은 공개 데이터 비교로 골랐다(하한 0: 가짜 0 그대로 + SkillCorner 페널티 골
    # 1개 더, src/eval_penalty_visible.py / 배치가 깨진 순간: 공개 페널티킥 시점 오차 0.08초 vs 마크 도착 0.20초,
    # src/eval_penalty_moment.py). 골키퍼는 keeper_hold.EVIDENCE(DFL 기준, v8·pitch_check 제외).
    "penalty": {"run_up_moment": "break", "min_visible": 0},
    # + lineout_lock soft(2026-11-17, src/eval_lineout_lock.py): 공이 바깥으로 움직이며 라인 0.36 m 바깥 관측이 2프레임
    #   이어지면(⑦) 넘은 프레임부터 데드 쪽 점수 +0.45 × bonus와 진입 비용 0, 이탈 직전 안쪽 프레임은 모델이 인플레이로
    #   보면 인플레이로 고정. DFL +0.0005(2/1), 쥠 공 삭제 +0.0005, Metrica 0, SkillCorner −0.0012(2/8).
    #   강제 데드 고정(hard)은 DFL −0.023: 사건의 99.7%가 2초 안에 실제 데드였지만 DFL은 넘은 순간보다 늦게 데드로 적고,
    #   강제 고정 뒤 짧은 인플레이 조각이 생겨 구간이 쪼개졌다.
    "slow_buildup": {"margin": 0.0, "strength": 1.0, "bonus": 0.85, "case1": True, "case6": True,
                     "restart_snap": True, "lineout_lock": True, "lock_mode": "soft"},

    # --- 대회 데이터 적용 (src/apply_competition.py, src/final_dead_balls.py) ---------------------
    # 스마트 태깅 SET PIECES 태그 창을 앞뒤로 이만큼(초) 넓혀 겹치는지 본다. 결과 보고(참고용 지표)에만 쓴다.
    "tag_pad_s": 5.0,
    "final": {
        "public_kickoffs": (11185, 110808, 113742),  # FAQ.md·DATA.md 공개값: 전반 킥오프, 후반 채점 시작, 후반 킥오프
        "boundary_margin_m": 0.5,       # 좌표 흔들림을 감안한 경기장 경계 여유
        "ball_smooth_frames": 7,        # 공 속도 계산: 위치 이동평균 프레임 수
        "ball_speed_gap_frames": 6,     # 공 속도 계산: 차분 간격 프레임 수
        "ball_moving_mps": 4.0,         # 빠르게 움직이는 공(경기 중 패스 수준)
        "line_out_moving_mps": 2.0,     # 라인을 넘기 직전 굴러가던 공(서 있는 공의 흔들림 약 1 m/s보다 위)
        "out_confirm_s": 2.0,           # 넘은 뒤 이 시간 안에 경기장 안에서 빠른 공이 보이면 높이 뜬 공의 투영으로 본다
        # 공이 나간 뒤 가장 빨리 재개된 시간(DFL 7경기, 스로인·골킥·코너킥 데드볼 451번 중 최솟값 1.6초).
        # 넘은 뒤 이보다 빨리 경기장 안에서 공이 보이면(속도 무관) 나간 적이 없다고 본다: 뜬 공의 지면 투영.
        "out_min_restart_s": 1.6,
        "out_start_window_s": 5.0,      # 모델 데드볼 시작 앞뒤 이 시간 안의 라인 아웃으로 시작을 옮긴다
    },
    # Ball of the provided match: "atd" = the provided Tactical ATD ball; "corrected" = the ball of the team's
    # ball_tracking checkpoint (model/phase1_extraction/ball_tracking, its README states its provenance).
    # With "corrected", Tactical frames that the checkpoint maps take its ball (missing there stays missing);
    # frames it does not map keep the ATD ball. Players always come from the provided ATD.
    "competition_ball": {
        "source": "corrected",
        "csv": os.path.join(ROOT, "outputs", "unified_tracking_v1", "tracking_all_frames.csv"),
        "sha256": "bdd017ba3f8335028a951631768780af570e71bf064c7bb850572af5419b0608",
    },
}
