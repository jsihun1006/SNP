"""골키퍼 쥠 규칙(keeper_hold.py) 진단: 박스 안에서 공이 사라진 경우마다 어느 조건에서 탈락했는지 보여 준다.
직접 연 터미널에서 실행할 것(대회 트래킹을 읽는다). 먼저 python src/final_dead_balls.py를 돌려 둘 것(그 판정을 쓴다).

    python src/diagnose_keeper_holds.py                        # 채점 구간 전체
    python src/diagnose_keeper_holds.py --start 1000 --end 2000   # 프레임 범위만

열(좌표는 출력하지 않는다):
  v, clock, seconds   공이 사라진 프레임, 시각, 다시 보이기까지(또는 상한 22.8초까지) 걸린 시간
  in_play             사라지기 직전 1초가 모두 인플레이로 판정돼 있었나
  keeper_near         그 박스를 지키는 팀에서 골라인에 가장 가까운 선수가 2 m 안에 있었나
  no_line_out         그 전 5초 안에 라인 아웃이 없었나
  zone                다시 나타난 자리: same_box(통과) | goal_area | penalty_mark | outside_box:touchline·corner·center·other | none
  accepted            모두 통과(인플레이로 바뀜)
  first_fail          처음 걸린 조건(in_play → keeper_near → no_line_out → zone 순)
끝에 keeper_near가 True인 경우(골키퍼가 잡았을 가능성이 큰 경우)만 모아 탈락 이유를 센다.
두 번째 표: 공이 보이고 박스 안 골키퍼 2 m 안에 1초 이상 붙어 있던 구간(몰고 나가기 후보)과 탈락 이유.
  rolling_s(굴러가는 속도 2 m/s 초과로 이어진 가장 긴 시간, 1초 이상이어야 함), in_dead(모델이 데드로 봄),
  no_line_out(그 전 22.8초 라인 아웃 없음), dead_started_recently(데드가 22.8초 안에 시작).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]
from config import CONFIG  # noqa: E402
import final_dead_balls as fd  # noqa: E402
import keeper_hold as kh  # noqa: E402

FPS = CONFIG["fps"]


def first_fail(c):
    if c["accepted"]:
        return None
    for key in ("in_play", "keeper_near", "no_line_out"):
        if not c[key]:
            return key
    return "zone:" + c["zone"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    args = ap.parse_args()
    import competition_data
    from apply_competition import clock
    match, _ = competition_data.load_competition_match()
    n = len(match.period)
    sig = fd.ball_signals(match)
    crossings = fd.line_crossings(sig)
    cross_in = crossings[(crossings > 0) & (crossings < n)]
    cross_in = cross_in[match.period[cross_in] == match.period[cross_in - 1]]
    frames = pd.read_csv(os.path.join(fd.OUT_DIR, "frames.csv"))
    state = np.zeros(n, dtype=bool)
    state[frames["frame"].to_numpy()] = frames["dead"].to_numpy().astype(bool)
    end = n if args.end is None else args.end
    rows = []
    for c in kh.candidates(match, sig, cross_in, state):
        if not (args.start <= c["v"] < end):
            continue
        rows.append({"v": c["v"], "clock": clock(c["v"]), "seconds": round((c["r"] - c["v"]) / FPS, 1),
                     "in_play": c["in_play"], "keeper_near": c["keeper_near"], "no_line_out": c["no_line_out"],
                     "zone": c["zone"], "accepted": c["accepted"], "first_fail": first_fail(c) or "-"})
    t = pd.DataFrame(rows, columns=["v", "clock", "seconds", "in_play", "keeper_near", "no_line_out", "zone",
                                    "accepted", "first_fail"])
    pd.set_option("display.width", 200)
    print(t.to_string(index=False) if len(t) else "no ball vanished inside a penalty box in this range")
    near = t[t["keeper_near"] & ~t["accepted"]]
    print("\n== ball vanished next to the keeper but rejected: {} ==".format(len(near)))
    if len(near):
        print(near["first_fail"].value_counts().to_string())

    # 몰고 나가기(공이 보이는 채로 골키퍼가 굴려 나감): 공이 박스 안 골키퍼 2 m 안에 1초 이상 붙어 있던 구간
    after_vanish, _ = kh.keep_holds_alive(state, match, sig, cross_in, dict(kh.SETTING, carry=False))
    carry = [c for c in kh.carry_candidates(match, sig, cross_in, after_vanish) if args.start <= c["a"] < end]
    ct = pd.DataFrame([{"a": c["a"], "clock": clock(c["a"]), "seconds": round(c["seconds"], 1),
                        "rolling_s": round(c["rolling_s"], 1), "in_dead": c["in_dead"], "no_line_out": c["no_line_out"],
                        "dead_started_recently": c["dead_started_recently"], "accepted": c["accepted"],
                        "first_fail": c["first_fail"]} for c in carry])
    print("\n== ball seen next to the keeper in his box for 1 s or more: {} ==".format(len(ct)))
    if len(ct):
        print(ct.to_string(index=False))
        print("\nreasons (only where the model had it dead):")
        print(ct[ct["in_dead"]]["first_fail"].value_counts().to_string())


if __name__ == '__main__':
    main()
