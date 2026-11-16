"""dead_ball(Phase 1) → model1_dead_ball(Phase 2) → 형식 검사 → 공개 5분 평가를 차례로 실행한다.

    python model/점유율계산/계산알고리즘/model1_dead_ball/run.py                    # dead_ball 다시 만들고 전부
    python model/점유율계산/계산알고리즘/model1_dead_ball/run.py --skip-dead-ball   # 이미 있는 dead_ball 실행 사용
    python model/점유율계산/계산알고리즘/model1_dead_ball/run.py --skip-evaluation  # 평가 없이 예측까지

직접 연 터미널에서 실행할 것(대회 데이터를 읽는다).
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-dead-ball", action="store_true", help="dead_ball extract.py를 다시 돌리지 않는다")
    parser.add_argument("--dead-run", help="쓸 dead_ball 실행ID(기본: 가장 최근)")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[3]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    steps = []
    if not args.skip_dead_ball:
        steps.append(root / "model" / "영상추출" / "모델" / "dead_ball" / "extract.py")
    steps += [here / "possession.py", here / "validate.py"]
    if not args.skip_evaluation:
        steps.append(here / "evaluation.py")
    for script in steps:
        extra = ["--dead-run", args.dead_run] if (script.name == "possession.py" and args.dead_run) else []
        print(f"\n== {script.relative_to(root)} ==", flush=True)
        subprocess.run([sys.executable, str(script), *extra], cwd=root, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
