"""prediction.csv를 공식 제출 형식 검사기로 확인한다(naive/validate.py의 validate_file 재사용).

    python model/점유율계산/계산알고리즘/model1_dead_ball/validate.py                # 가장 최근 실행
    python model/점유율계산/계산알고리즘/model1_dead_ball/validate.py --run-id <실행ID>
"""
import argparse
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model1"))
from common import ROOT, write_json  # noqa: E402

OUT_ROOT = ROOT / "model" / "모델별결과csv" / "model1_dead_ball"


def naive_validate():
    path = HERE.parent / "naive" / "validate.py"
    spec = importlib.util.spec_from_file_location("_naive_validate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_file


def run_dir(run_id):
    if run_id:
        return OUT_ROOT / run_id
    runs = [p for p in OUT_ROOT.iterdir() if (p / "prediction.csv").is_file()] if OUT_ROOT.is_dir() else []
    if not runs:
        raise SystemExit("no prediction; run possession.py first")
    return max(runs, key=lambda p: (p / "prediction.csv").stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    out = run_dir(args.run_id)
    row = naive_validate()(out / "prediction.csv")
    write_json(out / "validation.json", row)
    print(f"{'PASS' if row['valid'] else 'FAIL'} {out.name}/prediction.csv: {row.get('rows')} rows "
          f"{row.get('label_counts', '')} {row.get('reason') or ''}")
    return 0 if row["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
