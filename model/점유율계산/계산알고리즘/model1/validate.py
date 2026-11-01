"""Use the existing official-contract validator for model1 and its comparator."""
import argparse
from pathlib import Path
from common import HERE, naive_helper, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    validate = naive_helper("validate.py").validate_file
    files = sorted((args.out / "predictions").glob("*.csv"))
    if not files:
        parser.error("No predictions; run model.py first.")
    rows = [validate(path) for path in files]
    write_json(args.out / "validation.json", rows)
    for row in rows:
        print(f"{'PASS' if row['valid'] else 'FAIL'} {Path(row['file']).name}: {row.get('rows')} rows")
    return 0 if all(row["valid"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
