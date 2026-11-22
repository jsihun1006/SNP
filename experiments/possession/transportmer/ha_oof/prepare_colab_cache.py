"""Bundle the seven public DFL Parquet caches for the Colab notebook."""

import argparse
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from dfl_load import MATCHES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/external/dfl")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/possession/transportmer/ha_runs/DFL_7matches_Parquet.zip")
    args = parser.parse_args()

    paths = [args.cache / f"{match}_{kind}.parquet"
             for match in sorted(MATCHES) for kind in ("tracking", "events", "meta")]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error("Run python src/dfl_load.py all first. Missing files:\n" + "\n".join(missing))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".zip.tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
        for path in paths:
            archive.write(path, arcname=path.name)
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
