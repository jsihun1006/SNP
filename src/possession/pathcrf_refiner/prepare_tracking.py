"""Cache the official meter-coordinate tracking from local DFL XML."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from run_pathcrf_dfl import load_official_tracking


def prepare_tracking(upstream, raw, output, match):
    sys.path.insert(0, str(upstream.resolve()))
    path = output / f"{match}.parquet"
    if not path.exists():
        tracking = load_official_tracking(raw, match)
        cols = ["frame_id", "period_id", "timestamp", "ball_state", "ball_owning_team_id"]
        cols += [c for c in tracking if c.endswith(("_x", "_y"))]
        output.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.parquet")
        tracking[cols].to_parquet(tmp, index=False)
        tmp.replace(path)
    return path


if __name__ == "__main__":
    upstream, raw, output, *matches = sys.argv[1:]
    for match in matches:
        print(prepare_tracking(Path(upstream), Path(raw), Path(output), match), flush=True)
