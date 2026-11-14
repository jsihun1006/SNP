"""Check that regenerated feature selection matches the published schema."""
import json
from pathlib import Path
import sys

folder = Path(__file__).resolve().parent
expected = json.loads((folder / "lightgbm_vif50.json").read_text(encoding="utf-8"))
observed = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
columns = json.loads((folder / "feature_columns_370.json").read_text(encoding="utf-8"))
removed = [entry["column"] for entry in observed["removed"]]
if removed != [entry["column"] for entry in expected["removed"]]:
    raise ValueError("VIF elimination order differs.")
if observed["retained"] != expected["retained"] or max(observed["final_vif"].values()) > 50:
    raise ValueError("VIF retained features differ or exceed the limit.")
if [c for c in columns if c not in {"home_speed_mean", "away_speed_mean"}] != observed["retained"]:
    raise ValueError("Final schema differs after restoring the two mean-speed features.")
print(f"VERIFIED VIF: {len(removed)} removals; {len(columns)} final features")
