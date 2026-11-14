import argparse
import json
from pathlib import Path
from dataclasses import asdict
from ..geometry import load_camera, project_ground


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=Path)
    p.add_argument("--u", type=float, required=True)
    p.add_argument("--v", type=float, required=True)
    a = p.parse_args()
    result = asdict(project_ground(load_camera(a.camera) if a.camera else None, [a.u, a.v]))
    import math
    print(json.dumps({k: None if isinstance(v, float) and not math.isfinite(v) else v for k,v in result.items()}, allow_nan=False))


if __name__ == "__main__":
    main()
