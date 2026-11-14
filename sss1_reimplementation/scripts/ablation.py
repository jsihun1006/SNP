import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Same poses/data/thresholds, explicit detector training domains")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--coco", type=Path, required=True)
    p.add_argument("--full-checkpoint", type=Path, required=True)
    p.add_argument("--sahi-checkpoint", type=Path, required=True)
    p.add_argument("--pose-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    paths = [a.coco, a.full_checkpoint, a.sahi_checkpoint, a.pose_checkpoint]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    experiments = json.loads((Path(__file__).resolve().parents[1]/"configs/experiments.json").read_text(encoding="utf-8"))
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/"manifest.json").write_text(json.dumps(dict(experiments=experiments,
        artifacts=[dict(path=str(x.resolve()), sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in paths],
        pose_fixed=True, confidence=.25), indent=2), encoding="utf-8")
    for name, exp in experiments.items():
        out = a.output/name
        ckpt = a.sahi_checkpoint if exp["training_domain"] == "sahi" else a.full_checkpoint
        subprocess.run([sys.executable, "-m", "sss1_reimplementation.scripts.infer", "--input", str(a.images),
                        "--coco", str(a.coco), "--detector", str(ckpt), "--pose-checkpoint", str(a.pose_checkpoint),
                        "--mode", exp["inference"], "--device", a.device, "--output", str(out)], check=True)
        subprocess.run([sys.executable, "-m", "sss1_reimplementation.scripts.evaluate", "--gt", str(a.coco),
                        "--predictions", str(out/"results.json"), "--output", str(out/"metrics.json")], check=True)


if __name__ == "__main__":
    main()
