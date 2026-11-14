import argparse
from pathlib import Path
from ..detection.yolo import YOLODetector
from .probe_environment import require_cuda


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", default="yolo26l.pt")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    a = p.parse_args()
    require_cuda(a.device)
    if not a.data.is_file():
        raise FileNotFoundError(a.data)
    detector = YOLODetector(a.checkpoint, a.device)
    detector.model.train(data=str(a.data.resolve()), epochs=a.epochs, imgsz=1280,
                         batch=a.batch_size, device=a.device, workers=0,
                         project=str(a.output.resolve()), name="yolo26l", exist_ok=False)


if __name__ == "__main__":
    main()
