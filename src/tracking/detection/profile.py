"""Model settings for the shared video detection pipeline."""

from dataclasses import dataclass
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "games.py").is_file())


@dataclass(frozen=True)
class Detector:
    key: str
    name: str
    weights: str
    classes: dict[int, str]
    ball_class: int
    confidence: float
    football: bool = False

    @property
    def artifacts(self) -> Path:
        return ROOT / "artifacts" / "tracking" / self.key


PROFILES = {
    "video_naive": Detector("video_naive", "YOLO11n COCO pretrained", "yolo11n.pt",
                            {0: "person", 32: "sports ball"}, 32, .15),
    "football_yolov8x": Detector("football_yolov8x", "football_yolov8x", "football_yolov8x.pt",
                                 {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}, 0, .1, True),
}
DEFAULT_MODEL = "video_naive"


def model_argument(parser):
    """Select a detector before assigning defaults for its paths and flags."""
    parser.add_argument("--model", choices=PROFILES, default=DEFAULT_MODEL)
    known, _ = parser.parse_known_args()
    return PROFILES[known.model]
