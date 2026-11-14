"""Convert provided SynLoc COCO annotations without changing source images/ATD."""
import argparse
import copy
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from ..tiling import regular_grid
from ..image_io import read_image, write_image


def read_annotations(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not doc.get("images") or not doc.get("annotations"):
        raise ValueError("Real, nonempty COCO images and annotations required")
    ids = [im["id"] for im in doc["images"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate image IDs")
    annotation_ids = [a["id"] for a in doc["annotations"]]
    if len(set(annotation_ids)) != len(annotation_ids):
        raise ValueError("Duplicate annotation IDs")
    for a in doc["annotations"]:
        if a["image_id"] not in set(ids) or not np.isfinite(a["bbox"]).all() or min(a["bbox"][2:]) <= 0:
            raise ValueError("Invalid COCO annotation")
    return doc


def convert_pose(source, output):
    doc = copy.deepcopy(read_annotations(source))
    for a in doc["annotations"]:
        kp = np.asarray(a["keypoints"], dtype=float)
        if kp.size != 6 or not np.isfinite(kp).all():
            raise ValueError("Need exactly two labelled points; cannot derive ground from foot joints")
        kp = kp.reshape(2, 3)
        if not np.isin(kp[:, 2], [0, 1, 2]).all():
            raise ValueError("Visibility flags must be 0,1,2")
        a["keypoints"] = kp.reshape(-1).tolist()
        a["num_keypoints"] = int((kp[:, 2] > 0).sum())
        a.setdefault("iscrowd", 0)
    for c in doc["categories"]:
        c.update(keypoints=["pelvis", "ground"], skeleton=[[1, 2]])
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(doc, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


def prepare_detection(source, images, output, split, mode, min_visible=.25):
    doc = read_annotations(source)
    images, output = Path(images), Path(output)
    image_out, label_out = output/"images"/split, output/"labels"/split
    if image_out.exists() or label_out.exists():
        raise FileExistsError("Output split already exists; choose an empty dataset directory")
    image_out.mkdir(parents=True)
    label_out.mkdir(parents=True)
    count = 0
    for im in doc["images"]:
        image = read_image(images/im["file_name"])
        if image is None or image.shape[:2] != (im["height"], im["width"]):
            raise ValueError(f"Unreadable/wrong-size image: {im['file_name']}")
        h, w = image.shape[:2]
        tiles = regular_grid(w, h) if mode == "sahi" else regular_grid(w, h, max(w,h), 0)
        anns = [a for a in doc["annotations"] if a["image_id"] == im["id"] and not a.get("iscrowd", 0)]
        if any(a["category_id"] != 1 for a in anns):
            raise ValueError("Expected SynLoc person category_id=1")
        for index, t in enumerate(tiles):
            labels = []
            for ann in anns:
                x, y, bw, bh = ann["bbox"]
                x0, y0, x1, y1 = max(x,t.x1), max(y,t.y1), min(x+bw,t.x2), min(y+bh,t.y2)
                if x1 <= x0 or y1 <= y0 or (x1-x0)*(y1-y0)/(bw*bh) < min_visible:
                    continue
                tw, th = t.x2-t.x1, t.y2-t.y1
                labels.append(f"0 {((x0+x1)/2-t.x1)/tw:.9f} {((y0+y1)/2-t.y1)/th:.9f} {(x1-x0)/tw:.9f} {(y1-y0)/th:.9f}")
            stem = f"{im['id']}_{index}"
            if not write_image(image_out/f"{stem}.jpg", image[t.y1:t.y2,t.x1:t.x2]):
                raise OSError("Failed to write crop")
            (label_out/f"{stem}.txt").write_text("\n".join(labels), encoding="utf-8")
            count += 1
    # JSON is valid YAML; paths are absolute to remove Windows cwd ambiguity.
    dataset = dict(path=str(output.resolve()), train="images/train", val="images/val", names={0:"person"})
    (output/"dataset.yaml").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    (output/f"{split}_preparation.json").write_text(json.dumps(dict(source=str(Path(source).resolve()),
        source_images=str(images.resolve()), mode=mode, crops=count, min_visible=min_visible,
        implementation_choice="clip intersecting boxes; keep >= min_visible area; retain empty crops"), indent=2), encoding="utf-8")
    print(f"Prepared {count} {mode} images in {image_out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--images", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", choices=["pose", "detection"], required=True)
    p.add_argument("--mode", choices=["full", "sahi"], default="sahi")
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    a = p.parse_args()
    if a.task == "pose":
        convert_pose(a.annotations, a.output)
    elif a.images is None:
        p.error("--images required for detection preparation")
    else:
        prepare_detection(a.annotations, a.images, a.output, a.split, a.mode)


if __name__ == "__main__":
    main()
