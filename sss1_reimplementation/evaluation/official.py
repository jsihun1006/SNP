"""Adapters to COCO bbox AP and unmodified sskit LocSimCOCOeval."""
import copy
import json
from pathlib import Path
import numpy as np
from .metrics import localization_metrics


def _load_result(coco, results):
    if results:
        return coco.loadRes(copy.deepcopy(results))
    empty = type(coco)()
    empty.dataset = dict(images=copy.deepcopy(coco.dataset["images"]),
                         categories=copy.deepcopy(coco.dataset["categories"]), annotations=[])
    empty.createIndex()
    return empty


def evaluate(gt_path, results_path, score_threshold=.25, include_locsim=True):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt_doc = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    if not gt_doc.get("annotations"):
        raise ValueError("Ground truth missing/empty; cannot evaluate")
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    ids = {im["id"] for im in gt_doc["images"]}
    cats = {c["id"] for c in gt_doc["categories"]}
    for p in results:
        if p["image_id"] not in ids or p["category_id"] not in cats:
            raise ValueError("Prediction image/category is absent from ground truth")
        if (len(p["bbox"]) != 4 or not np.isfinite([*p["bbox"], p["score"]]).all()
                or min(p["bbox"][2:]) <= 0 or not 0 <= p["score"] <= 1):
            raise ValueError("Invalid prediction")
    coco = COCO()
    coco.dataset = copy.deepcopy(gt_doc)
    coco.dataset.setdefault("info", {})
    for index, a in enumerate(coco.dataset["annotations"]):
        # SynLoc make_coco starts at id=0, which standard pycocotools reserves
        # as the unmatched sentinel. Reindex this in-memory copy only.
        a["id"] = index + 1
        a.setdefault("iscrowd", 0)
    coco.createIndex()
    bbox = COCOeval(coco, _load_result(coco, results), "bbox")
    bbox.evaluate(); bbox.accumulate(); bbox.summarize()
    names = ["AP_50_95", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large",
             "AR_max1", "AR_max10", "AR_max100", "AR_small", "AR_medium", "AR_large"]
    output = {"detection_coco": dict(zip(names, map(float, bbox.stats))),
              "score_threshold": score_threshold}
    if not include_locsim:
        return output
    if cats != {1}:
        raise ValueError("SynLoc official LocSim adapter requires the single person category_id=1")
    if any("position_on_pitch" not in a for a in gt_doc["annotations"]):
        raise ValueError("Official LocSim requires world ground truth; use --detection-only for bbox AP")
    # Keep bbox detections with missing calibration in bbox evaluation, but they cannot enter world AP.
    valid = [p for p in results if p.get("world_valid", True) and "position_on_pitch" in p]
    if any(not np.isfinite(p["position_on_pitch"]).all() for p in valid):
        raise ValueError("Invalid world coordinate prediction")
    from xtcocotools.coco import COCO as XTCOCO
    from sskit.coco import LocSimCOCOeval
    xt = XTCOCO()
    xt.dataset = copy.deepcopy(coco.dataset)
    xt.createIndex()
    output["sskit"] = {}
    for tau in (1., 5.):
        ev = LocSimCOCOeval(xt, _load_result(xt, valid), "bbox")
        ev.params.useSegm = None
        ev.params.score_threshold = score_threshold
        ev.locsim_tau = tau
        ev.evaluate(); ev.accumulate(); ev.summarize()
        output["sskit"][f"tau{tau:g}m"] = dict(mAP_LocSim=float(ev.stats[0]),
                precision=float(ev.stats[12]), recall=float(ev.stats[13]), f1=float(ev.stats[14]),
                frame_accuracy=float(ev.stats[16]), stats=ev.stats.tolist())
    output["localization_diagnostics"] = localization_metrics(gt_doc, [p for p in results if p["score"] >= score_threshold])
    return output
