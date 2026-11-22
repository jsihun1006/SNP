import cv2
import numpy as np


def draw_tiles(image, tiles, boxes=()):
    out = image.copy()
    for t in tiles:
        cv2.rectangle(out, (t.x1, t.y1), (t.x2-1, t.y2-1), (0, 180, 255), 2)
    for b in boxes:
        cv2.rectangle(out, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 0, 255), 2)
    return out


def draw_predictions(image, rows):
    out = image.copy()
    for r in rows:
        box = [int(r[k]) for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
        cv2.rectangle(out, tuple(box[:2]), tuple(box[2:]), (60, 220, 60), 2)
        text = f'{r["detection_id"]}: {r["confidence"]:.2f}'
        if r["world_valid"]:
            text += f' ({r["world_x"]:.1f},{r["world_y"]:.1f})m'
        cv2.putText(out, text, (box[0], max(15, box[1]-4)), cv2.FONT_HERSHEY_SIMPLEX, .5, (60, 220, 60), 1)
        points = []
        for prefix, color in (("pelvis", (0, 140, 255)), ("ground", (255, 200, 0))):
            p = [r[f"{prefix}_u"], r[f"{prefix}_v"]]
            if all(v is not None and np.isfinite(v) for v in p):
                point = tuple(map(lambda v: int(round(v)), p))
                cv2.circle(out, point, 4, color, -1)
                points.append(point)
        if len(points) == 2:
            cv2.line(out, *points, (255, 200, 0), 1)
    return out
