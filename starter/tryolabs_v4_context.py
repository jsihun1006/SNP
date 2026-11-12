"""Weak, automatic image geometry. No pitch calibration or live-play truth."""
import cv2
import numpy as np
from collections import deque


class CameraTransitionMonitor:
    """Conservative hard-cut/dissolve hints, not broadcast-state recognition."""
    def __init__(self, policy):
        self.policy=policy;self.history=deque(maxlen=policy["history_frames"])

    def update(self, frame):
        im=cv2.resize(frame,(64,36));hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        hist=cv2.calcHist([hsv],[0,1],None,[18,8],[0,180,0,256])
        cv2.normalize(hist,hist)
        reason=None
        for offset,prefix in [(-1,"hard"),(0,"window")]:
            if not self.history:continue
            if prefix=="window" and len(self.history)<self.policy["history_frames"]:continue
            old,oldhist=self.history[offset]
            distance=cv2.compareHist(hist,oldhist,cv2.HISTCMP_BHATTACHARYYA)
            difference=float(np.abs(im.astype(float)-old.astype(float)).mean())
            if (distance>=self.policy[prefix+"_histogram_distance"]
                    and difference>=self.policy[prefix+"_mean_pixel_difference"]):
                reason="CAMERA_VIEW_TRANSITION_RECHECK"
        self.history.append((im,hist))
        return reason


def frame_context(frame, objects):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (640, 360))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (25, 40, 35), (85, 255, 255))
    coverage = float(np.mean(green > 0))
    people = 0
    for o in objects:
        try:p = np.asarray(o.get("points"),dtype=float)
        except (TypeError,ValueError):continue
        if p.shape==(2,2) and np.isfinite(p).all() and o.get("age",0)==0 and 0<p[1,1]-p[0,1]<h*.25:
            people += 1
    # A wide field view can also be pre-match or replay: do not call it live.
    return dict(green_coverage=coverage, small_people=people,
                field_view_supported=bool(coverage >= .35 and people >= 6),
                broadcast_state="UNVERIFIED", pitch_geometry_verified=False,
                technical_area_verified=False, camera_motion_compensated=False,
                white_mask=cv2.inRange(hsv, (0, 0, 170), (179, 70, 255)),
                frame_size=(w, h))


def goal_geometry_hint(points, context):
    """Look for a crossbar joining two roughly upright white segments nearby.

    Field lines, advertising and spectators can imitate this pattern. The output
    is explicitly a weak hypothesis and cannot establish a goalkeeper alone.
    """
    if not context.get("field_view_supported") or "white_mask" not in context:
        return False
    white = context["white_mask"].copy()
    w, h = context["frame_size"]
    p = np.asarray(points) * [640/w, 360/h]
    (x1, y1), (x2, y2) = p
    ph = max(6., y2-y1); cx = (x1+x2)/2
    left, right = max(0, int(cx-5*ph)), min(640, int(cx+5*ph))
    top, bottom = max(0, int(y1-2*ph)), min(360, int(y2+ph))
    window = white[top:bottom, left:right]
    if window.size == 0:
        return False
    lines = cv2.HoughLinesP(window, 1, np.pi/180, threshold=8,
                            minLineLength=max(5, int(ph*.4)), maxLineGap=4)
    if lines is None:
        return False
    posts = []; bars = []
    for a, b, c, d in lines[:, 0]:
        if abs(d-b) >= max(5, ph*.35) and abs(c-a) <= .45*abs(d-b):
            posts.append((np.array([a,b]) if b < d else np.array([c,d]), max(b,d)))
        if abs(c-a) >= ph*.6 and abs(d-b) <= .7*abs(c-a):
            bars.append((np.array([a,b]), np.array([c,d])))
    tolerance = max(5, ph*.5)
    for a, b in bars:
        for p1, _ in posts:
            if min(np.linalg.norm(a-p1), np.linalg.norm(b-p1)) > tolerance:
                continue
            for p2, _ in posts:
                if abs(p1[0]-p2[0]) < ph*.6:
                    continue
                if min(np.linalg.norm(a-p2), np.linalg.norm(b-p2)) <= tolerance:
                    return True
    return False
