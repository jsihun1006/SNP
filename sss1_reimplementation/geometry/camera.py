"""Explicit camera conventions; no estimated or fallback homography."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


@dataclass(frozen=True)
class WorldPoint:
    x: float = float("nan")
    y: float = float("nan")
    z: float = float("nan")
    valid: bool = False
    reason: str = "missing_calibration"


def _array(value, shape=None):
    a = np.asarray(value, dtype=np.float64)
    if (shape is not None and a.shape != shape) or not np.isfinite(a).all():
        raise ValueError(f"Invalid calibration array, expected {shape}")
    return a


class RayCamera:
    def __init__(self, width, height, rotation, translation, pitch_bounds=None):
        self.width, self.height = int(width), int(height)
        if min(self.width, self.height) <= 0:
            raise ValueError("Invalid calibrated image dimensions")
        self.R, self.t = _array(rotation, (3, 3)), _array(translation, (3,))
        if abs(np.linalg.det(self.R)) < 1e-12:
            raise ValueError("Singular camera matrix")
        self.inv_R = np.linalg.inv(self.R)
        self.center = -self.inv_R @ self.t
        self.pitch_bounds = pitch_bounds
        if pitch_bounds is not None:
            b = _array(pitch_bounds, (4,))
            if b[0] >= b[2] or b[1] >= b[3]:
                raise ValueError("pitch_bounds is [xmin,ymin,xmax,ymax]")

    def pixel_to_ground(self, uv):
        uv = np.asarray(uv, dtype=float)
        if uv.shape != (2,) or not np.isfinite(uv).all():
            return WorldPoint(reason="invalid_pixel")
        if not (0 <= uv[0] < self.width and 0 <= uv[1] < self.height):
            return WorldPoint(reason="outside_image")
        try:
            q = self.pixel_to_normalized(uv)
            direction = self.inv_R @ np.array([q[0], q[1], 1.])
            if not np.isfinite(direction).all():
                return WorldPoint(reason="invalid_ray")
            if abs(direction[2]) < 1e-10:
                return WorldPoint(reason="parallel_ray")
            distance = -self.center[2] / direction[2]
            if distance <= 0:
                return WorldPoint(reason="behind_camera")
            xyz = self.center + distance * direction
            if not np.isfinite(xyz).all():
                return WorldPoint(reason="invalid_intersection")
            if self.pitch_bounds is not None:
                x0, y0, x1, y1 = self.pitch_bounds
                if not (x0 <= xyz[0] <= x1 and y0 <= xyz[1] <= y1):
                    return WorldPoint(reason="outside_pitch")
            return WorldPoint(float(xyz[0]), float(xyz[1]), 0., True, "ok")
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return WorldPoint(reason="projection_failed")

    def world_to_pixel(self, xyz):
        point = _array(xyz, (3,))
        cam = self.R @ point + self.t
        if cam[2] <= 1e-10:
            raise ValueError("World point is on/behind camera plane")
        return self.normalized_to_pixel(cam[:2] / cam[2])


class OpenCVCamera(RayCamera):
    """X_camera = R @ X_world + t; Z_world=0, metres; Brown-Conrady lens."""
    def __init__(self, width, height, K, R, t, distortion=None, pitch_bounds=None):
        super().__init__(width, height, R, t, pitch_bounds)
        self.K = _array(K, (3, 3))
        if self.K[0, 0] <= 0 or self.K[1, 1] <= 0 or not np.allclose(self.K[2], [0, 0, 1]):
            raise ValueError("Invalid intrinsic matrix")
        if not np.isclose(self.K[0, 1], 0) or not np.isclose(self.K[1, 0], 0):
            raise ValueError("OpenCV adapter requires zero-skew intrinsics")
        if not np.allclose(self.R.T @ self.R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(self.R), 1):
            raise ValueError("R must be a proper world-to-camera rotation")
        self.distortion = _array(distortion if distortion is not None else [0]*5).reshape(-1)
        if len(self.distortion) not in (4, 5, 8, 12, 14):
            raise ValueError("Unsupported OpenCV distortion length")

    def pixel_to_normalized(self, uv):
        import cv2
        q = cv2.undistortPointsIter(uv.reshape(1, 1, 2), self.K, self.distortion, None, None,
                                   (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12))
        return q[0, 0]

    def normalized_to_pixel(self, q):
        import cv2
        xyz = np.array([q[0], q[1], 1.], dtype=np.float64)
        uv, _ = cv2.projectPoints(xyz.reshape(1, 3), np.zeros(3), np.zeros(3), self.K, self.distortion)
        return uv[0, 0]


class SynLocCamera(RayCamera):
    """sskit polynomial camera. Both axes divide by WIDTH, centre=(W-1,H-1)/2.

    Polynomial coefficients are descending order and map radius <-> ray angle.
    These coefficients are not OpenCV k1/k2/p1/p2 coefficients.
    """
    def __init__(self, width, height, camera_matrix, dist_poly, undist_poly, pitch_bounds=None):
        p = _array(camera_matrix)
        if p.shape == (4, 4):
            p = p[:3]
        if p.shape != (3, 4):
            raise ValueError("SynLoc camera_matrix must be 3x4 or 4x4")
        super().__init__(width, height, p[:, :3], p[:, 3], pitch_bounds)
        self.dist_poly, self.undist_poly = _array(dist_poly), _array(undist_poly)
        if self.dist_poly.ndim != 1 or self.undist_poly.ndim != 1 or min(self.dist_poly.size, self.undist_poly.size) < 1:
            raise ValueError("Invalid radial polynomials")
        self.principal = np.array([(self.width-1)/2, (self.height-1)/2])

    @staticmethod
    def _radial(point, radius):
        return point * radius / max(float(np.linalg.norm(point)), 1e-12)

    def pixel_to_normalized(self, uv):
        p = (uv - self.principal) / self.width
        angle = np.polyval(self.undist_poly, np.linalg.norm(p))
        if not np.isfinite(angle) or abs(angle) >= np.pi/2:
            raise ValueError("Lens ray beyond forward hemisphere")
        return self._radial(p, np.tan(angle))

    def normalized_to_pixel(self, q):
        radius = np.polyval(self.dist_poly, np.arctan(np.linalg.norm(q)))
        return self._radial(q, radius) * self.width + self.principal


def load_camera(path: Path):
    path = Path(path)
    if path.is_dir():
        # Delegate raw lens fitting to the official devkit, never reinterpret it as Brown distortion.
        from sskit.camera import load_camera as official_load
        p, d, u = official_load(path)
        from ..image_io import read_image
        image = read_image(path / "rgb.jpg")
        if image is None:
            raise ValueError("Raw SynLoc camera directory requires rgb.jpg for image size")
        h, w = image.shape[:2]
        return SynLocCamera(w, h, p.numpy(), d.numpy(), u.numpy())
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("model") == "opencv":
        return OpenCVCamera(**{k: doc[k] for k in ("width", "height", "K", "R", "t")},
                            distortion=doc.get("distortion"), pitch_bounds=doc.get("pitch_bounds"))
    if doc.get("model", "synloc") == "synloc":
        return SynLocCamera(**{k: doc[k] for k in ("width", "height", "camera_matrix", "dist_poly", "undist_poly")},
                            pitch_bounds=doc.get("pitch_bounds"))
    raise ValueError("Unknown camera model")


def project_ground(camera, uv):
    return camera.pixel_to_ground(uv) if camera is not None else WorldPoint()
