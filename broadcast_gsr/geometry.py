import csv
from pathlib import Path
import numpy as np


def project(matrix, points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    p = np.c_[points, np.ones(len(points))] @ np.asarray(matrix, dtype=np.float64).T
    good = np.isfinite(p).all(axis=1) & (np.abs(p[:, 2]) > 1e-10)
    result = np.full((len(points), 2), np.nan)
    result[good] = p[good, :2] / p[good, 2:3]
    return result


class Calibration:
    """Matrices always map normalized pitch (0..1) to original video pixels."""
    def __init__(self, path, error_max=None):
        self.rows = {}
        self.error_max = error_max
        if path is None:
            return
        with Path(path).open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            cols = ['h%d%d' % (i,j) for i in range(3) for j in range(3)]
            if not set(['frame','valid']+cols).issubset(reader.fieldnames or []):
                raise ValueError('Calibration CSV requires frame, valid, and h00..h22')
            for row in reader:
                frame = int(row['frame'])
                if frame < 0 or frame in self.rows:
                    raise ValueError('Calibration keys must be unique, nonnegative frame indices')
                matrix = np.array([float(row[k]) if row[k] else np.nan for k in cols]).reshape(3,3)
                error = float(row.get('reprojection_error') or 'nan')
                declared = row['valid'].strip().lower() in ('1','true','1.0')
                self.rows[frame] = (matrix, error, declared)

    def get(self, frame):
        if frame not in self.rows:
            return None, None, 'missing_homography'
        matrix, error, declared = self.rows[frame]
        error = float(error) if np.isfinite(error) else None
        if not declared or not np.isfinite(matrix).all():
            return None, error, 'invalid_homography'
        # Identity is also the upstream failure sentinel; it is not accepted as a pitch fit.
        if np.allclose(matrix, np.eye(3)):
            return None, error, 'identity_homography'
        if np.linalg.cond(matrix) > 1e12:
            return None, error, 'singular_homography'
        if self.error_max is not None and (error is None or error > self.error_max):
            return None, error, 'calibration_error_threshold'
        return matrix, error, ''

    def feet(self, frame, tlwh):
        matrix, error, flag = self.get(frame)
        if matrix is None:
            return np.full((3,2), np.nan), error, flag
        x,y,w,h = map(float,tlwh)
        pixels = [[x,y+h],[x+w,y+h],[x+w/2,y+h]]
        points = project(np.linalg.inv(matrix), pixels)
        if not np.isfinite(points).all():
            return points, error, 'projection_at_infinity'
        return points, error, ''


def template_to_pitch_h(matrix):
    """Upstream image->125x78 template; field corners (10,5)..(115,73)."""
    to_norm = np.array([[1/105.,0,-10/105.],[0,1/68.,-5/68.],[0,0,1.]])
    return np.linalg.inv(to_norm @ matrix)
