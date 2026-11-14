"""Read-only audit probes; no model inference, no project writes.

Run with Anaconda base for scipy probes, mtah2027 for OpenCV geometry probes.
AST extraction executes the upstream definitions without importing its model stack.
"""
import argparse
import ast
import collections
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True, help='Unmodified upstream checkout')
parser.add_argument('--project', type=Path, required=True, help='Read-only MTAH project root')
cli = parser.parse_args()
SRC = cli.source.resolve()
PROJECT = cli.project.resolve()


def definitions(path, names, env):
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env


result = {'python': platform.python_version(), 'executable': sys.executable,
          'upstream_sha': subprocess.check_output(['git', '-C', str(SRC), 'rev-parse', 'HEAD'], text=True).strip(),
          'scope': 'isolated functions and synthetic geometry; NOT neural inference or tracking benchmark',
          'probes': {}}
p = result['probes']
files = list(SRC.rglob('*.py'))
for f in files:
    ast.parse(f.read_text())
p['syntax'] = {'files': len(files), 'errors': 0, 'imports_tested': False}

# Actual assignment node: it rescales scores as well as coordinates.
tree = ast.parse((SRC/'inference_soccernetGSR.py').read_text())
node = next(n for n in ast.walk(tree) if isinstance(n, ast.AugAssign)
            and isinstance(n.target, ast.Name) and n.target.id == 'det')
env = {'det': np.array([[100., 100., 200., 300., .9, .9, 0.]]), 'scale': min(1440/1920, 800/1080)}
original = float(env['det'][0, 4]*env['det'][0, 5])
exec(compile(ast.Module(body=[node], type_ignores=[]), '<upstream det /= scale>', 'exec'), env)
p['confidence_rescaling'] = {'scale': env['scale'], 'original_score': original,
    'score_after_original_assignment': float(env['det'][0, 4]*env['det'][0, 5])}
assert p['confidence_rescaling']['score_after_original_assignment'] > 1

env = definitions(SRC/'IDATR/Tracklet.py', {'Tracklet'}, {})
Tracklet = env['Tracklet']
t = Tracklet(1, [1, 2], [.9, .8], [[1, 2, 3, 4], [2, 2, 3, 4]],
             ['Player']*2, ['8']*2, ['blue']*2, ['0']*2, [np.array([1., 0.])]*2)
s = t.extract(0, 0)
p['tracklet_extract'] = {'source_features': len(t.features), 'extracted_features': len(s.features),
                        'extracted_role_item_type': type(s.role[0]).__name__, 'extracted_jerseys': s.jn}
assert not s.features and not s.jn and not isinstance(s.role[0], str)

env = definitions(SRC/'IDATR/refine_tracklets.py', {'majority_vote', 'detect_id_switch'},
                  {'np': np, 'Counter': collections.Counter})
split = env['detect_id_switch']([1, 2, 3, 4], ['Player']*4, ['8','8','9','9'], ['0']*4, 25)
gap_split = env['detect_id_switch']([1, 2, 30, 31], ['Player']*4, ['8','8','9','9'], ['0']*4, 25)
p['identity_split'] = {'continuous_identity_change': split, 'gap_identity_change': gap_split}
assert split == [(0, 4)] and gap_split == [(0, 2), (2, 4)]

if importlib.util.find_spec('scipy'):
    import scipy.linalg
    from scipy.optimize import least_squares
    env = definitions(SRC/'kpts.py',
        {'get_inv_residual_line_circle_points_middle_pair', 'estimate_inv_homography_line_circle_points_middle_pair'},
        {'np': np, 'least_squares': least_squares})
    try:
        env['estimate_inv_homography_line_circle_points_middle_pair']([], [], np.eye(3), [], [])
    except ValueError as exc:
        p['homography_empty_keypoints'] = {'exception': type(exc).__name__, 'message': str(exc)}
    pts = np.array([[0.,0.,1.], [1.,0.,1.], [1.,1.,1.], [0.,1.,1.]])
    fun = env['get_inv_residual_line_circle_points_middle_pair']
    x0 = np.array([1.,0.,0.,0.,1.,0.,0.,0.])
    args = ([], [], np.eye(3), [], pts, pts, np.ones(4))
    assert len(fun(x0, *args)) == 4
    try:
        least_squares(fun, x0, args=args, method='lm')
    except ValueError as exc:
        p['homography_four_points_lm'] = {'residuals': 4, 'variables': 8, 'exception': str(exc)}
    line_pairs = [(np.array([1., 0., -2.]), [np.array([1., 1., 1.])])] * 5
    r1 = fun(x0, line_pairs, [1.]*5, np.eye(3), [[1., 2., 1.]], pts, pts, np.ones(4))
    r2 = fun(x0, line_pairs, [.01]*5, np.eye(3), [[1., 2., .01]], pts, pts, np.ones(4))
    p['homography_unused_confidences'] = {'line_confidence_and_circle_intensity_changes_leave_residuals_identical': bool(np.array_equal(r1,r2))}
    assert np.array_equal(r1,r2)
    kf_env = {}
    exec(compile((SRC/'yolox/EIoU_tracker/kalman_filter.py').read_text(), '<upstream Kalman>', 'exec'), kf_env)
    base_env = {}
    exec(compile((SRC/'yolox/EIoU_tracker/basetrack.py').read_text(), '<upstream BaseTrack>', 'exec'), base_env)
    env = definitions(SRC/'yolox/EIoU_tracker/Deep_EIoU.py', {'STrack'},
        {'np': np, 'deque': collections.deque, 'KalmanFilter': kf_env['KalmanFilter'],
         'BaseTrack': base_env['BaseTrack'], 'TrackState': base_env['TrackState']})
    STrack = env['STrack']
    a = STrack([10.,10.,20.,40.], .9)
    a.activate(kf_env['KalmanFilter'](), 1)
    before = a.mean.copy()
    a.update(STrack([110.,10.,20.,40.], .9), 2)
    p['strack_update_kalman'] = {'mean_unchanged': bool(np.array_equal(a.mean, before)),
                                'last_bbox_x': float(a.last_tlwh[0]), 'kalman_bbox_x': float(a.tlwh[0])}
    assert p['strack_update_kalman']['mean_unchanged']

if importlib.util.find_spec('cv2'):
    import cv2
    import pandas as pd
    from PIL import Image
    import yaml
    cfg = yaml.safe_load((SRC/'configs/config.yaml').read_text())
    result['checkpoint_presence'] = {k: (SRC/v['MODEL_PATH']).exists()
                                    for k,v in cfg.items() if isinstance(v,dict) and 'MODEL_PATH' in v}
    result['template_dimensions'] = {str(f): list(Image.open(SRC/f).size)
                                    for f in ['template/Radar.png', 'template/Radar_Dimen.png']}
    image_pts = np.array([[120.,80.],[1000.,100.],[950.,700.],[150.,650.]])
    template_pts = np.array([[10.,5.],[115.,5.],[115.,73.],[10.,73.]])
    H, _ = cv2.findHomography(image_pts, template_pts, 0)
    template_back = cv2.perspectiveTransform(image_pts[None], H)[0]
    image_back = cv2.perspectiveTransform(template_pts[None], np.linalg.inv(H))[0]
    env = definitions(SRC/'IDATR/create_court_file.py', {'pixel_to_meter_all'}, {})
    centered = np.array(env['pixel_to_meter_all'](*template_back[:3].reshape(-1),62.5,39.)).reshape(3,2)
    expected = np.array([[-52.5,-34.],[52.5,-34.],[52.5,34.]])
    norm = (template_back - [10.,5.])/[105.,68.]
    mtah_m = norm*[105.3,68.]
    p['synthetic_four_point_projection'] = {'image_to_template_max_error': float(abs(template_back-template_pts).max()),
        'inverse_max_pixel_error': float(abs(image_back-image_pts).max()),
        'upstream_center_subtraction_max_error': float(abs(centered-expected).max()),
        'mtah_normalized_corners': norm.tolist(), 'mtah_meter_corners': mtah_m.tolist(),
        'neural_homography_validated': False}
    assert np.allclose(template_back,template_pts,atol=1e-6)
    assert np.allclose(centered,expected,atol=1e-6)
    source_meta = {}
    for source in ['broadcast','tactical']:
        folder = PROJECT/'data'/source
        mp4 = folder/f'BAR-ATH_{source}-001.mp4'
        cap = cv2.VideoCapture(str(mp4))
        m = {'video_path': str(mp4.relative_to(PROJECT)), 'video_opened': cap.isOpened(),
             'video_metadata_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 'fps': cap.get(cv2.CAP_PROP_FPS),
             'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
             'full_video_decode_performed': False}
        cap.release()
        raw = folder/f'BAR-ATH_{source}_atd.txt'
        count = 0; contiguous = True
        with raw.open() as handle:
            for count,line in enumerate(handle,1):
                contiguous &= int(line.split(':',1)[0]) == count
        m.update(atd_records=count, atd_one_based_contiguous=contiguous)
        hom = pd.read_csv(folder/f'BAR-ATH_{source}_homography.csv')
        m.update(homography_rows=len(hom), homography_zero_based_contiguous=bool(np.array_equal(hom['frame'],np.arange(len(hom)))),
                 homography_valid_rows=int(hom['valid'].astype(bool).sum()))
        source_meta[source] = m
    result['local_input_metadata'] = source_meta
result['source_hashes'] = {str(f.relative_to(SRC)):hashlib.sha256(f.read_bytes()).hexdigest()
    for f in [SRC/'inference_soccernetGSR.py',SRC/'kpts.py',SRC/'IDATR/refine_tracklets.py', SRC/'configs/config.yaml']}
print(json.dumps(result,indent=2))
