import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PATH_KEYS = ('video', 'calibration_csv', 'upstream', 'reid_source', 'detector_weights',
             'reid_weights', 'clip_weights', 'sfr_weights', 'output_root', 'replay')


def load_config(path, overrides=None):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding='utf-8-sig'))
    root = (path.parent / cfg.get('base_dir', '.')).resolve()
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    for name in PATH_KEYS:
        if cfg.get(name):
            cfg[name] = str((root / cfg[name]).resolve())
    if cfg.get('scene_type', 'unverified') not in ('live', 'unverified', 'replay', 'closeup'):
        raise ValueError('scene_type must be live, unverified, replay, or closeup')
    if int(cfg.get('max_frames', 750)) < 1:
        raise ValueError('max_frames must be positive')
    if int(cfg.get('start_frame', 0)) < 0:
        raise ValueError('start_frame must be nonnegative')
    if float(cfg.get('pitch_length_m', 0)) <= 0 or float(cfg.get('pitch_width_m', 0)) <= 0:
        raise ValueError('Pitch dimensions must be positive')
    if any(v not in ('H', 'A') for v in cfg.get('team_colors', {}).values()):
        raise ValueError('team_colors values must be H or A')
    cfg['_config_path'] = str(path)
    return cfg


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def new_run(cfg, output=None, prefix='run'):
    root = Path(cfg['output_root'])
    path = Path(output).resolve() if output else root / (prefix+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    inputs = [Path(cfg[k]).resolve() for k in PATH_KEYS if cfg.get(k) and k != 'output_root']
    resolved = path.resolve()
    if any(resolved == p or resolved in p.parents for p in inputs):
        raise ValueError('Output must not contain an input path')
    path.mkdir(parents=True, exist_ok=False)
    json_write(path/'config.json', cfg)
    return path


def signature(path, full=False):
    p = Path(path)
    st = p.stat()
    result = {'path': str(p.resolve()), 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}
    if full:
        digest = hashlib.sha256()
        with p.open('rb') as handle:
            for block in iter(lambda: handle.read(1024*1024), b''):
                digest.update(block)
        result['sha256'] = digest.hexdigest()
    return result


def csv_write(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
