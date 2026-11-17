import json

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t})

md("""# PathCRF on Colab (public DFL data only)

Runs the published PathCRF model (Kim et al., KDD 2026, github.com/hyunsungkim-ds/pathcrf, MPL-2.0) on a GPU.

**Only public data goes here** (DFL/IDSSE open data, CC-BY 4.0). Never upload the competition match data.

Before running, put these two files (made locally by `make_upload_zip.py`) in Google Drive under `MyDrive/pathcrf/`:
- `tracking_processed.zip` (the repo's preprocessed DFL tracking, 7 matches)
- `event_processed.zip`

Outputs are written to `MyDrive/pathcrf/outputs/`; copy them to `SNP/possession_pathcrf/outputs/pathcrf_repo/` locally.

Runtime → Change runtime type → **GPU**.""")
code("""import torch, subprocess
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')""")
code("""from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/pathcrf'
import os; os.makedirs(DRIVE + '/outputs', exist_ok=True); print(os.listdir(DRIVE))""")
code("""%cd /content
!git clone --depth 1 https://github.com/hyunsungkim-ds/pathcrf.git
%cd /content/pathcrf
!pip -q install torch_geometric rdp kloppy
!mkdir -p data/sportec
!unzip -q -o {DRIVE}/tracking_processed.zip -d data/sportec/
!unzip -q -o {DRIVE}/event_processed.zip -d data/sportec/
!ls data/sportec/tracking_processed""")
md("Local fix for pandas 3 (read-only arrays). Harmless on pandas 2.")
code("""import pandas
print('pandas', pandas.__version__)
!cd /content/pathcrf && git checkout inference.py
p = '/content/pathcrf/inference.py'
src = open(p).read()
for old in ("    poss_prev = poss_player.ffill().bfill().map(poss_dict).to_numpy()\\n",
            "    poss_next = poss_player.bfill().ffill().map(poss_dict).to_numpy()\\n"):
    assert old in src, old
    src = src.replace(old, old[:-1] + ".copy()\\n")
open(p, 'w').write(src)
!grep -n "to_numpy().copy()" /content/pathcrf/inference.py""")
md("""## 1. Published model (trial 140 = fold 6, J03WR9 held out)
`correct_episode_lasts=False`: the repo option that fixes each episode's last node from the next ground-truth event is off, so no event data enters the prediction.""")
code("""import sys, time, numpy as np, pandas as pd, torch
sys.path.insert(0, '/content/pathcrf')
from evaluate import load_trial_args, resolve_model_path
from models.utils import build_model
from inference import inference

def run_trial(trial_dir, match_id, out_tag):
    args = load_trial_args(trial_dir)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args, device=dev)
    model.load_state_dict(torch.load(resolve_model_path(trial_dir, 'state_dict_best_acc.pt'), map_location=dev, weights_only=False))
    model.eval()
    tracking = pd.read_parquet(f'data/sportec/tracking_processed/{match_id}.parquet')
    t0 = time.time()
    with torch.no_grad():
        prev, nxt, micro, stats = inference(model, tracking, ball_feature=args.get('ball_feature', False), use_crf=True,
                                            decode='indep', correct_episode_lasts=False, evaluate=True,
                                            window_seconds=args.get('window_seconds'), fps=25.0, sample_freq=5,
                                            skip_red_phases=True)
    print(match_id, '%.0fs' % (time.time() - t0), {k: round(float(stats[k]), 4) for k in ('edge_acc', 'src_acc', 'dst_acc') if k in stats})
    for name, df in (('prev', prev), ('next', nxt), ('micro', micro)):
        df.to_parquet(f'{DRIVE}/outputs/{match_id}_{out_tag}_raw_{name}.parquet')

run_trial('saved/140', 'J03WR9', 'trial140')""")
md("""## 2. (Optional) Leave-one-match-out for the other 6 matches
The published weights exist only for fold 6. To score all 7 DFL matches fairly, train folds 0-5 (test = fold index,
validation = the file before it). Full 50 epochs at stride 1 is many GPU hours; `EPOCHS` and `STRIDE` below trade
fidelity for time. Folds are sorted file names: 0 J03WMX, 1 J03WN1, 2 J03WOH, 3 J03WOY, 4 J03WPY, 5 J03WQQ, 6 J03WR9.""")
code("""EPOCHS, STRIDE = 10, 5
FILES = sorted(f[:-8] for f in os.listdir('data/sportec/tracking_processed') if f.endswith('.parquet'))
for fold in range(6):
    trial = 900 + fold
    cmd = (f'python train.py --trial {trial} --sources sportec --lomocv --cv_fold_idx {fold} --agent_model set_tf '
           f'--seq_model sab --crf_model dynamic_sparse_crf --macro_type poss_edge --micro_type poss_edge '
           f'--node_in_dim 8 --target_fps 5 --window_stride {STRIDE} --flip_pitch --seq_dim 128 --crf_edge_dim 16 '
           f'--emit_loss --src_loss --dst_loss --ce_epochs 0 --macro_weight 1 --rloss_weight 0 --crf_weight 1 '
           f'--n_epochs {EPOCHS} --batch_size 32 --start_lr 1e-3 --min_lr 1e-5 --weight_decay 1e-4 --seed 100 --print_batch 200')
    print(cmd); t0 = time.time()
    subprocess.run(cmd, shell=True, check=True)
    print('fold', fold, 'trained in %.0f min' % ((time.time() - t0) / 60))
    run_trial(f'saved/{trial}', FILES[fold], f'fold{fold}')
    subprocess.run(f'cp -r saved/{trial} {DRIVE}/outputs/', shell=True)""")

md("""## 3. Metrica sample games (public, never seen by the model)
Upload `metrica_tracking.zip` (made locally by `possession_pathcrf/metrica_to_pathcrf.py`) anywhere in Drive.
Episodes, phases and running features are computed with the repo's own functions; no labels are used.""")
code("""import glob, os, subprocess, time
from datatools import utils
zips = glob.glob('/content/drive/MyDrive/**/metrica_tracking.zip', recursive=True)
print('found', zips)
subprocess.run(['unzip', '-q', '-o', zips[0], '-d', '/content/pathcrf/data/'], check=True)
args = load_trial_args('saved/140')
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model(args, device=dev)
model.load_state_dict(torch.load(resolve_model_path('saved/140', 'state_dict_best_acc.pt'), map_location=dev, weights_only=False))
model.eval()
for g in (1, 2, 3):
    t0 = time.time()
    tr = pd.read_parquet(f'/content/pathcrf/data/metrica_tracking/metrica{g}.parquet')
    tr, _ = utils.label_frames_and_episodes(tr)
    tr, _ = utils.label_phases(tr)
    tr = utils.calculate_running_features(tr)
    with torch.no_grad():
        prev, nxt, micro, stats = inference(model, tr, ball_feature=False, use_crf=True, decode='indep',
                                            correct_episode_lasts=False, evaluate=False,
                                            window_seconds=args.get('window_seconds'), fps=25.0, sample_freq=5,
                                            skip_red_phases=True)
    for name, df in (('prev', prev), ('next', nxt), ('micro', micro)):
        df.to_parquet(f'{DRIVE}/outputs/metrica{g}_trial140_raw_{name}.parquet')
    print(f'metrica{g}', '%.0fs' % (time.time() - t0), prev.shape)
!ls -la {DRIVE}/outputs""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("pathcrf_colab.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("written")
