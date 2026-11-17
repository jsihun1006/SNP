"""Run the published PathCRF model (trial 140, fold 6: J03WR9 held out) and save its per-frame node probabilities.

    python possession_pathcrf/run_pathcrf_repo.py [J03WR9]

Uses the cloned repository at ../pathcrf_repo (MPL-2.0, Kim et al., KDD 2026) unchanged except a local pandas-3 fix.
correct_episode_lasts=False: the repo's option that fixes each episode's last node from the next ground-truth
event is NOT used, so nothing from the event data enters the prediction.
Output: outputs/pathcrf_repo/<match>_probs.csv.gz with our frame index and player index:
  frame, player, p_sender, p_receiver   (macro_prev / macro_next of the model, players only)
and <match>_decoded.csv.gz: frame, sender, receiver (the model's own CRF decoding, our player index; -1 = outside).
"""

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.join(os.path.dirname(ROOT), "pathcrf_repo")
OUT = os.path.join(HERE, "outputs", "pathcrf_repo")


def our_player_index(match_id):
    """Map the repo's node names (home_<jersey>, away_<jersey>) to our DFL player index (features.match_from_dfl)."""
    sys.path[:0] = [os.path.join(ROOT, "src")]
    import dfl_load
    tracking, _, meta = dfl_load.load_match(match_id)
    team_of = dict(zip(meta["player_id"], meta["team"]))
    jersey_of = dict(zip(meta["player_id"], meta["jersey"]))
    ids = [c[:-2] for c in tracking.columns if c.endswith("_x") and not c.startswith("ball")]
    ids = [i for i in ids if i in team_of]
    name_to_idx = {"{}_{}".format(team_of[i], int(jersey_of[i])): k for k, i in enumerate(ids)}
    row_of = {int(f): r for r, f in enumerate(tracking["frame_id"].to_numpy())}
    return name_to_idx, row_of, len(tracking), len(ids)


def to_frame_ids(index, tracking):
    """The repo's outputs are indexed either by tracking row or by frame_id; return frame_id in both cases."""
    idx = np.asarray(index)
    fid = tracking["frame_id"].to_numpy()
    if idx.max() >= len(tracking) or idx.min() >= fid.min():
        return idx.astype(np.int64)            # already frame_id
    return fid[idx]


def main():
    match_id = sys.argv[1] if len(sys.argv) > 1 else "J03WR9"
    name_to_idx, row_of, n_rows, n_players = our_player_index(match_id)
    cwd = os.getcwd()
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    import torch
    from evaluate import load_trial_args, resolve_model_path
    from models.utils import build_model
    from inference import inference
    args = load_trial_args("saved/140")
    device = torch.device("cpu")
    model = build_model(args, device=device)
    model.load_state_dict(torch.load(resolve_model_path("saved/140", "state_dict_best_acc.pt"), map_location=device,
                                     weights_only=False))
    model.eval()
    tracking = pd.read_parquet("data/sportec/tracking_processed/{}.parquet".format(match_id))
    t0 = time.time()
    with torch.no_grad():
        prev, nxt, micro, stats = inference(model, tracking, ball_feature=args.get("ball_feature", False), use_crf=True,
                                            decode="indep", correct_episode_lasts=False, evaluate=True,
                                            window_seconds=args.get("window_seconds"), fps=25.0, sample_freq=5,
                                            skip_red_phases=True)
    os.chdir(cwd)
    os.makedirs(OUT, exist_ok=True)
    # save the raw model output first, so a later mapping error never costs the inference again
    prev.to_parquet(os.path.join(OUT, "{}_raw_prev.parquet".format(match_id)))
    nxt.to_parquet(os.path.join(OUT, "{}_raw_next.parquet".format(match_id)))
    micro.to_parquet(os.path.join(OUT, "{}_raw_micro.parquet".format(match_id)))
    print("inference {:.0f}s; repo stats: edge_acc {:.4f} src_acc {:.4f} dst_acc {:.4f}".format(
        time.time() - t0, stats.get("edge_acc", np.nan), stats.get("src_acc", np.nan), stats.get("dst_acc", np.nan)))
    frame_ids = to_frame_ids(prev.index, tracking)
    rows = np.array([row_of.get(int(f), -1) for f in frame_ids])
    keep = rows >= 0
    long = []
    for name in prev.columns:
        j = name_to_idx.get(name)
        if j is None:
            continue
        long.append(pd.DataFrame({"frame": rows[keep], "player": j, "p_sender": prev[name].to_numpy()[keep],
                                  "p_receiver": nxt[name].to_numpy()[keep]}))
    pd.concat(long).round(5).to_csv(os.path.join(OUT, "{}_probs.csv.gz".format(match_id)), index=False)
    dec_rows = np.array([row_of.get(int(f), -1) for f in to_frame_ids(micro.index, tracking)])
    ok = dec_rows >= 0
    pd.DataFrame({"frame": dec_rows[ok],
                  "sender": [name_to_idx.get(s, -1) for s in micro["edge_src"].to_numpy()[ok]],
                  "receiver": [name_to_idx.get(s, -1) for s in micro["edge_dst"].to_numpy()[ok]]}).to_csv(
        os.path.join(OUT, "{}_decoded.csv.gz".format(match_id)), index=False)
    unmapped = [c for c in prev.columns if c not in name_to_idx and not c.startswith("out")]
    print("saved; frames mapped {}/{}, our rows {}, players {}, unmapped node names {}".format(
        int(keep.sum()), len(keep), n_rows, n_players, unmapped[:5]))


if __name__ == "__main__":
    main()
