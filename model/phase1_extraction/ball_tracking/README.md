# ball_tracking (Phase 1) — placeholder

Ball-tracking correction of the provided ATD (owner: teammates). Not yet integrated in this branch.

Contract when added (see `model/README.md`):

- entry point `extract.py`, checks in `validate.py`, settings in `config.json`, pinned `requirements.txt`
- output `outputs/phase1/ball_tracking/<run_id>/tracking.csv`
  (`frame,time_seconds,track_id,id_kind,object_type,team,x_norm,y_norm,x_m,y_m,confidence,observed`),
  `frame_status.csv`, `run_manifest.json`, `validation.json`
- corrections come out of code only; nothing is fixed by hand on the provided match (RULES.md)
