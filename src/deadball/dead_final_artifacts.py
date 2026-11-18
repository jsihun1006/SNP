"""Shared file and checkpoint helpers for the Dead Final experiments."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_SRC), str(_SRC.parent)]

from hashlib import file_digest
import json

import torch

from transportmer_backbone import TranSPORTmer


def sha(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pending.replace(path)


def write_csv(path, frame):
    pending = path.with_suffix(".csv.tmp")
    frame.to_csv(pending, index=False)
    pending.replace(path)


def torch_weights(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def warm_weights(path, expected_training):
    source = torch_weights(path)
    if set(source["train_matches"]) != set(expected_training):
        raise ValueError(f"Wrong training matches in initialization checkpoint {path}")
    source = source["model_state_dict"]
    target = TranSPORTmer("dead", motion_input=True).state_dict()
    projections = {
        "qkv.weight": "in_proj_weight", "qkv.bias": "in_proj_bias",
        "out.weight": "out_proj.weight", "out.bias": "out_proj.bias",
    }
    old_blocks = [f"{stage}.{layer}" for stage in ("coarse", "fine")
                  for layer in ("temporal.0", "temporal.1", "social")]
    for index, block in enumerate(old_blocks):
        for old, new in projections.items():
            destination = f"blocks.{index}.mha.{new}"
            copied = source[f"{block}.{old}"]
            if copied.shape != target[destination].shape or not torch.isfinite(copied).all():
                raise ValueError(f"Incompatible attention projection {destination} from {path}")
            target[destination] = copied.clone()
    return target


def half_keys(matches):
    return [f"{match}_H{half}" for match in matches for half in (1, 2)]
