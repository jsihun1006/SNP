"""Focused regression checks for CRF, padding, and learned pooling."""
from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import numpy as np
import torch

from inputs import N_FEATURES
from refiner import Refiner, crf_sum, decode
from run import Window, collate, read_config, CONFIGS

torch.set_num_threads(2)
torch.manual_seed(100)
state = torch.randn(1, 4, 2, requires_grad=True)
change = torch.randn(1, 4, 3, requires_grad=True)
labels = torch.tensor([[0, 1, -1, 0]])
loss = crf_sum(state, change, labels)
valid_segments = [(0, 2), (3, 4)]
reference = state.new_zeros(())
for a, b in valid_segments:
    values = []
    gold = state.new_zeros(())
    for path in product((0, 1), repeat=b - a):
        v = sum(state[0, t, k] for t, k in zip(range(a, b), path, strict=True))
        for t in range(a + 1, b):
            old, new = path[t - a - 1:t - a + 1]
            v = v + change[0, t, 0 if old == new else 2 - new]
        values.append(v)
        if tuple(labels[0, a:b].tolist()) == path:
            gold = v
    reference = reference + torch.logsumexp(torch.stack(values), 0) - gold
torch.testing.assert_close(loss, reference)
loss.backward()
assert torch.isfinite(state.grad).all() and torch.isfinite(change.grad).all()

s, t = state.detach().numpy()[0], change.detach().numpy()[0]
for left, right in product((None, 0, 1), repeat=2):
    path = decode(s, t, left, right)
    possibilities = list(product((0, 1), repeat=len(s)))
    def score(p):
        v = sum(s[i, k] for i, k in enumerate(p))
        for i in range(1, len(p)):
            v += t[i, 0 if p[i - 1] == p[i] else 2 - p[i]]
        if left is not None:
            v += t[0, 0 if left == p[0] else 2 - p[0]]
        if right is not None:
            v += t[-1, 0 if p[-1] == right else 2 - right]
        return v
    assert score(path) >= max(map(score, possibilities)) - 1e-5

config = read_config(CONFIGS / "parameters.csv")
values = np.zeros((12, 5, N_FEATURES), dtype=np.float32)
values[..., :2] = np.random.default_rng(100).uniform(0, 68, (12, 5, 2))
values[..., 6] = 1
team = np.array([0, 0, 1, 1, 2])
values[..., 9:12] = np.eye(3)[team]
half = SimpleNamespace(features=values, labels=np.zeros(12, dtype=np.int8), team=team)
refs = [Window(0, 0, 12), Window(0, 2, 7)]
x, mask, y, trans = collate([half], refs, config, np.random.default_rng(100))
assert (y[1, 5:] == -1).all()
assert not mask[1, 5:].any()
model = Refiner(config, N_FEATURES).eval()
with torch.inference_mode():
    state_one, _ = model(x[:1, :12], mask[:1, :12])
    state_two, _ = model(x, mask)
torch.testing.assert_close(state_one, state_two[:1], atol=2e-5, rtol=2e-5)
assert torch.isfinite(state_two).all()
print("PASS CRF brute force, unknown-label spans, boundary Viterbi, padding, pooled attention")
