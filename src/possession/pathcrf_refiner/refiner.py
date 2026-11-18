"""25 Hz team-possession and directed transition model."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def safe_mask(mask):
    # PyTorch attention needs at least one key, including for padded sequences.
    mask = mask.clone()
    mask[:, 0] |= ~mask.any(dim=1)
    return ~mask


class InducedAttention(nn.Module):
    def __init__(self, width, heads, count, dropout):
        super().__init__()
        self.points = nn.Parameter(torch.randn(count, width) / math.sqrt(width))
        self.summarize = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.broadcast = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.point_norm = nn.LayerNorm(width)
        self.norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.out = nn.LayerNorm(width)

    def forward(self, tokens, visible):
        tokens = tokens.contiguous()
        points = self.points.expand(len(tokens), -1, -1).contiguous()
        summary = self.summarize(points, tokens, tokens, key_padding_mask=safe_mask(visible),
                                 need_weights=False)[0]
        summary = self.point_norm(points + summary)
        values = self.broadcast(tokens, summary, summary, need_weights=False)[0]
        values = self.norm(tokens + values)
        return self.out(values + self.ffn(values)).masked_fill(~visible[..., None], 0.0)


class SpaceTimeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, heads = config["width"], config["heads"]
        attention = (width, heads, config["inducing_points"], config["dropout"])
        self.team = InducedAttention(*attention)
        self.scene = InducedAttention(*attention)
        layer = nn.TransformerEncoderLayer(width, heads, config["ffn_width"],
                                           config["dropout"], activation="gelu", batch_first=True)
        self.temporal = nn.TransformerEncoder(layer, config["temporal_layers"], enable_nested_tensor=False)
        position = torch.arange(config["refine_window_frames"], dtype=torch.float32)[:, None]
        scale = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
        encoding = torch.stack(((position * scale).sin(), (position * scale).cos()), dim=-1)
        self.register_buffer("position", encoding.flatten(1), persistent=False)

    def forward(self, x, visible):
        batch, steps, count, width = x.shape
        side = (count - 1) // 2
        instant = x.flatten(0, 1)
        observed = visible.flatten(0, 1)
        players = self.team(instant[:, :-1].reshape(-1, side, width), observed[:, :-1].reshape(-1, side))
        instant = torch.cat((players.reshape(batch * steps, 2 * side, width), instant[:, -1:]), dim=1)
        instant = self.scene(instant, observed).reshape(batch, steps, count, width)
        by_track = instant.transpose(1, 2).reshape(batch * count, steps, width)
        mask = visible.transpose(1, 2).reshape(batch * count, steps)
        track = self.temporal(by_track + self.position[:steps], src_key_padding_mask=safe_mask(mask))
        track = track.masked_fill(~mask[..., None], 0.0)
        return track.reshape(batch, count, steps, width).transpose(1, 2)


class Refiner(nn.Module):
    def __init__(self, config, n_features):
        super().__init__()
        width = config["width"]
        self.project = nn.Linear(n_features, width)
        self.blocks = nn.ModuleList(SpaceTimeBlock(config) for _ in range(config["blocks"]))
        self.importance = nn.Linear(width, 1)
        self.scene = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Dropout(config["dropout"]))
        self.state = nn.Linear(width, 2)
        self.transition = nn.Linear(2 * width, 3)

    def forward(self, inputs, visible):
        x = self.project(inputs)
        x = x.masked_fill(~visible[..., None], 0.0)
        for block in self.blocks:
            x = block(x, visible)
        teams = x[:, :, :-1].reshape(*x.shape[:2], 2, -1, x.shape[-1])
        observed = visible[:, :, :-1].reshape(*x.shape[:2], 2, -1)
        logits = self.importance(teams).squeeze(-1).masked_fill(~observed, -1e4)
        weights = torch.softmax(logits, dim=-1) * observed
        pooled = (weights[..., None] * teams).sum(dim=-2).flatten(-2)
        scene = self.scene(torch.cat((pooled, x[:, :, -1]), dim=-1))
        previous = torch.cat((scene[:, :1], scene[:, :-1]), dim=1)
        return self.state(scene), self.transition(torch.cat((previous, scene), dim=-1))


TRANSITION_IDS = ((0, 1), (2, 0))


def crf_sum(state, change, labels):
    """Sum 2-state negative log-likelihood over contiguous supervised spans."""
    state, change = state.float(), change.float()
    valid = labels >= 0
    labels = labels.clamp_min(0)
    alpha = state[:, 0]
    init = state.new_zeros(len(state))
    total = state.new_zeros(())
    target = state.gather(-1, labels[..., None]).squeeze(-1)
    table = change[:, 1:, [0, 1, 2, 0]].reshape(*change[:, 1:].shape[:2], 2, 2)
    pair = 2 * labels[:, :-1] + labels[:, 1:]
    steps = table.flatten(-2).gather(-1, pair[..., None]).squeeze(-1)
    gold = (target * valid).sum() + (steps * (valid[:, 1:] & valid[:, :-1])).sum()
    for t in range(1, state.shape[1]):
        end = valid[:, t - 1] & ~valid[:, t]
        total = total + torch.where(end, torch.logsumexp(alpha, dim=-1), init).sum()
        continuation = torch.logsumexp(alpha[:, :, None] + table[:, t - 1], dim=1) + state[:, t]
        alpha = torch.where(valid[:, t - 1, None], continuation, state[:, t])
    total = total + torch.where(valid[:, -1], torch.logsumexp(alpha, dim=-1), init).sum()
    return total - gold


def loss_sums(state, change, labels, transitions, class_weight):
    state_loss = F.cross_entropy(state.float().transpose(1, 2), labels, ignore_index=-1, reduction="sum")
    change_loss = F.cross_entropy(change.float().transpose(1, 2), transitions, weight=class_weight,
                                  ignore_index=-1, reduction="sum")
    return torch.stack((state_loss, change_loss, crf_sum(state, change, labels)))


def decode(state, change, left=None, right=None, right_change=None):
    """Decode a patch while scoring transitions to its adjacent fixed H/A frames."""
    index = np.asarray(TRANSITION_IDS)
    table = change[:, index]
    dp = state[0].copy()
    back = np.zeros((len(state), 2), dtype=np.int8)
    if left in (0, 1):
        dp += table[0, left]
    for t in range(1, len(state)):
        step = dp[:, None] + table[t]
        back[t] = step.argmax(axis=0)
        dp = state[t] + step.max(axis=0)
    if right in (0, 1):
        boundary = right_change if right_change is not None else change[-1]
        dp += boundary[index][:, right]
    path = np.empty(len(state), dtype=np.int8)
    path[-1] = dp.argmax()
    for t in range(len(state) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
