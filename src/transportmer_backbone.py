"""Attention and trajectory model shared by the TranSPORTmer experiments."""

import math

import torch
from torch import nn

WIDTH, HEADS, FFN = 128, 16, 512
WINDOW, STEP, FPS = 60, 4, 25


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(WIDTH, HEADS, dropout=0.0, batch_first=True)
        self.norm_attention = nn.LayerNorm(WIDTH)
        self.norm_feed = nn.LayerNorm(WIDTH)
        self.feed = nn.Sequential(nn.Linear(WIDTH, FFN), nn.GELU(), nn.Linear(FFN, WIDTH))

    def forward(self, x, mask=None):
        keys = x
        if mask is not None:
            # A fully hidden trajectory has no valid key. A zero sentinel keeps
            # the attention finite while the query still carries its type and time.
            all_hidden = mask.all(dim=-1)
            keys = x.masked_fill(all_hidden[:, None, None], 0)
            mask = mask & ~(all_hidden[:, None] & (torch.arange(x.shape[1], device=x.device) == 0))
        y = self.mha(x, keys, keys, key_padding_mask=mask, need_weights=False)[0]
        x = self.norm_attention(x + y)
        return self.norm_feed(x + self.feed(x))


class TranSPORTmer(nn.Module):
    def __init__(self, task, use_events=False, motion_input=False):
        super().__init__()
        self.task = task
        self.motion_input = motion_input
        self.position = nn.Linear(5 if motion_input else 2, WIDTH)
        self.agent_type = nn.Embedding(4, WIDTH)
        self.source = nn.Embedding(3, WIDTH)
        self.event_input = nn.Linear(7, WIDTH) if use_events else None
        if task == "possession":
            self.restart = nn.Embedding(2, WIDTH)
        self.blocks = nn.ModuleList(Attention() for _ in range(6))
        self.readout = nn.Linear(WIDTH, {"ball": 2, "dead": 1, "possession": 1}[task])
        dims = torch.exp(torch.arange(0, WIDTH, 2) * (-math.log(10000) / WIDTH))
        angle = torch.arange(WINDOW)[:, None] * dims
        self.register_buffer("time", torch.stack([angle.sin(), angle.cos()], -1).flatten(1))

    def forward(self, xy, status, kinds, event_values=None, *, motion=None, padding=None, restart=None):
        batch, time, agents, _ = xy.shape
        coordinates = xy
        if self.task == "possession":
            centered = torch.where((status > 0)[..., None], 2 * xy - 1, 0)
            coordinates = torch.cat((centered, motion), dim=-1)
        elif self.motion_input:
            present = status > 0
            continuous = present[:, 1:] & present[:, :-1]
            # Displacements use the same slot and two consecutive observations.
            # Scale 6.25 Hz normalized displacements into a useful numerical range.
            change = 100 * (xy[:, 1:] - xy[:, :-1])
            change = torch.where(continuous[..., None], change, 0)
            change = torch.cat((torch.zeros_like(xy[:, :1]), change), dim=1)
            valid = torch.cat((torch.zeros_like(present[:, :1]), continuous), dim=1)
            centered = torch.where(present[..., None], 2 * xy - 1, 0)
            coordinates = torch.cat((centered, change, valid[..., None]), dim=-1)
        agent_type = self.agent_type(kinds)
        agent_type = agent_type[None, None] if kinds.ndim == 1 else agent_type[:, None]
        x = self.position(coordinates) + agent_type + self.source(status)
        if self.task != "ball":
            cls = self.agent_type.weight[3].expand(batch, time, 1, WIDTH)
            if self.event_input is not None:
                cls = cls + self.event_input(event_values).unsqueeze(2)
            if self.task == "possession":
                cls = cls + self.restart(restart).unsqueeze(2)
            x = torch.cat((x, cls), dim=2)
            status = torch.cat((status, torch.zeros_like(status[:, :, :1])), dim=2)
            agents += 1
        hidden = status == 0
        if padding is not None:
            time_padding = padding[:, None, :].expand(batch, agents, time).reshape(batch * agents, time)
            absent_slots = hidden.all(dim=1)
            absent_slots[:, -1] = False  # the per-frame classifier is a valid social token
            social_padding = absent_slots[:, None, :].expand(batch, time, agents) | padding[:, :, None]
            social_padding = social_padding.reshape(batch * time, agents)
        for stage in range(2):
            x = x + self.time[:time][None, :, None, :]
            x = x.transpose(1, 2).reshape(batch * agents, time, WIDTH)
            temporal = hidden.transpose(1, 2).reshape(batch * agents, time) if stage == 0 else None
            if padding is not None:
                temporal = (temporal | time_padding) if stage == 0 else time_padding
            for index in (0, 1):
                x = self.blocks[stage * 3 + index](x, temporal)
            x = x.reshape(batch, agents, time, WIDTH).transpose(1, 2).reshape(batch * time, agents, WIDTH)
            social = hidden.reshape(batch * time, agents) if stage == 0 else None
            if padding is not None:
                social = (social | social_padding) if stage == 0 else social_padding
            x = self.blocks[stage * 3 + 2](x, social).reshape(batch, time, agents, WIDTH)
        return self.readout(x[:, :, -1])
