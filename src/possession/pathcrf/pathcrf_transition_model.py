"""Training and Viterbi scoring for the DFL transition-head comparison."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TransitionPathCRF(nn.Module):
    def __init__(self, backbone, width, has_head):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(4 * width, 64), nn.ReLU(), nn.Linear(64, 3)) if has_head else None
        self.fine = None
        if has_head:
            backbone.micro_sab_t2.register_forward_hook(self.capture_fine)

        source, destination = backbone.comp_src.cpu().numpy(), backbone.comp_dst.cpu().numpy()
        node_count = 2 * backbone.team_size + 4
        groups = [np.flatnonzero(destination == node) for node in range(node_count)]
        incoming = np.full((node_count, max(map(len, groups))), -1, dtype=np.int64)
        for node, edges in enumerate(groups):
            incoming[node, :len(edges)] = edges
        legal = backbone.crf.inc_mask.cpu().numpy()
        allowed = backbone.crf.inc_idx.cpu().numpy()
        for edge, (src, dst) in enumerate(zip(source, destination, strict=True)):
            expected = set(groups[src]) | ({edge} if src != dst else set())
            if expected != set(allowed[edge, legal[edge]]):
                raise ValueError("The PathCRF predecessor graph differs from the pinned upstream")
        self.register_buffer("incoming_valid", torch.from_numpy(incoming >= 0), persistent=False)
        self.register_buffer("incoming_by_destination", torch.from_numpy(incoming.clip(min=0)), persistent=False)

    def capture_fine(self, module, inputs, output):
        self.fine = output

    def forward(self, features, home_first):
        macro, logits, embeddings = self.backbone(features)
        head_logits = None
        if self.head is not None:
            batch, steps, nodes, _ = features.shape
            fine = self.fine.reshape(batch, nodes, steps, -1).transpose(1, 2)
            self.fine = None
            teams = features[..., 0]
            left = teams.eq(-1).unsqueeze(-1)
            right = teams.eq(1).unsqueeze(-1)
            left = (fine * left).sum(dim=2) / left.sum(dim=2).clamp_min(1)
            right = (fine * right).sum(dim=2) / right.sum(dim=2).clamp_min(1)
            home = torch.where(home_first[:, None, None], left, right)
            away = torch.where(home_first[:, None, None], right, left)
            current = torch.cat([home, away], dim=-1)
            previous = torch.cat([current[:, :1], current[:, :-1]], dim=1)
            head_logits = self.head(torch.cat([previous, current], dim=-1))
        return macro, logits, embeddings, head_logits

    def objective(self, features, home_first, edges, targets, weights, class_weights, *,
                  base_scale=1, head_total=None):
        macro, logits, embeddings, head_logits = self(features, home_first)
        logits, macro = logits.float(), macro.float()
        nodes = features.shape[2]
        edge_ids = edges[..., 0] * nodes + edges[..., 1]
        edge_scores = logits.reshape(-1, nodes, nodes)
        sender, receiver = edges[..., 0].flatten(), edges[..., 1].flatten()
        sender_loss = F.cross_entropy(macro[..., 0].reshape(-1, nodes), sender)
        receiver_loss = F.cross_entropy(macro[..., 1].reshape(-1, nodes), receiver)
        emission_loss = F.cross_entropy(logits.flatten(0, 1), edge_ids.flatten())
        emission_loss += F.cross_entropy(torch.logsumexp(edge_scores, dim=2), sender)
        emission_loss += F.cross_entropy(torch.logsumexp(edge_scores, dim=1), receiver)
        backbone = self.backbone
        valid = backbone.valid_orig_edge_ids
        tags = backbone.orig2comp[edge_ids]
        crf_loss = self.crf_nll(logits.index_select(2, valid), embeddings.index_select(2, valid).float(), tags)
        loss = (sender_loss + receiver_loss + emission_loss + crf_loss / features.shape[1]) * base_scale
        if head_logits is not None:
            event_loss = F.cross_entropy(
                head_logits.float().flatten(0, 1), targets.flatten(), weight=class_weights,
                ignore_index=-1, reduction="none",
            ).reshape_as(weights)
            if head_total is None:
                head_total = torch.where(targets >= 0, class_weights[targets.clamp_min(0)] * weights, 0).sum()
            loss = loss + 0.25 * (weights * event_loss).sum() / head_total.clamp_min(1)
        return loss

    def crf_nll(self, emissions, embeddings, tags):
        """Compute the original learned linear transition score using destination groups."""
        backbone = self.backbone
        crf = backbone.crf
        previous, current = edge_projections(crf, embeddings)
        src, dst = backbone.comp_src, backbone.comp_dst
        continuations = src != dst
        incoming = self.incoming_by_destination.flatten()
        mask = self.incoming_valid
        batch = emissions.shape[0]
        alpha = emissions[:, 0]
        for t in range(1, emissions.shape[1]):
            prior = alpha + previous[:, t-1]
            by_destination = prior.index_select(1, incoming).reshape(batch, *mask.shape)
            summed = torch.logsumexp(by_destination.masked_fill(~mask, crf.forbid_score), dim=-1)
            from_receiver = summed.index_select(1, src)
            from_same_edge = prior.masked_fill(~continuations, crf.forbid_score)
            # MPS logaddexp can overflow on a long, unnormalized path.
            maximum = torch.maximum(from_receiver, from_same_edge)
            difference = (from_receiver - from_same_edge).abs()
            alpha = maximum + torch.log1p(torch.exp(-difference)) + current[:, t] + emissions[:, t]
        log_partition = torch.logsumexp(alpha, dim=1)
        gold = emissions.gather(2, tags[..., None]).squeeze(-1).sum(dim=1)
        gold += previous[:, :-1].gather(2, tags[:, :-1, None]).squeeze(-1).sum(dim=1)
        gold += current[:, 1:].gather(2, tags[:, 1:, None]).squeeze(-1).sum(dim=1)
        return (log_partition - gold).mean()


def edge_projections(crf, embeddings):
    previous, current = crf.linear.weight.split(crf.edge_embed_dim, dim=1)
    return F.linear(embeddings, previous).squeeze(-1), F.linear(embeddings, current, crf.linear.bias).squeeze(-1)


def decode_paths(emissions, previous, current, incoming, allowed, direction, bonus):
    """Score lawful PathCRF paths for one episode and one or more bonus settings."""
    variants, steps, _ = bonus.shape
    state_count = emissions.shape[1]
    delta = np.broadcast_to(emissions[0], (variants, state_count)).copy()
    best_predecessors = np.empty((steps, variants, state_count), dtype=np.int16)
    states = np.arange(state_count)[None, :, None]
    settings = np.arange(variants)[:, None, None]
    for t in range(1, steps):
        value = delta[:, incoming] + (previous[t-1, incoming] + current[t, :, None])
        if bonus[:, t].any():
            value += bonus[:, t, direction]
        value = np.where(allowed, value, -1e4)
        best = value.argmax(axis=-1)[..., None]
        delta = value[settings, states, best].squeeze(-1) + emissions[t]
        best_predecessors[t] = incoming[states, best].squeeze(-1)
    result = np.empty((variants, steps), dtype=np.int64)
    result[:, -1] = delta.argmax(axis=1)
    for t in range(steps-1, 0, -1):
        result[:, t-1] = best_predecessors[t, np.arange(variants), result[:, t]]
    return result
