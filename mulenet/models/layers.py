"""Attention-free multigraph message passing.

* Stage 1 (MEGA-GNN style): messages on parallel transfers between the same two
  accounts are first aggregated per (sender, receiver) pair with sum/mean/max/std.
* Stage 2 (PNA): pair messages are aggregated per node with mean/min/max/std and
  degree scalers, separately for incoming and outgoing directions (Multi-GNN
  reverse message passing).
* Edge-update MLP after each layer, residuals and batch norm.

No attention anywhere: every aggregation is a fixed permutation-invariant
function.
"""

from __future__ import annotations

import torch
from torch import nn


def scatter_sum(x: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
    return x.new_zeros((n,) + x.shape[1:]).index_add_(0, idx, x)


def scatter_count(idx: torch.Tensor, n: int, like: torch.Tensor) -> torch.Tensor:
    return like.new_zeros(n).index_add_(0, idx, like.new_ones(idx.shape[0]))


def scatter_mean(x, idx, n, count=None):
    count = scatter_count(idx, n, x) if count is None else count
    return scatter_sum(x, idx, n) / count.clamp(min=1).unsqueeze(-1)


def _scatter_ext(x, idx, n, reduce: str):
    fill = float("-inf") if reduce == "amax" else float("inf")
    out = x.new_full((n,) + x.shape[1:], fill)
    out = out.scatter_reduce(0, idx.unsqueeze(-1).expand_as(x), x, reduce=reduce, include_self=True)
    return torch.where(torch.isinf(out), torch.zeros_like(out), out)


def scatter_max(x, idx, n):
    return _scatter_ext(x, idx, n, "amax")


def scatter_min(x, idx, n):
    return _scatter_ext(x, idx, n, "amin")


def scatter_std(x, idx, n, count=None):
    mean = scatter_mean(x, idx, n, count)
    sq = scatter_mean(x * x, idx, n, count)
    return torch.sqrt(torch.relu(sq - mean * mean) + 1e-5)


def mlp(i: int, h: int, o: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, o))


class MultiEdgeAggregation(nn.Module):
    """Stage 1: aggregate parallel-edge messages into one message per pair."""

    def __init__(self, hidden: int, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.proj = nn.Linear(4 * hidden, hidden) if enabled else None

    def forward(self, msg, pair, num_pairs):
        if not self.enabled:
            return msg
        cnt = scatter_count(pair, num_pairs, msg)
        agg = torch.cat(
            [
                scatter_sum(msg, pair, num_pairs),
                scatter_mean(msg, pair, num_pairs, cnt),
                scatter_max(msg, pair, num_pairs),
                scatter_std(msg, pair, num_pairs, cnt),
            ],
            dim=-1,
        )
        return self.proj(agg)


class NeighbourAggregation(nn.Module):
    """Stage 2: PNA aggregators + degree scalers, or GIN-style sum."""

    def __init__(self, hidden: int, pna: bool = True, deg_log_mean: float = 1.0):
        super().__init__()
        self.pna = pna
        self.register_buffer("delta", torch.tensor(float(max(deg_log_mean, 1e-3))))
        self.proj = nn.Linear(12 * hidden, hidden) if pna else nn.Identity()

    def forward(self, msg, target, n):
        if not self.pna:
            return scatter_sum(msg, target, n)
        cnt = scatter_count(target, n, msg)
        aggs = torch.cat(
            [
                scatter_mean(msg, target, n, cnt),
                scatter_min(msg, target, n),
                scatter_max(msg, target, n),
                scatter_std(msg, target, n, cnt),
            ],
            dim=-1,
        )
        logd = torch.log(cnt + 1).unsqueeze(-1)
        amp = logd / self.delta
        att = torch.where(logd > 0, self.delta / logd.clamp(min=1e-6), torch.ones_like(logd))
        return self.proj(torch.cat([aggs, aggs * amp, aggs * att], dim=-1))


class MultiGraphLayer(nn.Module):
    def __init__(
        self,
        hidden: int,
        multi_edge: bool = True,
        reverse_mp: bool = True,
        pna: bool = True,
        edge_update: bool = True,
        deg_log_mean: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.reverse_mp = reverse_mp
        self.edge_update = edge_update
        self.msg_in = mlp(2 * hidden, hidden, hidden)
        self.pair_in = MultiEdgeAggregation(hidden, multi_edge)
        self.agg_in = NeighbourAggregation(hidden, pna, deg_log_mean)
        if reverse_mp:
            self.msg_out = mlp(2 * hidden, hidden, hidden)
            self.pair_out = MultiEdgeAggregation(hidden, multi_edge)
            self.agg_out = NeighbourAggregation(hidden, pna, deg_log_mean)
        n_in = (3 if reverse_mp else 2) * hidden
        self.update = mlp(n_in, hidden, hidden, dropout)
        self.norm = nn.BatchNorm1d(hidden)
        if edge_update:
            self.edge_mlp = mlp(3 * hidden, hidden, hidden, dropout)
            self.edge_norm = nn.BatchNorm1d(hidden)

    def _direction(self, h, e, b, msg_fn, pair_fn, agg_fn, incoming: bool):
        n = h.shape[0]
        sender = b.src if incoming else b.dst
        m = msg_fn(torch.cat([h[sender], e], dim=-1))
        if pair_fn.enabled:
            m = pair_fn(m, b.pair, b.num_pairs)
            target = b.pair_dst if incoming else b.pair_src
        else:
            target = b.dst if incoming else b.src
        return agg_fn(m, target, n)

    def forward(self, h, e, b):
        parts = [h, self._direction(h, e, b, self.msg_in, self.pair_in, self.agg_in, True)]
        if self.reverse_mp:
            parts.append(self._direction(h, e, b, self.msg_out, self.pair_out, self.agg_out, False))
        h = h + torch.relu(self.norm(self.update(torch.cat(parts, dim=-1))))
        if self.edge_update:
            e = e + torch.relu(self.edge_norm(self.edge_mlp(torch.cat([h[b.src], h[b.dst], e], dim=-1))))
        return h, e
