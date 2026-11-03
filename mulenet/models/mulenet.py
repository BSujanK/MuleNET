"""MuleNet: attention-free multigraph encoder with three heads.

* edge head      -> laundering logit per transaction
* typology head  -> 8 AMLworld patterns + "unspecified"
* account head   -> mule-risk logit per account (node state + pooled edge scores)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from ..data.schema import TYPOLOGIES
from .layers import MultiGraphLayer, mlp, scatter_max, scatter_mean


@dataclass
class ModelConfig:
    node_num_dim: int
    edge_num_dim: int
    num_banks: int
    num_currencies: int
    num_formats: int
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.1
    multi_edge: bool = True  # ablation: MEGA-style parallel-edge stage
    reverse_mp: bool = True  # ablation: reverse message passing
    pna: bool = True  # PNA aggregators (False -> GIN-style sum)
    edge_update: bool = True
    ego_ids: bool = True
    deg_log_mean: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


class MuleNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.node_in = nn.Linear(cfg.node_num_dim + 1, h)
        self.bank_emb = nn.Embedding(cfg.num_banks, h)
        self.edge_in = nn.Linear(cfg.edge_num_dim, h)
        self.cur_s = nn.Embedding(cfg.num_currencies, h)
        self.cur_r = nn.Embedding(cfg.num_currencies, h)
        self.fmt = nn.Embedding(cfg.num_formats, h)
        self.layers = nn.ModuleList(
            MultiGraphLayer(
                h,
                multi_edge=cfg.multi_edge,
                reverse_mp=cfg.reverse_mp,
                pna=cfg.pna,
                edge_update=cfg.edge_update,
                deg_log_mean=cfg.deg_log_mean,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.layers)
        )
        self.edge_head = mlp(3 * h, h, 1, cfg.dropout)
        self.typ_head = mlp(3 * h, h, len(TYPOLOGIES), cfg.dropout)
        self.acct_head = mlp(h + 4, h, 1, cfg.dropout)

    def encode(self, b, edge_num: torch.Tensor | None = None):
        ego = b.ego if self.cfg.ego_ids else torch.zeros_like(b.ego)
        h = self.node_in(torch.cat([b.node_num, ego.unsqueeze(-1)], dim=-1)) + self.bank_emb(b.node_bank)
        x = b.edge_num if edge_num is None else edge_num
        e = self.edge_in(x) + self.cur_s(b.edge_cat[:, 0]) + self.cur_r(b.edge_cat[:, 1]) + self.fmt(b.edge_cat[:, 2])
        for layer in self.layers:
            h, e = layer(h, e, b)
        return h, e

    def forward(self, b, edge_num: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        h, e = self.encode(b, edge_num)
        z = torch.cat([h[b.src], h[b.dst], e], dim=-1)
        edge_logit = self.edge_head(z).squeeze(-1)
        n = h.shape[0]
        el = edge_logit.detach().unsqueeze(-1)
        pooled = torch.cat(
            [
                scatter_max(el, b.dst, n),
                scatter_mean(el, b.dst, n),
                scatter_max(el, b.src, n),
                scatter_mean(el, b.src, n),
            ],
            dim=-1,
        )
        acct_logit = self.acct_head(torch.cat([h, pooled], dim=-1)).squeeze(-1)
        return {
            "edge_logit_all": edge_logit,
            "edge_logit": edge_logit[b.seed_pos],
            "typ_logit": self.typ_head(z[b.seed_pos]),
            "typ_logit_all": self.typ_head(z),
            "acct_logit": acct_logit[b.seed_nodes],
            "acct_logit_all": acct_logit,
        }
