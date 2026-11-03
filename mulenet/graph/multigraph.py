"""Directed multigraph snapshots and edge mini-batch neighbour sampling.

Accounts are nodes and transfers are directed multi-edges (several parallel
transfers can join the same two accounts). Pure NumPy/PyTorch, no PyG needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .features import EDGE_CAT_FEATURES, PORT_FEATURES, node_features


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Normalizer":
        return cls(x.mean(0), x.std(0) + 1e-6)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


@dataclass
class GraphData:
    """One graph snapshot (global, or one bank's view) with local node ids."""

    num_nodes: int
    node_ids: np.ndarray  # local -> global account id
    node_bank: np.ndarray
    node_num: np.ndarray  # [N, F_n]
    src: np.ndarray  # local ids
    dst: np.ndarray
    ts: np.ndarray
    edge_ids: np.ndarray  # local edge -> global edge id
    edge_num: np.ndarray  # [E, F_e] normalised
    edge_cat: np.ndarray  # [E, 3]
    label: np.ndarray
    typology: np.ndarray
    attempt: np.ndarray
    node_label: np.ndarray  # account touched a laundering edge in this snapshot
    own_node: np.ndarray  # 1 if the account belongs to the viewing bank (all 1 in global view)

    @property
    def num_edges(self) -> int:
        return len(self.src)

    def local_edges(self, global_edge_ids: np.ndarray) -> np.ndarray:
        """Map global edge ids to local positions (all must be present)."""
        pos = np.searchsorted(self.edge_ids, global_edge_ids)
        if len(global_edge_ids) and (pos.max() >= len(self.edge_ids) or np.any(self.edge_ids[pos] != global_edge_ids)):
            raise KeyError("some edges are not in this graph")
        return pos


def build_graph(
    tx: pd.DataFrame,
    edge_num_all: np.ndarray,
    edge_ids: np.ndarray,
    normalizer: Normalizer,
    own_bank: int | None = None,
    use_ports: bool = True,
    port_cols: tuple[int, ...] = (),
) -> GraphData:
    """Build a snapshot over ``edge_ids`` (sorted global ids) of the canonical table."""
    edge_ids = np.sort(edge_ids)
    sub = tx.iloc[edge_ids]
    gsrc, gdst = sub.src.to_numpy(), sub.dst.to_numpy()
    nodes, inv = np.unique(np.concatenate([gsrc, gdst]), return_inverse=True)
    m = len(edge_ids)
    src, dst = inv[:m], inv[m:]
    node_bank = np.zeros(len(nodes), dtype=np.int64)
    node_bank[src] = sub.src_bank.to_numpy()
    node_bank[dst] = sub.dst_bank.to_numpy()
    ts = sub.ts.to_numpy()
    label = sub.label.to_numpy().astype(np.int64)
    node_label = np.zeros(len(nodes), dtype=np.int64)
    np.maximum.at(node_label, src, label)
    np.maximum.at(node_label, dst, label)
    edge_num = normalizer(edge_num_all[edge_ids])
    if not use_ports:
        edge_num[:, list(port_cols)] = 0.0
    own = np.ones(len(nodes), dtype=np.int64) if own_bank is None else (node_bank == own_bank).astype(np.int64)
    return GraphData(
        num_nodes=len(nodes),
        node_ids=nodes,
        node_bank=node_bank,
        node_num=node_features(src, dst, ts, len(nodes)),
        src=src,
        dst=dst,
        ts=ts,
        edge_ids=edge_ids,
        edge_num=edge_num,
        edge_cat=sub[EDGE_CAT_FEATURES].to_numpy().astype(np.int64),
        label=label,
        typology=sub.typology.to_numpy().astype(np.int64),
        attempt=sub.attempt_id.to_numpy(),
        node_label=node_label,
        own_node=own,
    )


@dataclass
class Batch:
    """A sampled subgraph around a set of seed (target) edges, as tensors."""

    node_num: torch.Tensor
    node_bank: torch.Tensor
    ego: torch.Tensor  # 1 for endpoints of seed edges (Multi-GNN ego IDs)
    src: torch.Tensor
    dst: torch.Tensor
    pair: torch.Tensor  # parallel-edge group id per edge
    num_pairs: int
    pair_src: torch.Tensor
    pair_dst: torch.Tensor
    edge_num: torch.Tensor
    edge_cat: torch.Tensor
    seed_pos: torch.Tensor  # positions of seed edges among the batch edges
    seed_label: torch.Tensor
    seed_typology: torch.Tensor
    seed_nodes: torch.Tensor  # batch-local ids of seed endpoints
    seed_node_label: torch.Tensor
    node_global: np.ndarray  # batch-local -> graph-local node ids
    edge_local: np.ndarray  # batch edges -> graph-local edge ids

    @property
    def num_nodes(self) -> int:
        return int(self.node_num.shape[0])

    def to(self, device) -> "Batch":
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self


class NeighborSampler:
    """Uniform neighbour sampling around seed edges (100 + 100 by default).

    Each hop draws up to ``k`` incident edges per frontier node, split evenly
    between incoming and outgoing edges (reverse message passing needs both).
    Deduplication uses graph-sized bitmaps instead of sorting, so the cost of a
    batch grows with the batch, not with ``log`` of the graph.
    """

    def __init__(self, g: GraphData, fanouts: list[int] | None = (100, 100)):
        self.g = g
        self.fanouts = list(fanouts) if fanouts else None
        n = g.num_nodes
        self.out_order = np.argsort(g.src, kind="stable")
        self.out_ptr = np.concatenate([[0], np.cumsum(np.bincount(g.src, minlength=n))])
        self.in_order = np.argsort(g.dst, kind="stable")
        self.in_ptr = np.concatenate([[0], np.cumsum(np.bincount(g.dst, minlength=n))])
        # graph-level parallel-edge groups, computed once
        _, self.pair_id = np.unique(g.src.astype(np.int64) * n + g.dst, return_inverse=True)
        self.num_pairs = int(self.pair_id.max()) + 1 if len(self.pair_id) else 0
        self._edge_mask = np.zeros(g.num_edges, dtype=bool)
        self._node_mask = np.zeros(n, dtype=bool)
        self._node_map = np.full(n, -1, dtype=np.int64)
        self._pair_mask = np.zeros(self.num_pairs, dtype=bool)
        self._pair_map = np.full(self.num_pairs, -1, dtype=np.int64)

    @staticmethod
    def _incident(nodes, ptr, order, k, rng) -> np.ndarray:
        if len(nodes) == 0 or k <= 0:
            return np.empty(0, dtype=np.int64)
        start, deg = ptr[nodes], ptr[nodes + 1] - ptr[nodes]
        small = deg <= k
        s_start, s_deg = start[small], deg[small]
        tot = int(s_deg.sum())
        offs = np.arange(tot) - np.repeat(np.cumsum(s_deg) - s_deg, s_deg)
        take_all = np.repeat(s_start, s_deg) + offs
        b_start, b_deg = start[~small], deg[~small]
        if len(b_start):
            r = (rng.random((len(b_start), k)) * b_deg[:, None]).astype(np.int64)
            sampled = (b_start[:, None] + r).ravel()
        else:
            sampled = np.empty(0, dtype=np.int64)
        return order[np.concatenate([take_all, sampled])]

    def _unique(self, idx: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Sorted unique values of ``idx`` using a reusable all-False bitmap."""
        if len(idx) <= len(mask) // 64:  # small: sorting is cheaper than a full scan
            return np.unique(idx)
        mask[idx] = True
        out = np.flatnonzero(mask)
        mask[out] = False
        return out

    def sample(self, seed_edges: np.ndarray, rng: np.random.Generator | None = None) -> Batch:
        g = self.g
        rng = rng or np.random.default_rng()
        seed_edges = np.asarray(seed_edges, dtype=np.int64)
        if self.fanouts is None:
            return self.make_batch(np.arange(g.num_edges), seed_edges)
        seen = self._node_mask
        frontier = np.unique(np.concatenate([g.src[seed_edges], g.dst[seed_edges]]))
        visited = [frontier]
        seen[frontier] = True
        parts = [seed_edges]
        for k in self.fanouts:
            k_out, k_in = (k + 1) // 2, max(k // 2, 1)
            e = np.concatenate(
                [
                    self._incident(frontier, self.out_ptr, self.out_order, k_out, rng),
                    self._incident(frontier, self.in_ptr, self.in_order, k_in, rng),
                ]
            )
            parts.append(e)
            nb = np.concatenate([g.src[e], g.dst[e]])
            nb = nb[~seen[nb]]
            seen[nb] = True
            frontier = np.unique(nb)
            visited.append(frontier)
        seen[np.concatenate(visited)] = False
        edges = self._unique(np.concatenate(parts), self._edge_mask)
        return self.make_batch(edges, seed_edges)

    def make_batch(self, edges: np.ndarray, seed_edges: np.ndarray) -> Batch:
        """Batch over sorted unique ``edges`` (graph-local ids) around ``seed_edges``."""
        g = self.g
        gsrc, gdst = g.src[edges], g.dst[edges]
        nodes = self._unique(np.concatenate([gsrc, gdst]), self._node_mask)
        self._node_map[nodes] = np.arange(len(nodes))
        bsrc, bdst = self._node_map[gsrc], self._node_map[gdst]
        self._node_map[nodes] = -1
        gp = self.pair_id[edges]
        pairs = self._unique(gp, self._pair_mask)
        self._pair_map[pairs] = np.arange(len(pairs))
        pair = self._pair_map[gp]
        self._pair_map[pairs] = -1
        pair_src = np.empty(len(pairs), dtype=np.int64)
        pair_dst = np.empty(len(pairs), dtype=np.int64)
        pair_src[pair] = bsrc
        pair_dst[pair] = bdst
        seed_pos = np.searchsorted(edges, seed_edges)
        seed_nodes = np.unique(np.concatenate([bsrc[seed_pos], bdst[seed_pos]]))
        ego = np.zeros(len(nodes), dtype=np.float32)
        ego[seed_nodes] = 1.0
        t = torch.from_numpy
        return Batch(
            node_num=t(g.node_num[nodes]),
            node_bank=t(g.node_bank[nodes]),
            ego=t(ego),
            src=t(bsrc),
            dst=t(bdst),
            pair=t(pair),
            num_pairs=len(pairs),
            pair_src=t(pair_src),
            pair_dst=t(pair_dst),
            edge_num=t(g.edge_num[edges]),
            edge_cat=t(g.edge_cat[edges]),
            seed_pos=t(seed_pos.astype(np.int64)),
            seed_label=t(g.label[seed_edges]),
            seed_typology=t(g.typology[seed_edges]),
            seed_nodes=t(seed_nodes.astype(np.int64)),
            seed_node_label=t(g.node_label[nodes[seed_nodes]]),
            node_global=nodes,
            edge_local=edges,
        )


def port_columns() -> tuple[int, ...]:
    from .features import EDGE_NUM_FEATURES

    return tuple(EDGE_NUM_FEATURES.index(c) for c in PORT_FEATURES)
