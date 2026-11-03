"""Training and inference for MuleNet under any bank-view regime."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .bankview import CONSORTIUM_FEATURES, ScoreProvider, consortium_features, touches_bank, visible_edges
from .data.base import TxDataset
from .data.split import TemporalSplit, temporal_split
from .graph.features import EDGE_NUM_FEATURES, NODE_NUM_FEATURES, edge_features
from .graph.multigraph import GraphData, NeighborSampler, Normalizer, build_graph, port_columns
from .metrics import best_threshold, edge_report
from .models.losses import edge_loss, typology_loss
from .models.mulenet import ModelConfig, MuleNet


@dataclass
class TrainConfig:
    regime: str = "global"  # global | siloed | consortium
    bank: int | None = None  # viewing bank for siloed / consortium
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.1
    multi_edge: bool = True
    reverse_mp: bool = True
    pna: bool = True
    edge_update: bool = True
    ego_ids: bool = True
    ports: bool = True
    typology_head: bool = True
    loss: str = "wce"  # wce | focal
    pos_weight: float = 6.0
    typ_weight: float = 0.3
    acct_weight: float = 0.3
    lr: float = 5e-4
    batch_size: int = 4096
    fanouts: list[int] | None = field(default_factory=lambda: [100, 100])
    epochs: int = 50
    patience: int = 10
    max_batches_per_epoch: int | None = None
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Prepared:
    """Data for one regime: three snapshots and the edges scored in each."""

    split: TemporalSplit
    graphs: dict[str, GraphData]
    targets: dict[str, np.ndarray]  # local edge positions in graphs[split]
    edge_dim: int
    normalizer: Normalizer


def prepare(
    ds: TxDataset,
    cfg: TrainConfig,
    split: TemporalSplit | None = None,
    edge_feats: np.ndarray | None = None,
    provider: ScoreProvider | None = None,
) -> Prepared:
    tx = ds.tx
    split = split or temporal_split(tx)
    feats = edge_feats if edge_feats is not None else edge_features(tx).to_numpy()
    if cfg.regime == "consortium":
        feats = np.concatenate([feats, consortium_features(tx, cfg.bank, provider)], axis=1)
    vis = visible_edges(tx, cfg.regime, cfg.bank)
    train_graph_ids, _ = split.snapshot("train")
    normalizer = Normalizer.fit(feats[train_graph_ids[vis[train_graph_ids]]])
    own = None if cfg.regime == "global" else cfg.bank
    graphs, targets = {}, {}
    for name in ("train", "val", "test"):
        gids, tids = split.snapshot(name)
        gids, tids = gids[vis[gids]], tids[vis[tids]]
        g = build_graph(tx, feats, gids, normalizer, own_bank=own, use_ports=cfg.ports, port_cols=port_columns())
        graphs[name] = g
        targets[name] = g.local_edges(tids)
    return Prepared(split, graphs, targets, feats.shape[1], normalizer)


def model_config(ds: TxDataset, prep: Prepared, cfg: TrainConfig) -> ModelConfig:
    g = prep.graphs["train"]
    pairs = np.unique(g.src.astype(np.int64) * g.num_nodes + g.dst)
    deg = np.bincount(pairs % g.num_nodes, minlength=g.num_nodes)
    return ModelConfig(
        node_num_dim=len(NODE_NUM_FEATURES),
        edge_num_dim=prep.edge_dim,
        num_banks=max(ds.num_banks, 1),
        num_currencies=max(len(ds.currencies), 1),
        num_formats=max(len(ds.formats), 1),
        hidden=cfg.hidden,
        layers=cfg.layers,
        dropout=cfg.dropout,
        multi_edge=cfg.multi_edge,
        reverse_mp=cfg.reverse_mp,
        pna=cfg.pna,
        edge_update=cfg.edge_update,
        ego_ids=cfg.ego_ids,
        deg_log_mean=float(np.log(deg[deg > 0] + 1).mean()) if (deg > 0).any() else 1.0,
    )


@torch.no_grad()
def predict(model: MuleNet, g: GraphData, targets: np.ndarray, cfg: TrainConfig, seed: int = 1234) -> dict[str, np.ndarray]:
    """Score target edges (and their endpoint accounts) of a snapshot."""
    model.eval()
    sampler = NeighborSampler(g, cfg.fanouts)
    rng = np.random.default_rng(seed)
    logit = np.zeros(len(targets), dtype=np.float32)
    typ = np.zeros((len(targets), model.typ_head[-1].out_features), dtype=np.float32)
    acct_sum = np.zeros(g.num_nodes)
    acct_cnt = np.zeros(g.num_nodes)
    for i in range(0, len(targets), cfg.batch_size):
        seeds = targets[i : i + cfg.batch_size]
        b = sampler.sample(seeds, rng).to(cfg.device)
        out = model(b)
        logit[i : i + len(seeds)] = out["edge_logit"].cpu().numpy()
        typ[i : i + len(seeds)] = torch.softmax(out["typ_logit"], -1).cpu().numpy()
        nodes = b.node_global[b.seed_nodes.cpu().numpy()]
        np.add.at(acct_sum, nodes, torch.sigmoid(out["acct_logit"]).cpu().numpy())
        np.add.at(acct_cnt, nodes, 1)
    acct = np.where(acct_cnt > 0, acct_sum / np.maximum(acct_cnt, 1), np.nan)
    return {"logit": logit, "prob": 1 / (1 + np.exp(-logit)), "typology": typ, "account": acct}


def train(ds: TxDataset, cfg: TrainConfig, prep: Prepared | None = None, out_dir: str | Path | None = None, log=print):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    prep = prep or prepare(ds, cfg)
    mcfg = model_config(ds, prep, cfg)
    model = MuleNet(mcfg).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    g_tr = prep.graphs["train"]
    sampler = NeighborSampler(g_tr, cfg.fanouts)
    tr_targets = prep.targets["train"]
    y_val = prep.graphs["val"].label[prep.targets["val"]]

    best = {"f1": -1.0, "epoch": -1, "threshold": 0.5}
    best_state = None
    history = []
    bad = 0
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        perm = rng.permutation(tr_targets)
        losses = []
        for bi, i in enumerate(range(0, len(perm), cfg.batch_size)):
            if cfg.max_batches_per_epoch and bi >= cfg.max_batches_per_epoch:
                break
            b = sampler.sample(perm[i : i + cfg.batch_size], rng).to(cfg.device)
            out = model(b)
            loss = edge_loss(out["edge_logit"], b.seed_label, cfg.loss, cfg.pos_weight)
            if cfg.typology_head and cfg.typ_weight > 0:
                loss = loss + cfg.typ_weight * typology_loss(out["typ_logit"], b.seed_typology)
            if cfg.acct_weight > 0:
                loss = loss + cfg.acct_weight * edge_loss(out["acct_logit"], b.seed_node_label, cfg.loss, cfg.pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        val = predict(model, prep.graphs["val"], prep.targets["val"], cfg)
        thr, f1 = best_threshold(y_val, val["prob"])
        rec = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else float("nan"), "val_f1": f1, "secs": time.time() - t0}
        history.append(rec)
        log(json.dumps(rec))
        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": epoch, "threshold": thr}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    result = {"config": asdict(cfg), "best": best, "history": history}
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "model_config": mcfg.to_dict(), "train_config": asdict(cfg), "best": best}, out / "model.pt")
        (out / "train_log.json").write_text(json.dumps(result, indent=2))
    return model, prep, result


def evaluate_edges(
    model: MuleNet,
    ds: TxDataset,
    prep: Prepared,
    cfg: TrainConfig,
    threshold: float,
    split: str = "test",
    eval_bank: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Edge metrics on a split, optionally restricted to edges touching ``eval_bank``.

    Returns (metrics, scored-edge frame) so the decision layer can reuse scores.
    """
    g, tgt = prep.graphs[split], prep.targets[split]
    pred = predict(model, g, tgt, cfg)
    edges = ds.tx.iloc[g.edge_ids[tgt]].copy()
    edges["logit"] = pred["logit"]
    edges["prob"] = pred["prob"]
    edges["typ_pred"] = pred["typology"].argmax(1)
    acct = pred["account"]
    edges["acct_src"] = acct[g.src[tgt]]
    edges["acct_dst"] = acct[g.dst[tgt]]
    mask = np.ones(len(edges), dtype=bool) if eval_bank is None else touches_bank(edges, eval_bank)
    sub = edges[mask]
    rep = edge_report(sub.label.to_numpy(), sub.prob.to_numpy(), threshold)
    pos = sub[(sub.label == 1) & (sub.typology >= 0) & (sub.typology < 8)]
    rep["typology_acc"] = float((pos.typ_pred == pos.typology).mean()) if len(pos) else float("nan")
    rep["typology_coverage"] = float(len(pos) / max(int(sub.label.sum()), 1))
    return rep, edges


__all__ = ["CONSORTIUM_FEATURES", "EDGE_NUM_FEATURES", "Prepared", "TrainConfig", "evaluate_edges", "predict", "prepare", "train"]
