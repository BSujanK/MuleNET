"""Evidence packs: why an account was put in the queue (design doc step 9, RQ4).

For each alert: take the 2-hop subgraph around the account in a trailing window,
rank its edges by gradient-times-input saliency, attach the predicted typology and
any rule hits, and fill a fixed report template. No language model is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from .data.base import TxDataset
from .data.schema import TYPOLOGIES
from .graph.multigraph import GraphData, NeighborSampler
from .models.mulenet import MuleNet
from .rules import RULES

_RULE_TEXT = {
    "structuring": "repeated payments just under the reporting threshold",
    "fan_in": "fan-in from many distinct senders in one day",
    "fan_out": "fan-out to many distinct receivers in one day",
    "pass_through": "money arriving and leaving within hours at a similar amount",
    "short_cycle": "funds moving in a short cycle back to their origin",
}


@dataclass
class EvidencePack:
    account: str
    bank: str
    day: str
    risk: float
    rank: int | None
    typology: str
    typology_conf: float
    rule_hits: list[str]
    evidence: pd.DataFrame
    summary: dict = field(default_factory=dict)
    narrative: str = ""

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "evidence"}
        d["evidence"] = self.evidence.to_dict(orient="records")
        return d

    def to_markdown(self) -> str:
        cols = ["time", "from", "to", "amount", "currency", "format", "edge_risk", "saliency"]
        lines = [f"### Case: account {self.account} (bank {self.bank}) on {self.day}", "", self.narrative, ""]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for r in self.evidence[cols].itertuples(index=False):
            lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
        return "\n".join(lines)


def _window_subgraph(g: GraphData, node: int, t_end: int, window_days: int, cap: int) -> tuple[np.ndarray, np.ndarray]:
    lo = np.searchsorted(g.ts, t_end - window_days * 86400, side="left")
    hi = np.searchsorted(g.ts, t_end, side="left")
    win = np.arange(lo, hi)
    src, dst = g.src[win], g.dst[win]
    hop1 = win[(src == node) | (dst == node)]
    nb = np.unique(np.concatenate([g.src[hop1], g.dst[hop1], [node]]))
    in_nb = np.isin(src, nb) | np.isin(dst, nb)
    hop2 = win[in_nb]
    if len(hop2) > cap:  # keep the most recent edges
        hop2 = np.union1d(hop2[-cap:], hop1)
    return np.union1d(hop1, hop2), hop1


def explain_account(
    model: MuleNet,
    ds: TxDataset,
    g: GraphData,
    account: int,
    day: int,
    rule_flags: pd.DataFrame | None = None,
    rank: int | None = None,
    risk: float | None = None,
    top_k: int = 10,
    window_days: int = 7,
    cap: int = 5000,
    device: str = "cpu",
) -> EvidencePack:
    """Build the evidence pack for global ``account`` id on absolute ``day``."""
    pos = np.nonzero(g.node_ids == account)[0]
    if len(pos) == 0:
        raise KeyError(f"account {account} not in this graph")
    node = int(pos[0])
    t_end = (day + 1) * 86400
    edges, own = _window_subgraph(g, node, t_end, window_days, cap)
    if len(own) == 0:
        raise ValueError("account has no transfers in the window")
    sampler = NeighborSampler(g, None)
    b = sampler.make_batch(edges, own).to(device)
    model.eval()
    x = b.edge_num.clone().requires_grad_(True)
    out = model(b, edge_num=x)
    local_node = int(np.searchsorted(b.node_global, node))
    target = out["acct_logit_all"][local_node] + out["edge_logit"].max()
    target.backward()
    sal = (x.grad * x).abs().sum(1).detach().cpu().numpy()
    probs = torch.sigmoid(out["edge_logit_all"]).detach().cpu().numpy()
    typ = torch.softmax(out["typ_logit"], -1).detach().cpu().numpy()
    w = torch.sigmoid(out["edge_logit"]).detach().cpu().numpy()
    typ_mean = (typ * w[:, None]).sum(0) / max(w.sum(), 1e-9)
    t_idx = int(typ_mean.argmax())

    order = np.argsort(-sal)[:top_k]
    gl = g.edge_ids[b.edge_local[order]]
    tx = ds.tx.iloc[gl]
    evidence = pd.DataFrame(
        {
            "edge_id": gl,
            "time": pd.to_datetime(tx.ts.to_numpy(), unit="s").strftime("%Y-%m-%d %H:%M"),
            "from": [ds.accounts[i] for i in tx.src],
            "to": [ds.accounts[i] for i in tx.dst],
            "amount": tx.amount_sent.to_numpy(),
            "currency": [ds.currencies[i] for i in tx.cur_sent],
            "format": [ds.formats[i] for i in tx.fmt],
            "edge_risk": probs[order],
            "saliency": sal[order] / max(sal.max(), 1e-12),
            "label": tx.label.to_numpy(),
            "attempt_id": tx.attempt_id.to_numpy(),
        }
    )

    hits: list[str] = []
    if rule_flags is not None and (account, day) in rule_flags.index:
        row = rule_flags.loc[(account, day)]
        hits = [r for r in RULES if bool(row[r])]

    own_tx = ds.tx.iloc[g.edge_ids[own]]
    inc = own_tx[own_tx.dst == account]
    outg = own_tx[own_tx.src == account]
    summary = {
        "incoming": len(inc),
        "incoming_senders": int(inc.src.nunique()),
        "incoming_amount": float(inc.amount_recv.sum()),
        "outgoing": len(outg),
        "outgoing_receivers": int(outg.dst.nunique()),
        "outgoing_amount": float(outg.amount_sent.sum()),
        "span_hours": float((own_tx.ts.max() - own_tx.ts.min()) / 3600.0),
        "first_seen_days": float((t_end - g.ts[(g.src == node) | (g.dst == node)].min()) / 86400.0),
    }
    risk_val = float(risk) if risk is not None else float(torch.sigmoid(out["acct_logit_all"][local_node]))
    pack = EvidencePack(
        account=ds.accounts[account],
        bank=ds.banks[int(g.node_bank[node])] if ds.banks else str(g.node_bank[node]),
        day=pd.to_datetime(day * 86400, unit="s").strftime("%Y-%m-%d"),
        risk=risk_val,
        rank=rank,
        typology=TYPOLOGIES[t_idx],
        typology_conf=float(typ_mean[t_idx]),
        rule_hits=hits,
        evidence=evidence,
        summary=summary,
    )
    pack.narrative = narrative(pack)
    return pack


def narrative(p: EvidencePack) -> str:
    """Fixed template: deterministic, auditable, no language model."""
    s = p.summary
    rank = f"ranked #{p.rank} in the queue" if p.rank is not None else "flagged"
    parts = [
        f"Account {p.account} was {rank} on {p.day} with mule-risk {p.risk:.2f}.",
        f"Most likely typology: {p.typology.lower()} (confidence {p.typology_conf:.0%}).",
        f"In the 7 days to {p.day} it received {s['incoming']} transfers from {s['incoming_senders']} distinct "
        f"accounts totalling {s['incoming_amount']:,.2f}, and sent {s['outgoing']} transfers to "
        f"{s['outgoing_receivers']} distinct accounts totalling {s['outgoing_amount']:,.2f}, "
        f"over {s['span_hours']:.1f} hours of activity.",
    ]
    if s["first_seen_days"] < 3:
        parts.append(f"The account was first seen only {s['first_seen_days']:.1f} days earlier.")
    if p.rule_hits:
        parts.append("Rules matched: " + "; ".join(_RULE_TEXT[r] for r in p.rule_hits) + ".")
    parts.append(f"The {len(p.evidence)} transfers below carried most of the model's evidence.")
    parts.append("This is a triage aid for an analyst, not a decision to restrict the account.")
    return " ".join(parts)


def evidence_hit_rate(packs: list[EvidencePack]) -> float:
    """Share of top evidence edges that are truly laundering (RQ4)."""
    labels = [int(x) for p in packs for x in p.evidence.label]
    return float(np.mean(labels)) if labels else float("nan")
