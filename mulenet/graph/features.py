"""Edge and node feature engineering (Section 4.4 / pipeline step 4 of the design).

All edge features are causal: each one only uses the edge itself or earlier
transactions, so they can be computed once on the full table and reused by every
split and bank view.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EDGE_NUM_FEATURES = [
    "log_amount_sent",
    "log_amount_usd",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "log_dt_src",  # time since the sender's previous transfer (either direction)
    "log_dt_dst",  # time since the receiver's previous transfer
    "log_port_out",  # multigraph port number at the sender
    "log_port_in",  # multigraph port number at the receiver
    "cross_currency",
    "cross_bank",
]
PORT_FEATURES = ("log_port_out", "log_port_in")
EDGE_CAT_FEATURES = ["cur_sent", "cur_recv", "fmt"]
NODE_NUM_FEATURES = ["log_in_deg", "log_out_deg", "log_age_days"]

_NO_PREV = 30 * 86400  # gap used for an account's first transfer


def usd_rates(tx: pd.DataFrame) -> np.ndarray:
    """Estimate a USD conversion rate per currency from cross-currency payments.

    Returns ``rate[c]`` such that ``amount_in_c / rate[c]`` is in "reference"
    units. The most common currency is the reference when no currency is
    observed against it; rates are medians of observed ratios.
    """
    n_cur = int(max(tx.cur_sent.max(), tx.cur_recv.max())) + 1
    ref = int(tx.cur_sent.mode().iloc[0])
    rate = np.ones(n_cur)
    cross = tx[(tx.cur_sent != tx.cur_recv) & (tx.amount_sent > 0) & (tx.amount_recv > 0)]
    # amount_sent / amount_recv = rate[cur_sent] / rate[cur_recv]
    to_ref = cross[cross.cur_recv == ref]
    if len(to_ref):
        med = (to_ref.amount_sent / to_ref.amount_recv).groupby(to_ref.cur_sent).median()
        rate[med.index.to_numpy()] = med.to_numpy()
    from_ref = cross[cross.cur_sent == ref]
    if len(from_ref):
        med = (from_ref.amount_recv / from_ref.amount_sent).groupby(from_ref.cur_recv).median()
        missing = [c for c in med.index if rate[c] == 1.0 and c != ref]
        rate[missing] = med.loc[missing].to_numpy()
    rate[ref] = 1.0
    return rate


def amount_usd(tx: pd.DataFrame, rates: np.ndarray | None = None) -> np.ndarray:
    rates = usd_rates(tx) if rates is None else rates
    return tx.amount_sent.to_numpy() / rates[tx.cur_sent.to_numpy()]


def _ports(owner: np.ndarray, other: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Port number of ``other`` at ``owner``: rank of first contact, 1-based."""
    df = pd.DataFrame({"o": owner, "x": other, "t": order})
    first = df.groupby(["o", "x"], sort=False)["t"].transform("min")
    df["first"] = first
    pairs = df.drop_duplicates(["o", "x"])[["o", "x", "first"]].copy()
    pairs["port"] = pairs.groupby("o")["first"].rank(method="first").astype("int64")
    return df.merge(pairs[["o", "x", "port"]], on=["o", "x"], how="left")["port"].to_numpy()


def _time_since_prev(tx: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n = len(tx)
    acct = np.concatenate([tx.src.to_numpy(), tx.dst.to_numpy()])
    ts = np.concatenate([tx.ts.to_numpy(), tx.ts.to_numpy()])
    eid = np.concatenate([np.arange(n), np.arange(n)])
    order = np.lexsort((eid, ts, acct))
    a, t = acct[order], ts[order]
    gap = np.empty(2 * n, dtype=np.float64)
    gap[0] = _NO_PREV
    gap[1:] = np.where(a[1:] == a[:-1], t[1:] - t[:-1], _NO_PREV)
    out = np.empty(2 * n)
    out[order] = gap
    return out[:n], out[n:]


def edge_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Numeric edge features in EDGE_NUM_FEATURES order (unnormalised)."""
    ts = tx.ts.to_numpy()
    hour = (ts % 86400) / 3600.0
    dow = ((ts // 86400) + 3) % 7  # 1970-01-01 was a Thursday
    dt_src, dt_dst = _time_since_prev(tx)
    order = tx.edge_id.to_numpy()
    feats = pd.DataFrame(
        {
            "log_amount_sent": np.log1p(tx.amount_sent.to_numpy()),
            "log_amount_usd": np.log1p(amount_usd(tx)),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "log_dt_src": np.log1p(dt_src),
            "log_dt_dst": np.log1p(dt_dst),
            "log_port_out": np.log1p(_ports(tx.src.to_numpy(), tx.dst.to_numpy(), order)),
            "log_port_in": np.log1p(_ports(tx.dst.to_numpy(), tx.src.to_numpy(), order)),
            "cross_currency": (tx.cur_sent != tx.cur_recv).to_numpy().astype(np.float64),
            "cross_bank": (tx.src_bank != tx.dst_bank).to_numpy().astype(np.float64),
        }
    )
    return feats[EDGE_NUM_FEATURES]


def node_features(src: np.ndarray, dst: np.ndarray, ts: np.ndarray, num_nodes: int) -> np.ndarray:
    """Per-snapshot node features: log degrees and account age in the window."""
    in_deg = np.bincount(dst, minlength=num_nodes)
    out_deg = np.bincount(src, minlength=num_nodes)
    first = np.full(num_nodes, np.iinfo(np.int64).max)
    np.minimum.at(first, src, ts)
    np.minimum.at(first, dst, ts)
    end = ts.max() if len(ts) else 0
    age = np.where(first == np.iinfo(np.int64).max, 0, end - first) / 86400.0
    return np.stack([np.log1p(in_deg), np.log1p(out_deg), np.log1p(age)], axis=1).astype(np.float32)
