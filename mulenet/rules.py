"""Rule-engine baseline: five classic AML rules (design doc, pipeline step 7).

This is the realistic "what a bank runs today" baseline. Rules fire per
(account, day) on whatever transactions the viewing bank can see.

1. structuring     >= ``structuring_min`` sends in a day just under the reporting threshold
2. fan_in          >= ``fan_k`` distinct senders in a day
3. fan_out         >= ``fan_k`` distinct receivers in a day
4. pass_through    money arrives and leaves within ``pass_hours`` at a similar amount
5. short_cycle     account lies on a directed cycle of <= 4 hops in a trailing window
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .graph.features import amount_usd

RULES = ("structuring", "fan_in", "fan_out", "pass_through", "short_cycle")
LOCAL_RULES = ("structuring", "fan_in", "fan_out", "pass_through")  # need only the account's own edges


@dataclass
class RuleConfig:
    threshold_usd: float = 10_000.0
    structuring_band: float = 0.9
    structuring_min: int = 3
    fan_k: int = 10
    pass_hours: float = 6.0
    pass_tol: float = 0.1
    cycle_window_days: int = 3
    cycle_max_degree: int = 200  # skip very-high-degree hubs in cycle search


def day_index(ts: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(ts) // 86400


def run_rules(tx: pd.DataFrame, cfg: RuleConfig | None = None, rates: np.ndarray | None = None) -> pd.DataFrame:
    """Return a frame indexed by (account, day) with one boolean column per rule.

    Only (account, day) pairs where at least one rule fires are returned.
    """
    cfg = cfg or RuleConfig()
    if len(tx) == 0:
        return pd.DataFrame(columns=list(RULES), index=pd.MultiIndex.from_arrays([[], []], names=["account", "day"]))
    day = day_index(tx.ts)
    usd = amount_usd(tx, rates)
    df = pd.DataFrame({"src": tx.src.to_numpy(), "dst": tx.dst.to_numpy(), "ts": tx.ts.to_numpy(), "day": day, "usd": usd})
    hits: dict[str, pd.Index] = {}

    lo = cfg.structuring_band * cfg.threshold_usd
    s = df[(df.usd >= lo) & (df.usd < cfg.threshold_usd)].groupby(["src", "day"]).size()
    hits["structuring"] = s[s >= cfg.structuring_min].index

    fi = df.groupby(["dst", "day"])["src"].nunique()
    hits["fan_in"] = fi[fi >= cfg.fan_k].index
    fo = df.groupby(["src", "day"])["dst"].nunique()
    hits["fan_out"] = fo[fo >= cfg.fan_k].index

    hits["pass_through"] = _pass_through(df, cfg)
    hits["short_cycle"] = _short_cycles(df, cfg)

    frames = []
    for name, idx in hits.items():
        if len(idx):
            f = pd.DataFrame({"account": idx.get_level_values(0), "day": idx.get_level_values(1)})
            f[name] = True
            frames.append(f.set_index(["account", "day"]))
    if not frames:
        return pd.DataFrame(columns=list(RULES), index=pd.MultiIndex.from_arrays([[], []], names=["account", "day"]))
    out = pd.concat(frames, axis=1)
    for r in RULES:
        if r not in out:
            out[r] = False
    return out[list(RULES)].fillna(False).astype(bool).sort_index()


def _pass_through(df: pd.DataFrame, cfg: RuleConfig) -> pd.MultiIndex:
    inc = df[["dst", "ts", "usd"]].rename(columns={"dst": "acct", "usd": "usd_in"}).sort_values("ts")
    out = df[["src", "ts", "usd", "day"]].rename(columns={"src": "acct", "usd": "usd_out"}).sort_values("ts")
    m = pd.merge_asof(
        out,
        inc,
        on="ts",
        by="acct",
        direction="backward",
        tolerance=int(cfg.pass_hours * 3600),
        allow_exact_matches=True,
    )
    ok = m.usd_in.notna() & ((m.usd_out - m.usd_in).abs() <= cfg.pass_tol * m.usd_in)
    return pd.MultiIndex.from_frame(m.loc[ok, ["acct", "day"]].drop_duplicates())


def _short_cycles(df: pd.DataFrame, cfg: RuleConfig) -> pd.MultiIndex:
    """Accounts on a directed cycle of length 2..4 within a trailing window of days."""
    out: list[tuple[int, int]] = []
    days = np.sort(df.day.unique())
    for d in days:
        w = df[(df.day > d - cfg.cycle_window_days) & (df.day <= d)]
        active = np.unique(np.concatenate([w.loc[w.day == d, "src"], w.loc[w.day == d, "dst"]]))
        nodes, inv = np.unique(np.concatenate([w.src.to_numpy(), w.dst.to_numpy()]), return_inverse=True)
        m = len(w)
        s, t = inv[:m], inv[m:]
        keep = s != t
        n = len(nodes)
        deg = np.bincount(s[keep], minlength=n) + np.bincount(t[keep], minlength=n)
        keep &= (deg[s] <= cfg.cycle_max_degree) & (deg[t] <= cfg.cycle_max_degree)
        a = sp.csr_matrix((np.ones(keep.sum()), (s[keep], t[keep])), shape=(n, n))
        a.data[:] = 1.0
        a2 = a @ a
        on_cycle = np.asarray(a.multiply(a.T).sum(1)).ravel() > 0  # length 2
        on_cycle |= np.asarray(a2.multiply(a.T).sum(1)).ravel() > 0  # length 3
        on_cycle |= np.asarray(a2.multiply(a2.T).sum(1)).ravel() > 0  # length 4
        flagged = nodes[on_cycle]
        flagged = flagged[np.isin(flagged, active)]
        out.extend((int(x), int(d)) for x in flagged)
    if not out:
        return pd.MultiIndex.from_arrays([[], []])
    return pd.MultiIndex.from_tuples(out)


def rule_score(flags: pd.DataFrame, rules: tuple[str, ...] = RULES) -> pd.Series:
    """Number of rules fired per (account, day)."""
    return flags[list(rules)].sum(axis=1).astype(float)
