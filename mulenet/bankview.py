"""Bank-view slicer (design doc Section 4.3): Global, Siloed and Consortium-lite.

* ``global``      every transaction across all banks (the published protocol)
* ``siloed``      only transactions with at least one end at bank B
* ``consortium``  siloed view plus one shared number per external account: a risk
                  score from the bank that holds it, refreshed daily (lagged by one
                  day so it is never from the future). No raw transactions shared.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .rules import LOCAL_RULES, RuleConfig, day_index, rule_score, run_rules

REGIMES = ("global", "siloed", "consortium")
CONSORTIUM_FEATURES = ["ext_risk_src", "ext_risk_dst"]


def touches_bank(tx: pd.DataFrame, bank: int) -> np.ndarray:
    return ((tx.src_bank == bank) | (tx.dst_bank == bank)).to_numpy()


def visible_edges(tx: pd.DataFrame, regime: str, bank: int | None) -> np.ndarray:
    """Boolean mask of the transactions a viewer can see."""
    if regime == "global":
        return np.ones(len(tx), dtype=bool)
    if bank is None:
        raise ValueError(f"regime {regime!r} needs a bank")
    return touches_bank(tx, bank)


def top_laundering_banks(tx: pd.DataFrame, k: int = 10, edge_ids: np.ndarray | None = None) -> list[int]:
    """Banks that touch the most laundering transactions (for the per-bank study)."""
    t = tx if edge_ids is None else tx.iloc[edge_ids]
    pos = t[t.label == 1]
    counts = pd.concat([pos.src_bank, pos.dst_bank[pos.dst_bank != pos.src_bank]]).value_counts()
    return [int(b) for b in counts.index[:k]]


# A provider maps the canonical table to (risk, day0): ``risk[account, day - day0]``
# in [0, 1] is the holding bank's own score for its account at the end of ``day``.
ScoreProvider = Callable[[pd.DataFrame], tuple[np.ndarray, int]]


def rule_score_provider(cfg: RuleConfig | None = None, window_days: int = 7) -> ScoreProvider:
    """Default consortium signal: the holding bank's rule-engine score.

    Uses only the rules that need nothing but the account's own transactions,
    which the holding bank always sees in full: the share of those rules fired,
    averaged over a trailing ``window_days`` window. Needs no labels, so it cannot
    leak them. A trained per-bank model can be plugged in instead.
    """

    def provider(tx: pd.DataFrame) -> tuple[np.ndarray, int]:
        days = day_index(tx.ts)
        day0, n_days = int(days.min()), int(days.max() - days.min() + 1)
        n_acct = int(max(tx.src.max(), tx.dst.max())) + 1
        daily = np.zeros((n_acct, n_days))
        s = rule_score(run_rules(tx, cfg), LOCAL_RULES) / len(LOCAL_RULES)
        if not s.empty:
            a = s.index.get_level_values(0).to_numpy().astype(np.int64)
            d = s.index.get_level_values(1).to_numpy().astype(np.int64) - day0
            daily[a, d] = s.to_numpy()
        c = np.cumsum(np.pad(daily, ((0, 0), (1, 0))), axis=1)
        lo = np.maximum(np.arange(n_days) + 1 - window_days, 0)
        width = np.arange(n_days) + 1 - lo
        return (c[:, 1:] - c[:, lo]) / width, day0

    return provider


def consortium_features(tx: pd.DataFrame, bank: int, provider: ScoreProvider | None = None) -> np.ndarray:
    """Per-edge external-risk features for bank ``bank``'s view.

    For each edge, the risk of each endpoint that is held by *another* bank, as
    published at the end of the previous day. Own-bank endpoints get 0.
    """
    provider = provider or rule_score_provider()
    risk, day0 = provider(tx)
    prev = day_index(tx.ts) - 1 - day0
    ok = prev >= 0
    out = np.zeros((len(tx), 2), dtype=np.float64)
    for col, (acct, b) in enumerate([("src", "src_bank"), ("dst", "dst_bank")]):
        a = tx[acct].to_numpy()
        m = ok & (tx[b].to_numpy() != bank)
        out[m, col] = risk[a[m], prev[m]]
    return out
