"""Decision layer: calibration, account ranking and a capped daily alert queue.

Analysts work accounts, not transactions, and can review only K accounts a day.
This module turns edge scores into a daily account queue and measures it the way
a compliance team would (design doc Sections 5.8 and 6.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .metrics import pr_auc
from .rules import day_index

# ---------------------------------------------------------------- calibration


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def fit(self, logit: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        z = torch.tensor(logit, dtype=torch.float64)
        t = torch.tensor(y, dtype=torch.float64)
        log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z / log_t.exp(), t)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = float(log_t.detach().exp().item())
        return self

    def __call__(self, logit: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.asarray(logit) / self.temperature))


class IsotonicCalibrator:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, logit: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.iso.fit(logit, y)
        return self

    def __call__(self, logit: np.ndarray) -> np.ndarray:
        return self.iso.predict(logit)


def calibrator(kind: str = "temperature"):
    return {"temperature": TemperatureScaler, "isotonic": IsotonicCalibrator}[kind]()


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


# ------------------------------------------------------------ account scores

ACCOUNT_FEATURES = ["max_24h", "max_7d", "noisy_or_24h", "log_n_24h", "acct_head"]


def account_day_table(edges: pd.DataFrame, own_bank: int | None = None, prob_col: str = "prob") -> pd.DataFrame:
    """One row per (account, day) an account is active, with pooled edge scores.

    ``edges`` needs ts, src, dst, src_bank, dst_bank, label and ``prob_col``;
    optional acct_src / acct_dst carry the model's account-head scores. Only
    accounts held by ``own_bank`` are candidates (a bank can act only on its own
    accounts); ``None`` means every account.
    """
    day = day_index(edges.ts)
    parts = []
    for side in ("src", "dst"):
        f = pd.DataFrame(
            {
                "account": edges[side].to_numpy(),
                "bank": edges[f"{side}_bank"].to_numpy(),
                "day": day,
                "p": edges[prob_col].to_numpy(),
                "label": edges.label.to_numpy(),
                "acct_head": edges[f"acct_{side}"].to_numpy() if f"acct_{side}" in edges else np.nan,
            }
        )
        parts.append(f)
    long = pd.concat(parts, ignore_index=True)
    if own_bank is not None:
        long = long[long.bank == own_bank]
    long["log1m"] = np.log1p(-np.clip(long.p, 0, 1 - 1e-7))
    g = long.groupby(["account", "day"])
    t = pd.DataFrame(
        {
            "max_24h": g.p.max(),
            "noisy_or_24h": 1 - np.exp(g.log1m.sum()),
            "log_n_24h": np.log1p(g.size()),
            "acct_head": g.acct_head.mean(),
            "label": g.label.max(),
        }
    ).reset_index()
    t["acct_head"] = t.acct_head.fillna(t.max_24h)
    # trailing 7-day max per account
    t = t.sort_values(["account", "day"])
    t["max_7d"] = _trailing_max(t, "max_24h", 7)
    return t.reset_index(drop=True)


def _trailing_max(t: pd.DataFrame, col: str, days: int) -> np.ndarray:
    out = t[col].to_numpy().copy()
    acct, day, val = t.account.to_numpy(), t.day.to_numpy(), t[col].to_numpy()
    for lag in range(1, days):
        # compare with earlier rows of the same account within `days`
        prev_ok = np.zeros(len(t), dtype=bool)
        prev_ok[lag:] = (acct[lag:] == acct[:-lag]) & (day[lag:] - day[:-lag] < days)
        shifted = np.full(len(t), -np.inf)
        shifted[lag:] = val[:-lag]
        out = np.where(prev_ok, np.maximum(out, shifted), out)
    return out


class AccountPooler:
    """Learned pooling of 24 h / 7 d edge scores + account head (fit on validation)."""

    def __init__(self, kind: str = "learned"):
        self.kind = kind
        self.lr: LogisticRegression | None = None

    def fit(self, table: pd.DataFrame) -> "AccountPooler":
        if self.kind == "learned" and table.label.nunique() == 2:
            self.lr = LogisticRegression(class_weight="balanced", max_iter=1000)
            self.lr.fit(table[ACCOUNT_FEATURES].to_numpy(), table.label.to_numpy())
        return self

    def score(self, table: pd.DataFrame) -> np.ndarray:
        if self.lr is None:
            return table["max_24h"].to_numpy()
        return self.lr.predict_proba(table[ACCOUNT_FEATURES].to_numpy())[:, 1]


# ---------------------------------------------------------------- queue


def daily_queue(
    table: pd.DataFrame, k: int, score_col: str = "score", dedupe: bool = True, min_score: float | None = None
) -> pd.DataFrame:
    """Top-``k`` accounts per day. With ``dedupe`` an account alerted on an earlier
    day is not re-queued (analysts already have the case). Rows scoring below
    ``min_score`` are never queued (e.g. accounts no rule fired on)."""
    alerted: set[int] = set()
    picks = []
    cand = table if min_score is None else table[table[score_col] >= min_score]
    for d, day_rows in cand.sort_values(score_col, ascending=False).groupby("day", sort=True):
        rows = day_rows[~day_rows.account.isin(alerted)] if dedupe else day_rows
        top = rows.head(k)
        alerted.update(top.account.tolist())
        picks.append(top)
    return pd.concat(picks) if picks else table.head(0)


def capacity_metrics(
    table: pd.DataFrame, k: int, score_col: str = "score", dedupe: bool = True, min_score: float | None = None
) -> dict:
    """Precision@K and recall@K per day, plus period-level unique-account recall.

    Precision is averaged over days with at least one alert; recall over days
    with at least one mule account.
    """
    q = daily_queue(table, k, score_col, dedupe, min_score)
    per_day = []
    for d, rows in table.groupby("day"):
        qd = q[q.day == d]
        pos = int(rows.label.sum())
        tp = int(qd.label.sum())
        per_day.append({"day": int(d), "alerts": len(qd), "tp": tp, "positives": pos})
    pdf = pd.DataFrame(per_day)
    mules = set(table.loc[table.label == 1, "account"])
    caught = set(q.loc[q.label == 1, "account"])
    return {
        "k": k,
        "precision_at_k": float((pdf.tp / pdf.alerts)[pdf.alerts > 0].mean()) if (pdf.alerts > 0).any() else float("nan"),
        "recall_at_k": float((pdf.tp / pdf.positives.clip(lower=1))[pdf.positives > 0].mean()) if (pdf.positives > 0).any() else float("nan"),
        "account_recall": len(caught) / max(len(mules), 1),
        "mule_accounts": len(mules),
        "alerts": len(q),
        "account_pr_auc": pr_auc(table.label.to_numpy(), table[score_col].to_numpy()),
    }


def alert_reduction_at_equal_recall(table: pd.DataFrame, rule_hits: pd.Series, score_col: str = "score") -> dict:
    """How many fewer (account, day) alerts the model needs to match rule recall.

    ``rule_hits`` is a boolean Series aligned with ``table`` (any rule fired).
    """
    y = table.label.to_numpy().astype(bool)
    hits = rule_hits.to_numpy().astype(bool)
    n_pos = max(int(y.sum()), 1)
    rule_alerts = int(hits.sum())
    rule_recall = float((hits & y).sum() / n_pos)
    order = np.argsort(-table[score_col].to_numpy(), kind="stable")
    cum = np.cumsum(y[order]) / n_pos
    reach = np.nonzero(cum >= rule_recall - 1e-12)[0]
    model_alerts = int(reach[0] + 1) if len(reach) and rule_recall > 0 else 0
    return {
        "rule_alerts": rule_alerts,
        "rule_recall": rule_recall,
        "rule_precision": float((hits & y).sum() / max(rule_alerts, 1)),
        "model_alerts_at_rule_recall": model_alerts,
        "alert_reduction": 1 - model_alerts / rule_alerts if rule_alerts else float("nan"),
    }


def analyst_hours_saved(alerts_saved_per_day: float, minutes_per_alert: float = 35.0) -> float:
    """Business-impact helper (design doc Section 7): hours freed per day."""
    return alerts_saved_per_day * minutes_per_alert / 60.0
