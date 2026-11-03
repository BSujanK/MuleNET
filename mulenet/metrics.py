"""Edge-level metrics used by every published AMLworld result."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def f1_at(y: np.ndarray, p: np.ndarray, thr: float) -> float:
    pred = p >= thr
    tp = float(np.sum(pred & (y == 1)))
    fp = float(np.sum(pred & (y == 0)))
    fn = float(np.sum(~pred & (y == 1)))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def best_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Threshold that maximises minority-class F1 (pick it on validation only)."""
    if y.sum() == 0:
        return 0.5, 0.0
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)[:-1]
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i])


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")


def edge_report(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    return {"f1": f1_at(y, p, thr), "pr_auc": pr_auc(y, p), "threshold": thr, "positives": int(y.sum()), "edges": len(y)}
