"""Graph-features + gradient-boosted-trees baseline (GFP + XGBoost family).

If IBM's ``snapml`` is installed, its GraphFeaturePreprocessor (cycle, fan and
scatter-gather pattern counts) is used. Otherwise a portable fallback builds
similar per-edge pattern features from per-day fan counts and the rule engine.
The classifier is XGBoost when installed, else scikit-learn's
HistGradientBoosting.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..metrics import best_threshold, edge_report
from ..rules import RULES, day_index, run_rules


def pattern_features(tx: pd.DataFrame, rule_flags: pd.DataFrame | None = None) -> np.ndarray:
    """Per-edge endpoint pattern features computed on the visible table ``tx``."""
    day = day_index(tx.ts)
    df = pd.DataFrame({"src": tx.src.to_numpy(), "dst": tx.dst.to_numpy(), "day": day})
    fin = df.groupby(["dst", "day"]).src.nunique()
    fout = df.groupby(["src", "day"]).dst.nunique()
    nin = df.groupby(["dst", "day"]).size()
    nout = df.groupby(["src", "day"]).size()
    flags = rule_flags if rule_flags is not None else run_rules(tx)
    cols = []
    for side in ("src", "dst"):
        key = pd.MultiIndex.from_arrays([df[side], df.day])
        for s in (fin, fout, nin, nout):
            cols.append(np.log1p(s.reindex(key).fillna(0).to_numpy()))
        f = flags.reindex(key).fillna(False)
        for r in RULES:
            cols.append(f[r].to_numpy().astype(float))
    return np.stack(cols, axis=1)


def _classifier(seed: int, pos_weight: float):
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight, random_state=seed, n_jobs=-1, tree_method="hist",
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=seed)


def gfp_features(tx: pd.DataFrame) -> np.ndarray | None:
    try:
        from snapml import GraphFeaturePreprocessor  # type: ignore
    except ImportError:
        return None
    try:  # parameter names follow the snapml docs; fall back quietly if they change
        gp = GraphFeaturePreprocessor()
        gp.set_params({"num_threads": 4, "time_window": 86400, "vertex_stats": True, "fan": True, "degree": True,
                       "scatter-gather": True, "temp-cycle": True, "lc-cycle": True, "lc-cycle_len": 4})
        raw = np.stack([tx.edge_id, tx.src, tx.dst, tx.ts], axis=1).astype(np.float64)
        out = gp.fit_transform(raw)
        return out[:, raw.shape[1]:]
    except Exception as exc:  # pragma: no cover - depends on snapml version
        warnings.warn(f"snapml GFP failed ({exc}); using portable pattern features")
        return None


def fit_eval_gbt(tx, edge_feats, split, visible, seed: int = 0) -> dict:
    """Train on the train period, pick the threshold on val, report on test."""
    vis_tx = tx[visible]
    pat = gfp_features(vis_tx)
    if pat is None:
        pat = pattern_features(vis_tx)
    X_all = np.zeros((len(tx), edge_feats.shape[1] + pat.shape[1]))
    X_all[visible] = np.concatenate([edge_feats[visible], pat], axis=1)
    y = tx.label.to_numpy()
    sets = {}
    for name in ("train", "val", "test"):
        _, tids = split.snapshot(name)
        sets[name] = tids[visible[tids]]
    tr = sets["train"]
    pw = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
    clf = _classifier(seed, pw)
    clf.fit(X_all[tr], y[tr])
    p_val = clf.predict_proba(X_all[sets["val"]])[:, 1]
    thr, _ = best_threshold(y[sets["val"]], p_val)
    p_test = clf.predict_proba(X_all[sets["test"]])[:, 1]
    return {"report": edge_report(y[sets["test"]], p_test, thr), "test_ids": sets["test"], "test_prob": p_test,
            "val_ids": sets["val"], "val_prob": p_val}
