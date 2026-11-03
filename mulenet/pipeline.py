"""End-to-end experiment: train, score, queue, compare with rules, explain.

One call answers RQ1-RQ4 for one (regime, bank, seed) run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .bankview import visible_edges
from .data.base import TxDataset
from .decision import (
    AccountPooler,
    account_day_table,
    alert_reduction_at_equal_recall,
    calibrator,
    capacity_metrics,
    expected_calibration_error,
)
from .explain import evidence_hit_rate, explain_account
from .rules import RULES, run_rules
from .train import Prepared, TrainConfig, evaluate_edges, prepare, train


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def rule_table(tx: pd.DataFrame, visible: np.ndarray, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rule flags aligned to an (account, day) table, computed on the visible edges."""
    flags = run_rules(tx[visible])
    key = pd.MultiIndex.from_arrays([table.account, table.day])
    aligned = flags.reindex(key).fillna(False).astype(bool)
    return flags, aligned


def run_experiment(
    ds: TxDataset,
    cfg: TrainConfig,
    ks: tuple[int, ...] = (50, 100, 200),
    explain_top: int = 20,
    calibration: str = "temperature",
    eval_bank: int | None = None,
    prep: Prepared | None = None,
    out_dir: str | Path | None = None,
    log=print,
) -> dict:
    model, prep, train_res = train(ds, cfg, prep=prep, out_dir=out_dir, log=log)
    bank = cfg.bank if cfg.regime != "global" else eval_bank
    result = assess(model, ds, prep, cfg, train_res["best"]["threshold"], bank, ks, explain_top, calibration, out_dir)
    result["train"] = train_res["best"]
    return result


def assess(
    model,
    ds: TxDataset,
    prep: Prepared,
    cfg: TrainConfig,
    thr: float,
    bank: int | None = None,
    ks: tuple[int, ...] = (50, 100, 200),
    explain_top: int = 20,
    calibration: str = "temperature",
    out_dir: str | Path | None = None,
) -> dict:
    """Score a trained model from bank ``bank``'s seat (None = all accounts)."""

    # ---- edge level (comparable with published AMLworld numbers)
    val_rep, val_edges = evaluate_edges(model, ds, prep, cfg, thr, "val", bank)
    test_rep, test_edges = evaluate_edges(model, ds, prep, cfg, thr, "test", bank)

    # ---- calibration (fit on validation)
    cal = calibrator(calibration).fit(val_edges.logit.to_numpy(), val_edges.label.to_numpy())
    for e in (val_edges, test_edges):
        e["prob_cal"] = cal(e.logit.to_numpy())
    test_rep["ece_raw"] = expected_calibration_error(test_edges.prob.to_numpy(), test_edges.label.to_numpy())
    test_rep["ece_calibrated"] = expected_calibration_error(test_edges.prob_cal.to_numpy(), test_edges.label.to_numpy())

    # ---- account level, capped daily queue
    val_tab = account_day_table(val_edges, bank, "prob_cal")
    test_tab = account_day_table(test_edges, bank, "prob_cal")
    pooler = AccountPooler("learned").fit(val_tab)
    test_tab["score"] = pooler.score(test_tab)

    # The rule engine always runs on what the evaluating bank itself can see.
    vis = visible_edges(ds.tx, "global" if bank is None else "siloed", bank)
    t0 = int(test_edges.ts.min()) - 7 * 86400  # history for windowed rules
    vis_hist = vis & (ds.tx.ts.to_numpy() >= t0)
    flags, aligned = rule_table(ds.tx, vis_hist, test_tab)
    test_tab["rule_score"] = aligned[list(RULES)].sum(axis=1).to_numpy().astype(float)
    test_tab["rule_hit"] = test_tab.rule_score > 0
    # rule engine queue: only flagged accounts, most rules fired first, random tie-break
    rng = np.random.default_rng(cfg.seed)
    test_tab["rule_rank_score"] = np.where(test_tab.rule_hit, test_tab.rule_score + rng.random(len(test_tab)) * 0.01, -1.0)

    capacity = {}
    for k in ks:
        capacity[k] = {
            "model": capacity_metrics(test_tab, k, "score"),
            "rules": capacity_metrics(test_tab, k, "rule_rank_score", min_score=0.0),
        }
    reduction = alert_reduction_at_equal_recall(test_tab, test_tab.rule_hit, "score")

    # ---- evidence packs for the top alerts
    packs = []
    if explain_top:
        top = test_tab.sort_values("score", ascending=False).head(explain_top)
        g = prep.graphs["test"]
        for rank, r in enumerate(top.itertuples(), 1):
            try:
                packs.append(
                    explain_account(model, ds, g, int(r.account), int(r.day), flags, rank=rank, risk=float(r.score), device=cfg.device)
                )
            except (KeyError, ValueError):
                continue
    result = {
        "regime": cfg.regime,
        "bank": bank,
        "seed": cfg.seed,
        "val_edge": val_rep,
        "test_edge": test_rep,
        "capacity": capacity,
        "alert_reduction": reduction,
        "evidence_hit_rate": evidence_hit_rate(packs),
        "typology_acc": test_rep.get("typology_acc"),
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(result, indent=2, default=_json_default))
        test_tab.to_csv(out / "account_scores_test.csv", index=False)
        with open(out / "evidence_packs.md", "w") as f:
            for p in packs:
                f.write(p.to_markdown() + "\n\n")
    result["_packs"] = packs
    return result
