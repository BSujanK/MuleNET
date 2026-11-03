"""Command line entry point: ``python -m mulenet <command> ...``.

Commands
--------
summary     dataset statistics and the banks with the most laundering
synth       write a tiny AMLworld-format fixture (tests / smoke runs only)
run         train + assess one regime over several seeds (mean ± std)
bank-study  RQ1/RQ2: Global vs Siloed vs Consortium-lite for the top-N banks
gbt         graph-features + gradient-boosted-trees baseline
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from .bankview import REGIMES, top_laundering_banks, visible_edges
from .data import load_dataset, temporal_split
from .graph.features import edge_features
from .train import TrainConfig, prepare, train


def _load(a):
    return load_dataset(a.dataset, root=a.root, variant=a.variant, nrows=a.nrows)


def _cfg(a, **over) -> TrainConfig:
    keys = {f.name for f in fields(TrainConfig)}
    kw = {k: v for k, v in vars(a).items() if k in keys and v is not None}
    if a.fanouts is not None:
        kw["fanouts"] = None if a.fanouts == "all" else [int(x) for x in a.fanouts.split(",")]
    kw.update(over)
    return TrainConfig(**kw)


def _flat(r: dict) -> dict:
    out = {
        "regime": r["regime"],
        "bank": r["bank"],
        "seed": r["seed"],
        "edge_f1": r["test_edge"]["f1"],
        "edge_pr_auc": r["test_edge"]["pr_auc"],
        "typology_acc": r["typology_acc"],
        "evidence_hit_rate": r["evidence_hit_rate"],
        "alert_reduction": r["alert_reduction"]["alert_reduction"],
    }
    for k, v in r["capacity"].items():
        out[f"recall@{k}"] = v["model"]["account_recall"]
        out[f"precision@{k}"] = v["model"]["precision_at_k"]
        if v["rules"]:
            out[f"rules_recall@{k}"] = v["rules"]["account_recall"]
    return out


def _agg(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    num = [c for c in df.columns if c not in by + ["seed"]]
    g = df.groupby(by, dropna=False)[num]
    mean, std = g.mean(), g.std().fillna(0)
    return mean.round(4).astype(str) + " ± " + std.round(4).astype(str)


def cmd_summary(a):
    ds = _load(a)
    print(json.dumps(ds.summary(), indent=2))
    split = temporal_split(ds.tx)
    top = top_laundering_banks(ds.tx, a.banks, np.arange(split.train_end))
    print("top laundering banks in the train period (MuleNET bank index -> raw bank id):")
    for b in top:
        print(f"  {b} -> {ds.banks[b]}")


def cmd_synth(a):
    from .data.synthetic import generate

    print(generate(a.out, variant=a.variant, seed=a.seed))


def cmd_run(a):
    from .pipeline import run_experiment

    ds = _load(a)
    feats = edge_features(ds.tx).to_numpy()
    split = temporal_split(ds.tx)
    rows = []
    for seed in range(a.seeds):
        cfg = _cfg(a, seed=seed)
        prep = prepare(ds, cfg, split, feats)
        out = Path(a.out) / f"{cfg.regime}_bank{cfg.bank}_seed{seed}"
        r = run_experiment(ds, cfg, ks=tuple(a.k), explain_top=a.explain_top, eval_bank=a.eval_bank, prep=prep, out_dir=out)
        r.pop("_packs", None)
        rows.append(_flat(r))
    df = pd.DataFrame(rows)
    df.to_csv(Path(a.out) / "runs.csv", index=False)
    print(_agg(df, ["regime", "bank"]).T.to_string())


def cmd_bank_study(a):
    from .pipeline import assess

    ds = _load(a)
    feats = edge_features(ds.tx).to_numpy()
    split = temporal_split(ds.tx)
    banks = [int(b) for b in a.bank_list.split(",")] if a.bank_list else top_laundering_banks(ds.tx, a.banks, np.arange(split.train_end))
    regimes = a.regimes.split(",")
    rows = []
    for seed in range(a.seeds):
        if "global" in regimes:  # one global model per seed, assessed from each bank's seat
            cfg = _cfg(a, seed=seed, regime="global", bank=None)
            model, prep, res = train(ds, cfg, prep=prepare(ds, cfg, split, feats))
            for b in banks:
                r = assess(model, ds, prep, cfg, res["best"]["threshold"], b, tuple(a.k), a.explain_top,
                           out_dir=Path(a.out) / f"global_bank{b}_seed{seed}")
                rows.append(_flat(r))
        for regime in [r for r in regimes if r != "global"]:
            for b in banks:
                cfg = _cfg(a, seed=seed, regime=regime, bank=b)
                model, prep, res = train(ds, cfg, prep=prepare(ds, cfg, split, feats))
                r = assess(model, ds, prep, cfg, res["best"]["threshold"], b, tuple(a.k), a.explain_top,
                           out_dir=Path(a.out) / f"{regime}_bank{b}_seed{seed}")
                rows.append(_flat(r))
            pd.DataFrame(rows).to_csv(Path(a.out) / "bank_study_runs.csv", index=False)
    df = pd.DataFrame(rows)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(a.out) / "bank_study_runs.csv", index=False)
    # visibility gap (Global - Siloed) and recovery (Consortium - Siloed), per bank
    piv = df.groupby(["bank", "regime"])["edge_f1"].mean().unstack()
    if {"global", "siloed"} <= set(piv.columns):
        piv["visibility_gap"] = piv["global"] - piv["siloed"]
    if {"consortium", "siloed"} <= set(piv.columns):
        piv["recovery"] = piv["consortium"] - piv["siloed"]
    piv.to_csv(Path(a.out) / "visibility_gap.csv")
    print(_agg(df, ["regime", "bank"]).to_string())
    print(piv.round(4).to_string())


def cmd_gbt(a):
    from .baselines.gbt import fit_eval_gbt

    ds = _load(a)
    feats = edge_features(ds.tx).to_numpy()
    split = temporal_split(ds.tx)
    vis = visible_edges(ds.tx, a.regime, a.bank)
    reps = [fit_eval_gbt(ds.tx, feats, split, vis, seed)["report"] for seed in range(a.seeds)]
    f1 = [r["f1"] for r in reps]
    print(json.dumps({"regime": a.regime, "bank": a.bank, "f1_mean": float(np.mean(f1)), "f1_std": float(np.std(f1)), "runs": reps}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mulenet")
    sub = p.add_subparsers(dest="cmd", required=True)

    def data_args(s):
        s.add_argument("--dataset", default="amlworld")
        s.add_argument("--root", required=True, help="folder with <variant>_Trans.csv")
        s.add_argument("--variant", default="HI-Small")
        s.add_argument("--nrows", type=int, default=None)

    def model_args(s):
        s.add_argument("--hidden", type=int)
        s.add_argument("--layers", type=int)
        s.add_argument("--epochs", type=int)
        s.add_argument("--batch-size", dest="batch_size", type=int)
        s.add_argument("--lr", type=float)
        s.add_argument("--loss", choices=["wce", "focal"])
        s.add_argument("--fanouts", default=None, help="e.g. 100,100 or 'all' for full graph")
        s.add_argument("--max-batches", dest="max_batches_per_epoch", type=int)
        s.add_argument("--device")
        for flag in ("multi_edge", "reverse_mp", "pna", "edge_update", "ego_ids", "ports", "typology_head"):
            s.add_argument(f"--no-{flag.replace('_', '-')}", dest=flag, action="store_const", const=False,
                           help=f"ablation: disable {flag}")
        s.add_argument("--seeds", type=int, default=5)
        s.add_argument("--k", type=int, nargs="+", default=[50, 100, 200])
        s.add_argument("--explain-top", dest="explain_top", type=int, default=20)
        s.add_argument("--out", default="runs")

    s = sub.add_parser("summary")
    data_args(s)
    s.add_argument("--banks", type=int, default=10)
    s.set_defaults(fn=cmd_summary)

    s = sub.add_parser("synth")
    s.add_argument("--out", required=True)
    s.add_argument("--variant", default="SYN-Small")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("run")
    data_args(s)
    model_args(s)
    s.add_argument("--regime", choices=REGIMES, default="global")
    s.add_argument("--bank", type=int, default=None)
    s.add_argument("--eval-bank", dest="eval_bank", type=int, default=None)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("bank-study")
    data_args(s)
    model_args(s)
    s.add_argument("--banks", type=int, default=5)
    s.add_argument("--bank-list", dest="bank_list", default=None, help="comma-separated bank ids")
    s.add_argument("--regimes", default="global,siloed,consortium")
    s.set_defaults(fn=cmd_bank_study)

    s = sub.add_parser("gbt")
    data_args(s)
    s.add_argument("--regime", choices=REGIMES, default="global")
    s.add_argument("--bank", type=int, default=None)
    s.add_argument("--seeds", type=int, default=5)
    s.set_defaults(fn=cmd_gbt)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
