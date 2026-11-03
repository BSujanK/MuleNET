# MuleNET

A money-laundering and mule-account detector built on IBM's public **AMLworld**
transaction data, using an **attention-free multigraph GNN**. What sets it apart
is how it is evaluated: the way a bank actually works.

* **One bank's limited view.** It trains and scores from a single bank's seat
  (Global, Siloed and Consortium-lite regimes), not only on the all-banks graph.
* **A fixed daily analyst capacity.** It ranks accounts, caps the queue at K
  alerts a day, and reports precision@K, recall@K and alert reduction at equal
  recall against a rule engine.
* **Explanations an investigator can act on.** Each alert comes with an evidence
  pack: the predicted typology, the most salient transfers, the rules matched,
  and a fixed-template narrative. No language model is used.

> Positioning: multigraph GNNs (Multi-GNN, MEGA-GNN), GFP + boosted trees, and
> federated AML on AMLworld already exist. The contribution here is the
> bank-realistic evaluation and decision layer on top of a strong attention-free
> model. AMLworld is synthetic, so results are relative gains under a controlled
> benchmark. Real data (for example Elliptic) is planned for later fine-tuning and
> validation, through a new dataset adapter.

## Research questions

| RQ | Question | Where |
|---|---|---|
| RQ1 | How much does one bank lose when it sees only its own transactions? | `bank-study`, `mulenet/bankview.py` |
| RQ2 | Does a shared per-account risk score (no raw data) recover part of it? | `consortium` regime |
| RQ3 | At K alerts a day, how many mule accounts are caught vs a rule engine? | `mulenet/decision.py`, `mulenet/rules.py` |
| RQ4 | Can the model name the typology and point at the right evidence? | `mulenet/explain.py` |

## Install

```bash
pip install -e ".[dev]"            # core + pytest
pip install -e ".[baselines]"      # optional: XGBoost for the GBT baseline
pip install snapml                 # optional, Linux: IBM Graph Feature Preprocessor
```

PyTorch is the only deep-learning dependency. PyTorch Geometric is **not**
required: message passing and neighbour sampling are implemented directly.

## Data

```bash
pip install kaggle   # needs ~/.kaggle/kaggle.json
scripts/download_amlworld.sh data/amlworld HI-Small LI-Small
```

The adapter reads the raw Kaggle files (`HI-Small_Trans.csv`,
`HI-Small_Patterns.txt`) directly, so IBM's `format_kaggle_files.py` is not
needed. Pattern files are matched back to transactions to give each laundering
edge its typology (8 patterns, or `UNSPECIFIED`) and attempt id.

## Quick start

```bash
# dataset stats and the banks with the most laundering (MuleNET bank index -> raw id)
python -m mulenet summary --root data/amlworld --variant HI-Small

# Global regime, 5 seeds, the published protocol (60/20/20 temporal split, 100+100 sampling)
python -m mulenet run --root data/amlworld --variant HI-Small --regime global --seeds 5

# RQ1/RQ2: Global vs Siloed vs Consortium-lite for the top 5 laundering banks
python -m mulenet bank-study --root data/amlworld --variant HI-Small --banks 5 --seeds 5

# GBT baseline (GFP features if snapml is installed, else portable pattern features)
python -m mulenet gbt --root data/amlworld --variant HI-Small --regime siloed --bank 3

# Ablations: --no-multi-edge --no-reverse-mp --no-ports --no-ego-ids --no-typology-head --no-pna --loss focal
```

Each run writes `results.json`, `account_scores_test.csv`, `evidence_packs.md`,
`model.pt` and a training log under `--out`.

Smoke-test without downloading anything (a tiny AMLworld-format fixture; never
report numbers from it):

```bash
python -m mulenet synth --out /tmp/syn
python -m mulenet run --root /tmp/syn --variant SYN-Small --seeds 1 --epochs 3 --fanouts 20,20 --device cpu --out /tmp/runs
```

## Architecture

```
AMLworld CSV ─► dataset adapter ─► canonical tx table ─► features (ports, time gaps, amounts)
                                          │
                                          ├─► bank-view slicer: Global · Siloed · Consortium-lite
                                          │
                                          ▼
           attention-free multigraph encoder (mulenet/models)
             1. multi-edge stage: sum/mean/max/std over parallel transfers per account pair (MEGA-style)
             2. 2-3 PNA layers (mean/min/max/std + degree scalers), separate in/out (reverse MP),
                ego IDs, port numbers, edge-update MLPs, residual + batch norm
                                          │
                     ┌────────────────────┼─────────────────────┐
               edge head             typology head          account head
           laundering logit       8 patterns + unspecified  node state + pooled edge scores
                                          │
            decision layer: calibration ─► 24 h / 7 d account pooling ─► top-K daily queue
                                          │
            evidence pack: 2-hop window subgraph, grad×input saliency, typology, rule hits, narrative
```

| Module | What it does |
|---|---|
| `mulenet/data/` | `schema.py` canonical table · `base.py` adapter registry · `amlworld.py` Kaggle reader · `split.py` temporal split · `synthetic.py` test fixture |
| `mulenet/graph/` | `features.py` edge/node features · `multigraph.py` snapshots, pair grouping, 100+100 neighbour sampler |
| `mulenet/models/` | `layers.py` multi-edge + PNA directed layers · `mulenet.py` encoder and three heads · `losses.py` weighted CE / focal |
| `mulenet/bankview.py` | regimes, top-bank selection, consortium-lite signal (lagged one day) |
| `mulenet/rules.py` | 5-rule engine: structuring, fan-in, fan-out, rapid pass-through, short cycles |
| `mulenet/decision.py` | temperature / isotonic calibration, account pooling, daily queue, capacity metrics, alert reduction |
| `mulenet/explain.py` | evidence packs and the fixed narrative template |
| `mulenet/baselines/gbt.py` | GFP / pattern features + XGBoost or HistGradientBoosting |
| `mulenet/pipeline.py` | train → assess (edge metrics, calibration, queue, rules, explanations) |

## Evaluation details

* **Edge level:** minority-class F1 (threshold picked on validation F1, never
  accuracy) and PR-AUC, mean ± std over 5 seeds, test-period edges only.
* **Account level:** for each test day, accounts held by the evaluating bank
  that were active that day are scored (learned pooling of calibrated 24 h and
  7 d edge scores plus the account head, fit on validation). The analyst queue
  takes the top K per day and does not re-queue an account already alerted.
  An account-day is positive if the account touched a laundering transfer that day.
* **Rule engine:** runs on exactly what the evaluating bank sees; its queue can
  only contain accounts it flagged, ranked by number of rules fired.
* **Alert reduction at equal recall:** how many model alerts are needed to reach
  the rule engine's recall, relative to the rule engine's alert count.
  `decision.analyst_hours_saved` turns that into analyst-hours.
* **Consortium-lite:** for each edge, the risk of each external endpoint as
  published by its own bank at the end of the previous day. The default signal is
  the holding bank's rule score over its own accounts' transactions (no labels,
  no raw data shared). A trained per-bank model can be plugged in as a
  `ScoreProvider`.
* **RQ4:** typology accuracy on positives that carry a pattern label (coverage is
  reported, since many AMLworld positives are unattributed) and evidence hit
  rate (share of top evidence transfers that are truly laundering).

## Adding a dataset

Write a function that returns a `TxDataset` with the canonical columns in
`mulenet/data/schema.py` and register it:

```python
from mulenet.data import register, TxDataset

@register("elliptic")
def load_elliptic(root, **kw) -> TxDataset: ...
```

Then `python -m mulenet run --dataset elliptic --root ...`. Nothing in the model,
bank view or decision layer needs to change.

## Status against the 8-week plan

| Week | Plan | Status |
|---|---|---|
| 1 | Data, formatting, reproduce GIN / Multi-GIN + EU | Data adapter, split and features done; ablation flags give GIN-style (`--no-pna --no-multi-edge`) and Multi-GIN-style variants. Reproduction on HI-Small still to run on a GPU |
| 2 | GFP + XGBoost, rule engine, EDA and bank selection | Done in code (`gbt`, `rules.py`, `summary`) |
| 3 | Multi-edge stage + Multi-PNA encoder | Done |
| 4 | Typology/account heads, calibration, daily queue | Done |
| 5 | Siloed and Consortium-lite runs | Done in code (`bank-study`) |
| 6 | Ablations, 5 seeds, optional FedAvg | Ablation flags and seeds done; FedAvg not yet |
| 7 | Evidence pack, Streamlit demo, latency | Evidence pack done; demo and latency benchmark not yet |
| 8 | Write-up | Not started |

All code is tested on a small synthetic fixture (`pytest`); no AMLworld numbers
have been produced yet.

## References

Altman et al., NeurIPS 2023 (AMLworld) · Egressy et al., AAAI 2024 (Multi-GNN) ·
Blanuša et al., ICAIF 2024 (Graph Feature Preprocessor) · Bilgi, Chen, Atasu
(MEGA-GNN), arXiv:2412.00241.
