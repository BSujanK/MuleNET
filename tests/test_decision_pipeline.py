import numpy as np
import pandas as pd

from mulenet.decision import TemperatureScaler, alert_reduction_at_equal_recall, capacity_metrics, daily_queue
from mulenet.pipeline import run_experiment
from mulenet.train import TrainConfig


def _table():
    rows = []
    for day in range(3):
        for a in range(10):
            rows.append({"account": a, "day": day, "label": int(a < 2), "score": 1.0 - a / 10})
    return pd.DataFrame(rows)


def test_daily_queue_cap_and_dedupe():
    t = _table()
    q = daily_queue(t, 3)
    assert q.groupby("day").size().max() <= 3
    assert q.account.is_unique  # an account is queued once
    q2 = daily_queue(t, 3, dedupe=False)
    assert len(q2) == 9


def test_capacity_metrics():
    m = capacity_metrics(_table(), 2, dedupe=False)
    assert m["precision_at_k"] == 1.0 and m["recall_at_k"] == 1.0 and m["account_recall"] == 1.0


def test_alert_reduction():
    t = _table()
    rule_hits = pd.Series(t.account < 6)  # rules: 6 alerts/day to catch both mules
    r = alert_reduction_at_equal_recall(t, rule_hits)
    assert r["rule_recall"] == 1.0
    assert r["model_alerts_at_rule_recall"] == 6  # 2 mules x 3 days at the top
    assert abs(r["alert_reduction"] - (1 - 6 / 18)) < 1e-9


def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.3
    logit = (y * 2 - 1) * 1.0 * 5 + rng.normal(0, 5, 2000)  # overconfident logits
    t = TemperatureScaler().fit(logit, y.astype(float))
    assert t.temperature > 1.0


def test_end_to_end_siloed(syn_ds):
    from mulenet.bankview import top_laundering_banks

    bank = top_laundering_banks(syn_ds.tx, 1)[0]
    cfg = TrainConfig(regime="consortium", bank=bank, epochs=1, batch_size=512, fanouts=[5, 5], hidden=16, device="cpu")
    r = run_experiment(syn_ds, cfg, ks=(5,), explain_top=2, log=lambda s: None)
    assert 0.0 <= r["test_edge"]["f1"] <= 1.0
    assert r["capacity"][5]["model"]["alerts"] > 0
    packs = r["_packs"]
    assert packs and packs[0].narrative.startswith("Account ")
    assert len(packs[0].evidence) > 0
