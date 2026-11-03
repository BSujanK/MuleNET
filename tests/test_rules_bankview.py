import numpy as np

from mulenet.bankview import consortium_features, rule_score_provider, visible_edges
from mulenet.rules import RuleConfig, run_rules

from .conftest import make_tx

DAY = 86400


def test_fan_in_and_fan_out():
    rows = [(DAY + i, 100 + i, 0, 0, 0, 50.0, 0) for i in range(10)]  # 10 senders -> 0
    rows += [(DAY + 20 + i, 1, 200 + i, 0, 0, 50.0, 0) for i in range(9)]  # only 9 receivers
    flags = run_rules(make_tx(rows).tx)
    assert flags.loc[(0, 1), "fan_in"]
    assert (1, 1) not in flags.index or not flags.loc[(1, 1), "fan_out"]


def test_structuring_pass_through_and_cycle():
    rows = [(DAY + i * 60, 5, 6, 0, 0, 9500.0, 0) for i in range(3)]  # structuring by 5
    rows += [(DAY + 1000, 7, 8, 0, 0, 1000.0, 0), (DAY + 3000, 8, 9, 0, 0, 990.0, 0)]  # pass-through at 8
    rows += [(DAY + 4000, 20, 21, 0, 0, 1.0, 0), (DAY + 4100, 21, 22, 0, 0, 1.0, 0), (DAY + 4200, 22, 20, 0, 0, 1.0, 0)]
    rows.sort()
    flags = run_rules(make_tx(rows).tx, RuleConfig())
    assert flags.loc[(5, 1), "structuring"]
    assert flags.loc[(8, 1), "pass_through"]
    for a in (20, 21, 22):
        assert flags.loc[(a, 1), "short_cycle"]
    assert (6, 1) not in flags.index or not flags.loc[(6, 1)].any()


def test_siloed_view_only_sees_own_bank():
    ds = make_tx([(0, 0, 1, 0, 1, 1.0, 0), (1, 2, 3, 1, 1, 1.0, 0), (2, 3, 0, 1, 0, 1.0, 0)])
    assert visible_edges(ds.tx, "siloed", 0).tolist() == [True, False, True]
    assert visible_edges(ds.tx, "global", None).all()


def test_consortium_signal_is_lagged_and_external_only():
    # account 9 (bank 1) fans in on day 1; bank 0 should only see that risk from day 2
    rows = [(DAY + i, 100 + i, 9, 1, 1, 5.0, 0) for i in range(12)]
    rows += [(DAY + 500, 0, 9, 0, 1, 5.0, 0), (2 * DAY + 10, 0, 9, 0, 1, 5.0, 0), (2 * DAY + 20, 9, 0, 1, 0, 5.0, 0)]
    rows.sort()
    ds = make_tx(rows)
    feats = consortium_features(ds.tx, bank=0, provider=rule_score_provider())
    tx = ds.tx
    same_day = np.nonzero((tx.src == 0).to_numpy() & (tx.ts < 2 * DAY).to_numpy())[0]
    next_day = np.nonzero((tx.src == 0).to_numpy() & (tx.ts >= 2 * DAY).to_numpy())[0]
    assert feats[same_day, 1].max() == 0.0  # no same-day leakage
    assert feats[next_day, 1].min() > 0.0  # receiver's risk visible next day
    back = np.nonzero((tx.src == 9).to_numpy() & (tx.dst == 0).to_numpy())[0]
    assert feats[back, 0].min() > 0.0 and feats[back, 1].max() == 0.0  # own account gets 0
