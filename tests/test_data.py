import numpy as np

from mulenet.data import TYPOLOGIES, temporal_split
from mulenet.data.amlworld import read_patterns


def test_adapter_reads_kaggle_layout(syn_ds):
    tx = syn_ds.tx
    assert len(tx) > 1000
    assert np.all(np.diff(tx.ts.to_numpy()) >= 0)
    assert tx.edge_id.tolist() == list(range(len(tx)))
    assert tx.label.sum() > 0
    # negatives never carry a typology; positives always do
    assert (tx.loc[tx.label == 0, "typology"] == -1).all()
    assert (tx.loc[tx.label == 1, "typology"] >= 0).all()
    # pattern file rows were matched to typologies other than UNSPECIFIED
    named = tx.loc[tx.label == 1, "typology"] < TYPOLOGIES.index("UNSPECIFIED")
    assert named.any() and not named.all()  # some positives are unattributed, like AMLworld
    assert (tx.loc[tx.attempt_id >= 0, "label"] == 1).all()


def test_accounts_are_bank_scoped(syn_ds):
    assert all(":" in a for a in syn_ds.accounts)
    assert len(set(syn_ds.accounts)) == syn_ds.num_accounts


def test_temporal_split(syn_ds):
    s = temporal_split(syn_ds.tx)
    n = len(syn_ds.tx)
    assert 0.5 * n < s.train_end < s.val_end < n
    ts = syn_ds.tx.ts.to_numpy()
    assert ts[s.train_end - 1] < ts[s.train_end]  # no timestamp straddles the cut
    g, t = s.snapshot("val")
    assert g[-1] == s.val_end - 1 and t[0] == s.train_end
    g, t = s.snapshot("test")
    assert len(g) == n and t[0] == s.val_end


def test_read_patterns(tmp_path):
    p = tmp_path / "x_Patterns.txt"
    p.write_text(
        "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 2-degree Fan-Out\n"
        "2022/09/01 00:06,021,A1,012,B1,10.00,Euro,10.00,Euro,ACH,1\n"
        "2022/09/01 00:07,021,A1,012,B2,11.00,Euro,11.00,Euro,ACH,1\n"
        "END LAUNDERING ATTEMPT - FAN-OUT\n\n"
        "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 3 hops\n"
        "2022/09/02 00:06,1,C,2,D,5.00,Euro,5.00,Euro,ACH,1\n"
        "END LAUNDERING ATTEMPT - CYCLE\n"
    )
    pat = read_patterns(p)
    assert pat.Typology.tolist() == ["FAN-OUT", "FAN-OUT", "CYCLE"]
    assert pat.Attempt.tolist() == [0, 0, 1]
    assert pat["From Bank"].tolist() == [21, 21, 1]
