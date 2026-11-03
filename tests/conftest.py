import numpy as np
import pandas as pd
import pytest

from mulenet.data import load_dataset
from mulenet.data.base import TxDataset
from mulenet.data.synthetic import generate


@pytest.fixture(scope="session")
def syn_ds(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    generate(root, variant="T", n_accounts=600, n_banks=6, n_days=8, tx_per_day=600, n_attempts=30, seed=1)
    return load_dataset("amlworld", root=root, variant="T")


def make_tx(rows, n_banks=2):
    """rows: (ts, src, dst, src_bank, dst_bank, amount, label)."""
    df = pd.DataFrame(rows, columns=["ts", "src", "dst", "src_bank", "dst_bank", "amount_sent", "label"])
    df["edge_id"] = np.arange(len(df))
    df["amount_recv"] = df.amount_sent
    df["cur_sent"] = 0
    df["cur_recv"] = 0
    df["fmt"] = 0
    df["typology"] = np.where(df.label == 1, 8, -1)
    df["attempt_id"] = -1
    n = int(max(df.src.max(), df.dst.max())) + 1
    return TxDataset("toy", df, banks=[str(i) for i in range(n_banks)], accounts=[f"a{i}" for i in range(n)],
                     currencies=["USD"], formats=["ACH"])
