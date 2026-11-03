import numpy as np
import torch

from mulenet.data import temporal_split
from mulenet.graph.features import EDGE_NUM_FEATURES, edge_features
from mulenet.graph.multigraph import NeighborSampler, Normalizer, build_graph
from mulenet.models.layers import scatter_max, scatter_mean, scatter_min, scatter_std, scatter_sum
from mulenet.models.mulenet import ModelConfig, MuleNet

from .conftest import make_tx


def test_ports_and_time_gaps():
    # account 0 pays 1, 2, 1, 3 -> out ports 1, 2, 1, 3
    ds = make_tx([(0, 0, 1, 0, 0, 10, 0), (60, 0, 2, 0, 0, 10, 0), (120, 0, 1, 0, 0, 10, 0), (180, 0, 3, 0, 1, 10, 0)])
    f = edge_features(ds.tx)
    ports = np.expm1(f["log_port_out"].to_numpy()).round().astype(int)
    assert ports.tolist() == [1, 2, 1, 3]
    gaps = np.expm1(f["log_dt_src"].to_numpy()).round().astype(int)
    assert gaps[1:].tolist() == [60, 60, 60]
    assert f["cross_bank"].tolist() == [0, 0, 0, 1]


def test_scatter_ops():
    x = torch.tensor([[1.0], [3.0], [5.0]])
    idx = torch.tensor([0, 0, 2])
    assert scatter_sum(x, idx, 3).squeeze().tolist() == [4.0, 0.0, 5.0]
    assert scatter_mean(x, idx, 3).squeeze().tolist() == [2.0, 0.0, 5.0]
    assert scatter_max(x, idx, 3).squeeze().tolist() == [3.0, 0.0, 5.0]
    assert scatter_min(x, idx, 3).squeeze().tolist() == [1.0, 0.0, 5.0]
    assert abs(float(scatter_std(x, idx, 3)[0]) - 1.0) < 1e-2


def _graph(ds):
    split = temporal_split(ds.tx)
    feats = edge_features(ds.tx).to_numpy()
    gids, _ = split.snapshot("test")
    return build_graph(ds.tx, feats, gids, Normalizer.fit(feats))


def test_sampler_keeps_seeds_and_bounds_fanout(syn_ds):
    g = _graph(syn_ds)
    s = NeighborSampler(g, [5, 5])
    seeds = np.array([10, 200, 3000])
    b = s.sample(seeds, np.random.default_rng(0))
    assert np.array_equal(b.edge_local[b.seed_pos.numpy()], seeds)
    assert b.seed_label.tolist() == g.label[seeds].tolist()
    assert b.ego.sum() == len(b.seed_nodes)
    assert len(b.edge_local) < g.num_edges
    # pair ids group exactly the parallel edges
    k = b.src.numpy() * b.num_nodes + b.dst.numpy()
    assert len(np.unique(k)) == b.num_pairs


def test_model_is_permutation_invariant_over_edges(syn_ds):
    """Attention-free aggregations: shuffling edge order must not change outputs."""
    torch.manual_seed(0)
    g = _graph(syn_ds)
    s = NeighborSampler(g, [5, 5])
    seeds = np.arange(50)
    b = s.sample(seeds, np.random.default_rng(0))
    cfg = ModelConfig(node_num_dim=3, edge_num_dim=len(EDGE_NUM_FEATURES), num_banks=syn_ds.num_banks,
                      num_currencies=len(syn_ds.currencies), num_formats=len(syn_ds.formats), hidden=16)
    m = MuleNet(cfg).eval()
    out = m(b)["edge_logit_all"].detach()
    perm = torch.randperm(len(b.src))
    b2 = s.make_batch(b.edge_local, seeds)
    for name in ("src", "dst", "pair", "edge_num", "edge_cat"):
        setattr(b2, name, getattr(b2, name)[perm])
    out2 = m(b2)["edge_logit_all"].detach()
    assert torch.allclose(out[perm], out2, atol=1e-4)
