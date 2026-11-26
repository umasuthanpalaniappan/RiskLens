"""Shape and behaviour tests for the LSTM and the gated GNN (CPU, tiny inputs)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src import config, graph, models

N, WINDOW, F = 4, 10, len(config.FEATURE_COLUMNS)


def _x(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(N, WINDOW, F, generator=g)


def _laplacian(edges):
    adj = np.zeros((N, N))
    for i, j, w in edges:
        adj[i, j] = adj[j, i] = w
    return torch.tensor(graph.graph_laplacian(adj), dtype=torch.float32)


def test_lstm_outputs_one_ratio_per_sample():
    out = models.LSTMVolModel()(_x())
    assert out.shape == (N,)


def test_gnn_returns_four_per_stock_outputs():
    model = models.GNNVolModel(num_nodes=N).eval()
    outs = model(_x(), _laplacian([(0, 1, 0.8), (1, 2, 0.5)]))
    assert len(outs) == 4 and all(o.shape == (N,) for o in outs)


def test_gate_starts_at_one_half_and_stays_inside_zero_one():
    model = models.GNNVolModel(num_nodes=N).eval()
    alpha = model(_x(), _laplacian([(0, 1, 0.8)]))[3]
    assert torch.allclose(alpha, torch.full((N,), 0.5))
    with torch.no_grad():
        model.raw_alpha.copy_(torch.tensor([-10.0, -1.0, 1.0, 10.0]))
    alpha = model(_x(), _laplacian([(0, 1, 0.8)]))[3]
    assert ((alpha > 0) & (alpha < 1)).all()


def test_prediction_is_the_gate_weighted_blend_of_the_two_heads():
    model = models.GNNVolModel(num_nodes=N).eval()
    with torch.no_grad():
        model.raw_alpha.copy_(torch.tensor([-2.0, -0.5, 0.5, 2.0]))
    blended, lstm_pred, gnn_pred, alpha = model(_x(), _laplacian([(0, 1, 0.8), (2, 3, 0.4)]))
    assert torch.allclose(blended, alpha * gnn_pred + (1 - alpha) * lstm_pred, atol=1e-6)


def test_graph_branch_ignores_a_stocks_own_history():
    """The propagation matrix has a zero diagonal (no self-loops): with one graph layer,
    a stock's graph prediction depends on its neighbours only. Its own history reaches the
    final forecast through the LSTM head and the gate."""
    model = models.GNNVolModel(num_nodes=N, gnn_layers=1).eval()
    lap = _laplacian([(0, 1, 0.8)])
    x = _x()
    _, lstm_a, gnn_a, _ = model(x, lap)
    x2 = x.clone()
    x2[0] += 5.0  # change stock 0's own history only
    _, lstm_b, gnn_b, _ = model(x2, lap)
    assert not torch.allclose(lstm_a[0], lstm_b[0])  # LSTM head sees it
    assert torch.allclose(gnn_a[0], gnn_b[0], atol=1e-6)  # graph head does not
    assert not torch.allclose(gnn_a[1], gnn_b[1])  # its neighbour's graph head does


def test_a_graph_without_edges_gives_every_stock_the_same_graph_prediction():
    model = models.GNNVolModel(num_nodes=N).eval()
    _, _, gnn_pred, _ = model(_x(), _laplacian([]))
    assert torch.allclose(gnn_pred, gnn_pred[0].expand(N), atol=1e-6)


def test_same_seed_gives_identical_initial_weights():
    torch.manual_seed(7)
    a = models.GNNVolModel(num_nodes=N).state_dict()
    torch.manual_seed(7)
    b = models.GNNVolModel(num_nodes=N).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
