"""Unit tests for the rolling correlation graph and its Laplacian."""
import numpy as np
import pandas as pd
import pytest

from src import graph


def _window():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    return pd.DataFrame({"A": a, "B": a + rng.normal(0, 0.1, 200), "C": -a + rng.normal(0, 0.1, 200), "D": rng.normal(size=200)})


def test_correlation_matrix_is_symmetric_with_unit_diagonal_and_keeps_signs():
    c = graph.correlation_matrix(_window())
    assert np.allclose(c, c.T)
    assert np.allclose(np.diag(c), 1.0)
    assert c[0, 1] > 0.9 and c[0, 2] < -0.9


def test_adjacency_is_unsigned_with_an_empty_diagonal():
    adj = graph.correlation_adjacency(_window(), threshold=0.0)
    assert (adj >= 0).all()
    assert np.allclose(np.diag(adj), 0.0)
    assert adj[0, 2] == pytest.approx(abs(graph.correlation_matrix(_window())[0, 2]))  # negative link kept as strength


def test_adjacency_threshold_removes_weak_links():
    adj = graph.correlation_adjacency(_window(), threshold=0.5)
    nonzero = adj[adj > 0]
    assert nonzero.size > 0 and (nonzero >= 0.5).all()


def test_two_node_normalised_laplacian_is_independent_of_edge_weight():
    for w in (0.1, 0.5, 3.0):
        lap = graph.graph_laplacian(np.array([[0.0, w], [w, 0.0]]))
        assert np.allclose(lap, [[1.0, -1.0], [-1.0, 1.0]])


def test_normalised_laplacian_eigenvalues_lie_in_zero_two():
    lap = graph.graph_laplacian(graph.correlation_adjacency(_window(), threshold=0.0))
    assert np.allclose(lap, lap.T)
    eig = np.linalg.eigvalsh(lap)
    assert eig.min() > -1e-9 and eig.max() < 2 + 1e-9


def test_combinatorial_laplacian_rows_sum_to_zero():
    adj = graph.correlation_adjacency(_window(), threshold=0.0)
    assert np.allclose(graph.graph_laplacian(adj, normalized=False).sum(axis=1), 0.0)


def test_isolated_node_gives_finite_values_and_receives_nothing_from_neighbours():
    adj = np.array([[0.0, 0.8, 0.0], [0.8, 0.0, 0.0], [0.0, 0.0, 0.0]])
    lap = graph.graph_laplacian(adj)
    assert np.isfinite(lap).all()
    propagation = np.eye(3) - lap  # what the GNN layer multiplies by
    assert np.allclose(propagation[2], 0.0)
