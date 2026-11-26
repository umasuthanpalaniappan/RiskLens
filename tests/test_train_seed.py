"""Training must be reproducible: the same seed gives the same weights and shuffles."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src import models, train


def test_set_seed_makes_random_draws_repeatable():
    train.set_seed(123)
    a = (torch.rand(5), np.random.rand(3))
    train.set_seed(123)
    b = (torch.rand(5), np.random.rand(3))
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_a_seeded_model_starts_from_the_same_weights_every_time():
    train.set_seed(42)
    a = models.LSTMVolModel().state_dict()
    train.set_seed(42)
    b = models.LSTMVolModel().state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_different_seeds_give_different_weights():
    train.set_seed(1)
    a = models.LSTMVolModel().state_dict()
    train.set_seed(2)
    b = models.LSTMVolModel().state_dict()
    assert any(not torch.equal(a[k], b[k]) for k in a)


def test_the_loader_shuffle_order_is_repeatable_under_a_seed():
    ds = torch.utils.data.TensorDataset(torch.arange(20.0), torch.arange(20.0))
    orders = []
    for _ in range(2):
        train.set_seed(5)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
        orders.append([int(x) for xb, _ in loader for x in xb])
    assert orders[0] == orders[1]
