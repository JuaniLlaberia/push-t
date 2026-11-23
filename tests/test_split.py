from __future__ import annotations

import numpy as np
import pytest

from pusht_bc.split import split_episodes

pytestmark = pytest.mark.core

def test_deterministic_for_seed():
    eps = list(range(206))
    assert split_episodes(eps, 0.1, seed=0) == split_episodes(eps, 0.1, seed=0)

def test_different_seeds_differ():
    eps = list(range(100))
    assert split_episodes(eps, 0.1, seed=0)[1] != split_episodes(eps, 0.1, seed=1)[1]

def test_disjoint_and_full_coverage():
    eps = [3, 7, 10, 11, 12, 20, 21, 40, 41, 55, 60, 61, 70, 80, 99]
    train, val = split_episodes(eps, 0.2, seed=5)
    assert not set(train) & set(val)
    assert sorted(train + val) == sorted(eps)

@pytest.mark.parametrize(
    "n, frac, expected",
    [(206, 0.1, 21), (100, 0.1, 10), (50, 0.2, 10), (10, 0.3, 3), (5, 0.1, 1), (3, 0.01, 1)],
)
def test_validation_size(n, frac, expected):
    train, val = split_episodes(list(range(n)), frac, seed=0)
    assert len(val) == expected
    assert len(train) == n - expected

def test_sorted_plain_int_lists():
    eps = list(np.arange(30)[::-1])
    train, val = split_episodes(eps, 0.25, seed=2)
    for lst in (train, val):
        assert isinstance(lst, list)
        assert lst == sorted(lst)
        assert all(type(x) is int for x in lst)

def test_input_not_modified():
    eps = list(range(20))[::-1]
    copy = list(eps)
    split_episodes(eps, 0.2, seed=0)
    assert eps == copy
