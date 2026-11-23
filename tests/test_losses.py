from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from pusht_bc.losses import bc_loss

pytestmark = pytest.mark.core

def _rand(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))

def test_zero_for_equal_inputs():
    x = _rand(8, 2)
    assert float(bc_loss(x, x.clone())) == 0.0
    assert float(bc_loss(x, x.clone(), kind="l1")) == 0.0

def test_scalar_and_equals_mse_without_mask():
    p, t = _rand(8, 2, seed=1), _rand(8, 2, seed=2)
    loss = bc_loss(p, t)
    assert loss.ndim == 0
    assert torch.allclose(loss, F.mse_loss(p, t))

def test_l1_equals_l1_loss():
    p, t = _rand(8, 2, seed=1), _rand(8, 2, seed=2)
    assert torch.allclose(bc_loss(p, t, kind="l1"), F.l1_loss(p, t))
    pk, tk = _rand(4, 8, 2, seed=3), _rand(4, 8, 2, seed=4)
    assert torch.allclose(bc_loss(pk, tk, kind="l1"), F.l1_loss(pk, tk))

def test_mask_b_masked_entries_ignored():
    p, t = _rand(6, 2, seed=1), _rand(6, 2, seed=2)
    mask = torch.tensor([True, False, True, True, False, True])
    loss = bc_loss(p, t, mask)
    assert torch.allclose(loss, F.mse_loss(p[mask], t[mask]))
    p2 = p.clone()
    p2[~mask] += 100.0
    assert torch.allclose(bc_loss(p2, t, mask), loss)

def test_mask_bk_on_chunks():
    p, t = _rand(4, 8, 2, seed=1), _rand(4, 8, 2, seed=2)
    mask = torch.ones(4, 8, dtype=torch.bool)
    mask[0, 5:] = False
    mask[2, 1:] = False
    loss = bc_loss(p, t, mask)
    assert torch.allclose(loss, F.mse_loss(p[mask], t[mask]))
    p2 = p.clone()
    p2[~mask] = 1e3
    assert torch.allclose(bc_loss(p2, t, mask), loss)
    assert torch.allclose(bc_loss(p2, t, mask, kind="l1"), F.l1_loss(p[mask], t[mask]))
    assert torch.allclose(bc_loss(p, t, torch.ones(4, 8, dtype=torch.bool)), F.mse_loss(p, t))

def test_all_false_mask_returns_zero_without_nan():
    p = _rand(4, 8, 2, seed=1).requires_grad_(True)
    t = _rand(4, 8, 2, seed=2)
    loss = bc_loss(p, t, torch.zeros(4, 8, dtype=torch.bool))
    assert torch.isfinite(loss) and float(loss) == 0.0
    if loss.requires_grad:
        loss.backward()
        assert p.grad is None or torch.isfinite(p.grad).all()
    loss_b = bc_loss(_rand(3, 2), _rand(3, 2, seed=1), torch.zeros(3, dtype=torch.bool))
    assert torch.isfinite(loss_b) and float(loss_b) == 0.0

def test_gradient_flows():
    p = _rand(5, 2, seed=1).requires_grad_(True)
    bc_loss(p, _rand(5, 2, seed=2)).backward()
    assert p.grad is not None and p.grad.abs().sum() > 0
