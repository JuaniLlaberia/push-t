from __future__ import annotations

import pytest
import torch

from fakes import ACTION_MAX, ACTION_MIN, IMAGENET_MEAN, STATE_MAX, STATE_MIN, synthetic_stats
from pusht_bc.normalize import Normalizer

pytestmark = pytest.mark.core

@pytest.fixture()
def norm() -> Normalizer:
    return Normalizer.from_stats(synthetic_stats())

def _t(x, device=None):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def test_min_max_map_to_minus_one_plus_one(norm):
    assert torch.allclose(norm.normalize_state(_t(STATE_MIN)), _t([-1.0, -1.0]), atol=1e-6)
    assert torch.allclose(norm.normalize_state(_t(STATE_MAX)), _t([1.0, 1.0]), atol=1e-6)
    assert torch.allclose(norm.normalize_action(_t(ACTION_MIN)), _t([-1.0, -1.0]), atol=1e-6)
    assert torch.allclose(norm.normalize_action(_t(ACTION_MAX)), _t([1.0, 1.0]), atol=1e-6)
    mid = (ACTION_MIN + ACTION_MAX) / 2
    assert torch.allclose(norm.normalize_action(_t(mid)), _t([0.0, 0.0]), atol=1e-6)

def test_state_and_action_use_their_own_stats(norm):
    x = _t(STATE_MIN)
    assert not torch.allclose(norm.normalize_action(x), norm.normalize_state(x))

@pytest.mark.parametrize("shape", [(16, 2), (4, 8, 2), (2,)])
def test_action_round_trip(norm, shape):
    g = torch.Generator().manual_seed(0)
    a = torch.rand(*shape, generator=g) * 512
    a_norm = norm.normalize_action(a)
    assert a_norm.shape == a.shape
    assert torch.max(torch.abs(norm.unnormalize_action(a_norm) - a)) < 1e-5 * 512
    z = torch.rand(*shape, generator=g) * 2 - 1
    assert torch.max(torch.abs(norm.normalize_action(norm.unnormalize_action(z)) - z)) < 1e-5

def test_state_batch_shapes(norm):
    assert norm.normalize_state(torch.rand(5, 2) * 512).shape == (5, 2)
    assert norm.normalize_state(torch.rand(3, 4, 2) * 512).shape == (3, 4, 2)

def test_zero_range_dimension_is_finite():
    stats = synthetic_stats()
    stats["observation.state"]["max"][1] = stats["observation.state"]["min"][1]
    stats["action"]["max"][0] = stats["action"]["min"][0]
    n = Normalizer.from_stats(stats)
    x = torch.rand(10, 2) * 512
    assert torch.isfinite(n.normalize_state(x)).all()
    assert torch.isfinite(n.normalize_action(x)).all()
    assert torch.isfinite(n.unnormalize_action(torch.rand(10, 2) * 2 - 1)).all()

def test_image_shape_and_imagenet_mean_maps_to_zero(norm):
    img = torch.rand(4, 3, 96, 96)
    assert norm.normalize_image(img).shape == img.shape
    assert norm.normalize_image(torch.rand(3, 84, 84)).shape == (3, 84, 84)
    mean_img = _t(IMAGENET_MEAN).view(1, 3, 1, 1).expand(2, 3, 96, 96).clone()
    assert torch.allclose(norm.normalize_image(mean_img), torch.zeros_like(mean_img), atol=1e-5)
    ones = norm.normalize_image(torch.ones(1, 3, 8, 8))
    assert ones[0, 0, 0, 0] > 2.0

def test_works_on_test_device(norm, device):
    x = (torch.rand(8, 2) * 512).to(device)
    out = norm.normalize_state(x)
    assert out.device.type == device.type
    a = norm.unnormalize_action(norm.normalize_action(x))
    assert a.device.type == device.type
    assert torch.allclose(a.cpu(), x.cpu(), atol=1e-3)
    img = torch.rand(2, 3, 96, 96, device=device)
    assert norm.normalize_image(img).device.type == device.type

def test_state_dict_rebuilds_equivalent_normalizer(norm):
    rebuilt = Normalizer.from_stats(norm.state_dict())
    x = torch.rand(6, 2) * 512
    assert torch.allclose(rebuilt.normalize_state(x), norm.normalize_state(x))
    assert torch.allclose(rebuilt.normalize_action(x), norm.normalize_action(x))
    z = torch.rand(6, 2) * 2 - 1
    assert torch.allclose(rebuilt.unnormalize_action(z), norm.unnormalize_action(z))
    img = torch.rand(1, 3, 96, 96)
    assert torch.allclose(rebuilt.normalize_image(img), norm.normalize_image(img))
