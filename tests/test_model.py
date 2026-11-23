from __future__ import annotations

import pytest
import torch

from pusht_bc.encoders import make_resnet18_encoder
from pusht_bc.model import BCPolicy

pytestmark = pytest.mark.core

def _n_trainable(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

@pytest.fixture(scope="module")
def frozen_encoder():
    return make_resnet18_encoder(pretrained=True, frozen=True)

def test_cached_path_shape():
    m = BCPolicy()
    out = m(torch.randn(7, 512), torch.rand(7, 2) * 2 - 1)
    assert out.shape == (7, 2)
    assert out.dtype == torch.float32

def test_attributes_exist():
    m = BCPolicy()
    assert hasattr(m, "encoder") and m.encoder is None
    assert isinstance(m.head, torch.nn.Module)
    assert m.chunk_size is None

def test_default_cached_model_parameter_count(capsys):
    m = BCPolicy()
    n = _n_trainable(m)
    with capsys.disabled():
        print(f"\n[test_model] trainable parameters of BCPolicy(): {n:,}")
    assert n == 198_146
    assert _n_trainable(m.head) == n

def test_image_path_with_frozen_encoder(frozen_encoder):
    m = BCPolicy(encoder=frozen_encoder)
    m.train()
    out = m(torch.randn(3, 3, 96, 96), torch.rand(3, 2) * 2 - 1)
    assert out.shape == (3, 2)
    assert all(not p.requires_grad for p in m.encoder.parameters())
    assert all(p.requires_grad for p in m.head.parameters())
    assert _n_trainable(m) == _n_trainable(m.head)
    out.sum().backward()
    assert all(p.grad is not None for p in m.head.parameters())
    assert all(p.grad is None for p in m.encoder.parameters())

def test_chunked_output_shape(frozen_encoder):
    m = BCPolicy(chunk_size=8)
    assert m.chunk_size == 8
    assert m(torch.randn(5, 512), torch.rand(5, 2)).shape == (5, 8, 2)
    mi = BCPolicy(encoder=frozen_encoder, chunk_size=8)
    assert mi(torch.randn(2, 3, 96, 96), torch.rand(2, 2)).shape == (2, 8, 2)

def test_state_changes_output():
    torch.manual_seed(0)
    m = BCPolicy()
    feat = torch.randn(1, 512).expand(2, 512)
    out = m(feat, torch.tensor([[-1.0, -1.0], [1.0, 1.0]]))
    assert not torch.allclose(out[0], out[1])

def test_head_state_dict_roundtrip():
    a, b = BCPolicy(), BCPolicy()
    b.head.load_state_dict(a.head.state_dict())
    x, s = torch.randn(4, 512), torch.rand(4, 2)
    assert torch.allclose(a(x, s), b(x, s))

def test_trainable_encoder_stays_trainable():
    enc = make_resnet18_encoder(pretrained=False, frozen=False)
    m = BCPolicy(encoder=enc)
    assert all(p.requires_grad for p in m.encoder.parameters())
    assert m(torch.randn(2, 3, 84, 84), torch.rand(2, 2)).shape == (2, 2)
