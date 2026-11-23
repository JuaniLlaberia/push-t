from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from fakes import ACTION_MAX, ACTION_MIN, STATE_MAX, STATE_MIN, make_synthetic_cache, synthetic_stats
from pusht_bc.config import Config
from pusht_bc.model import BCPolicy
from pusht_bc.normalize import Normalizer
from pusht_bc.train import TrainResult, compute_val_loss, train, train_one_epoch
from pusht_bc.utils import checkpoint_path, load_checkpoint

pytestmark = pytest.mark.core

class _DictDataset(Dataset):
    def __init__(self, **tensors):
        self.tensors = tensors
        self.n = len(next(iter(tensors.values())))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.tensors.items()}

def _synthetic_samples(n: int, seed: int = 0) -> _DictDataset:
    g = torch.Generator().manual_seed(seed)
    lo_s, hi_s = torch.tensor(STATE_MIN).float(), torch.tensor(STATE_MAX).float()
    lo_a, hi_a = torch.tensor(ACTION_MIN).float(), torch.tensor(ACTION_MAX).float()
    return _DictDataset(
        features=torch.randn(n, 512, generator=g),
        **{
            "observation.state": lo_s + torch.rand(n, 2, generator=g) * (hi_s - lo_s),
            "action": lo_a + torch.rand(n, 2, generator=g) * (hi_a - lo_a),
        },
    )

def test_overfits_32_samples(device):
    torch.manual_seed(0)
    loader = DataLoader(_synthetic_samples(32), batch_size=32, shuffle=True)
    model = BCPolicy().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    norm = Normalizer.from_stats(synthetic_stats())
    losses = []
    for _ in range(500):
        losses.append(train_one_epoch(model, loader, opt, norm, device))
        if losses[-1] < 1e-2:
            break
    assert all(isinstance(x, float) and math.isfinite(x) for x in losses)
    assert losses[0] < 5.0
    assert min(losses) < 1e-2, f"best loss {min(losses):.4f} after {len(losses)} steps"

def test_compute_val_loss_float_and_no_update(device):
    torch.manual_seed(0)
    loader = DataLoader(_synthetic_samples(40, seed=1), batch_size=16)
    model = BCPolicy().to(device)
    norm = Normalizer.from_stats(synthetic_stats())
    before = {k: v.clone() for k, v in model.state_dict().items()}
    v1 = compute_val_loss(model, loader, norm, device)
    v2 = compute_val_loss(model, loader, norm, device)
    assert isinstance(v1, float) and math.isfinite(v1) and v1 > 0
    assert v1 == pytest.approx(v2)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k
    assert all(p.grad is None for p in model.parameters())

def test_val_loss_drops_after_training(device):
    torch.manual_seed(0)
    loader = DataLoader(_synthetic_samples(32), batch_size=32)
    model = BCPolicy().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    norm = Normalizer.from_stats(synthetic_stats())
    start = compute_val_loss(model, loader, norm, device)
    for _ in range(50):
        train_one_epoch(model, loader, opt, norm, device)
    assert compute_val_loss(model, loader, norm, device) < start

def test_train_end_to_end_on_tiny_cache(tmp_path):
    cache_file = tmp_path / "cache.pt"
    torch.save(make_synthetic_cache([6] * 20), cache_file)
    cfg = Config(
        name="tiny",
        use_cached_features=True,
        cache_path=str(cache_file),
        batch_size=16,
        epochs=2,
        output_dir=str(tmp_path / "runs"),
    )
    result = train(cfg)
    assert isinstance(result, TrainResult)
    # 20 episodes, 2 held out -> 18 * 6 = 108 frames -> 7 batches of 16 per epoch
    assert len(result.history["train_loss"]) == 2 * 7
    assert len(result.history["val_loss"]) == 2
    assert result.best_val_loss == pytest.approx(min(result.history["val_loss"]))
    assert [str(p) for p in result.checkpoint_paths] == [str(checkpoint_path(cfg, e)) for e in (1, 2)]
    for e, p in enumerate(result.checkpoint_paths, start=1):
        ck = load_checkpoint(p)
        assert ck["epoch"] == e
        assert ck["encoder"] is None
        assert ck["config"].name == "tiny"
        Normalizer.from_stats(ck["normalizer_stats"])

def test_checkpoint_loads_for_eval(tmp_path):
    from pusht_bc.evaluate import load_policy_for_eval

    cache_file = tmp_path / "cache.pt"
    torch.save(make_synthetic_cache([6] * 10), cache_file)
    cfg = Config(name="tiny", cache_path=str(cache_file), batch_size=16, epochs=1, output_dir=str(tmp_path / "runs"))
    result = train(cfg)
    model, stats = load_policy_for_eval(result.checkpoint_paths[-1], torch.device("cpu"))
    assert model.encoder is not None
    out = model(torch.rand(2, 3, 96, 96), torch.zeros(2, 2))
    assert out.shape == (2, 2)
