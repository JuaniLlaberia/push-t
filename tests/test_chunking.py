from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from fakes import ACTION_MAX, ACTION_MIN, ConstantModel, FakePushTEnv, synthetic_stats
from pusht_bc.data import env_obs_to_batch
from pusht_bc.evaluate import BCAgent
from pusht_bc.model import BCPolicy
from pusht_bc.normalize import Normalizer
from pusht_bc.train import compute_val_loss, train_one_epoch

pytestmark = pytest.mark.core

K = 8
CHUNK = torch.stack([torch.linspace(-0.9, 0.9, K), torch.linspace(0.9, -0.9, K)], dim=1)

@pytest.fixture()
def normalizer():
    return Normalizer.from_stats(synthetic_stats())

def _obs():
    env = FakePushTEnv([0.1])
    obs, _ = env.reset(seed=0)
    return env_obs_to_batch(obs)

def _expected(normalizer, j):
    return normalizer.unnormalize_action(CHUNK[j]).numpy()

def test_model_called_once_every_n_execute(normalizer, device):
    model = ConstantModel(CHUNK).to(device)
    agent = BCAgent(model, normalizer, device, n_execute=4)
    agent.reset()
    obs = _obs()
    actions = [agent.act(obs) for _ in range(12)]
    assert model.calls == 3
    for t, a in enumerate(actions):
        assert a.shape == (2,)
        assert np.allclose(a, _expected(normalizer, t % 4), atol=1e-3), t

@pytest.mark.parametrize("n_execute, steps, calls", [(8, 16, 2), (1, 5, 5), (3, 7, 3)])
def test_other_n_execute(normalizer, device, n_execute, steps, calls):
    model = ConstantModel(CHUNK).to(device)
    agent = BCAgent(model, normalizer, device, n_execute=n_execute)
    agent.reset()
    obs = _obs()
    for _ in range(steps):
        agent.act(obs)
    assert model.calls == calls

def test_reset_forces_new_prediction(normalizer, device):
    model = ConstantModel(CHUNK).to(device)
    agent = BCAgent(model, normalizer, device, n_execute=4)
    agent.reset()
    obs = _obs()
    agent.act(obs)
    agent.act(obs)
    assert model.calls == 1
    agent.reset()
    a = agent.act(obs)
    assert model.calls == 2
    assert np.allclose(a, _expected(normalizer, 0), atol=1e-3)

def test_chunked_actions_clipped(normalizer, device):
    far = torch.full((K, 2), 7.0)
    far[:, 1] = -7.0
    agent = BCAgent(ConstantModel(far).to(device), normalizer, device, n_execute=4)
    agent.reset()
    assert np.allclose(agent.act(_obs()), [512.0, 0.0])

def test_train_one_epoch_ignores_padded_targets(normalizer, device):
    torch.manual_seed(0)
    n = 32
    g = torch.Generator().manual_seed(0)
    lo_a, hi_a = torch.tensor(ACTION_MIN).float(), torch.tensor(ACTION_MAX).float()
    action = lo_a + torch.rand(n, K, 2, generator=g) * (hi_a - lo_a)
    pad = torch.zeros(n, K, dtype=torch.bool)
    pad[: n // 2, 5:] = True
    action[pad] = 1.0e4
    samples = [
        {
            "features": torch.randn(512, generator=g),
            "observation.state": torch.rand(2, generator=g) * 400 + 50,
            "action": action[i],
            "action_is_pad": pad[i],
        }
        for i in range(n)
    ]
    loader = DataLoader(samples, batch_size=n)
    model = BCPolicy(chunk_size=K).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = [train_one_epoch(model, loader, opt, normalizer, device) for _ in range(300)]
    assert all(math.isfinite(x) for x in losses)
    assert losses[0] < 5.0
    assert min(losses) < 5e-2
    assert compute_val_loss(model, loader, normalizer, device) < 5e-2
