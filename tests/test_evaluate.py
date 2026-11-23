from __future__ import annotations

import numpy as np
import pytest
import torch

from fakes import ACTION_MAX, ACTION_MIN, ConstantModel, FakePushTEnv, FixedAgent, synthetic_stats
from pusht_bc.data import env_obs_to_batch, make_image_transform
from pusht_bc.evaluate import BCAgent, EpisodeResult, EvalSummary, binomial_ci, evaluate, rollout
from pusht_bc.normalize import Normalizer

pytestmark = pytest.mark.core

class _CheckingAgent(FixedAgent):
    def act(self, obs_batch):
        assert set(obs_batch) >= {"observation.image", "observation.state"}
        assert obs_batch["observation.image"].shape == (3, 96, 96)
        assert obs_batch["observation.image"].dtype == torch.float32
        assert float(obs_batch["observation.image"].max()) <= 1.0
        assert obs_batch["observation.state"].shape == (2,)
        return super().act(obs_batch)

def test_rollout_success_stops_at_terminating_step():
    env = FakePushTEnv([0.2, 0.5, 0.96, 0.4])
    agent = _CheckingAgent(action=(100.0, 200.0))
    res = rollout(agent, env, seed=7, max_steps=300)
    assert isinstance(res, EpisodeResult)
    assert res.seed == 7 and env.reset_seeds == [7]
    assert res.success
    assert res.max_coverage == pytest.approx(0.96)
    assert res.steps == 3
    assert res.rewards == pytest.approx([0.2, 0.5, 0.96])
    assert len(env.actions) == 3
    assert np.allclose(env.actions[0], [100.0, 200.0])
    assert agent.resets >= 1
    assert res.frames is None

def test_rollout_failure_runs_to_max_steps():
    env = FakePushTEnv([0.1, 0.3, 0.5], terminate_at_success=False)
    res = rollout(FixedAgent(), env, seed=0, max_steps=20)
    assert not res.success
    assert res.steps == 20
    assert len(res.rewards) == 20 and len(env.actions) == 20
    assert res.max_coverage == pytest.approx(0.5)

def test_success_uses_max_coverage_not_final():
    env = FakePushTEnv([0.2, 0.97, 0.5], terminate_at_success=False)
    res = rollout(FixedAgent(), env, seed=0, max_steps=10)
    assert res.success
    assert res.max_coverage == pytest.approx(0.97)
    assert res.steps == 10

def test_rollout_stops_on_truncation():
    env = FakePushTEnv([0.1], terminate_at_success=False, truncate_after=5)
    res = rollout(FixedAgent(), env, seed=0, max_steps=300)
    assert res.steps == 5 and not res.success

def test_rollout_record_frames():
    env = FakePushTEnv([0.2, 0.5, 0.96])
    res = rollout(FixedAgent(), env, seed=1, record=True)
    assert isinstance(res.frames, list) and len(res.frames) >= res.steps
    for f in res.frames:
        f = np.asarray(f)
        assert f.dtype == np.uint8 and f.ndim == 3 and f.shape[-1] == 3

def test_evaluate_aggregates():
    rewards = {0: [0.96], 1: [0.3, 0.5], 2: [0.4, 0.99], 3: [0.1]}
    env = FakePushTEnv(rewards)
    seeds = [0, 1, 2, 3]
    s = evaluate(FixedAgent(), env, seeds, record_seeds=(2,))
    assert isinstance(s, EvalSummary)
    assert s.n_episodes == 4
    assert s.success_rate == pytest.approx(0.5)
    assert s.avg_max_coverage == pytest.approx((0.96 + 0.5 + 0.99 + 0.1) / 4)
    lo, hi = binomial_ci(2, 4)
    assert (s.ci_low, s.ci_high) == (pytest.approx(lo), pytest.approx(hi))
    assert [e.seed for e in s.episodes] == seeds
    assert [e.success for e in s.episodes] == [True, False, True, False]
    assert s.episodes[2].frames is not None
    assert all(s.episodes[i].frames is None for i in (0, 1, 3))

@pytest.fixture()
def normalizer():
    return Normalizer.from_stats(synthetic_stats())

def _obs(seed=0):
    env = FakePushTEnv([0.1])
    obs, _ = env.reset(seed=seed)
    return env_obs_to_batch(obs)

def test_bcagent_clips_to_action_range(normalizer, device):
    agent = BCAgent(ConstantModel([5.0, -5.0]).to(device), normalizer, device)
    agent.reset()
    a = agent.act(_obs())
    assert isinstance(a, np.ndarray) and a.shape == (2,)
    assert np.issubdtype(a.dtype, np.floating)
    assert np.allclose(a, [512.0, 0.0])

def test_bcagent_unnormalizes(normalizer, device):
    agent = BCAgent(ConstantModel([-1.0, 1.0]).to(device), normalizer, device)
    agent.reset()
    a = agent.act(_obs())
    assert np.allclose(a, [ACTION_MIN[0], ACTION_MAX[1]], atol=1e-3)

def test_bcagent_feeds_normalized_inputs(normalizer, device):
    model = ConstantModel([0.0, 0.0]).to(device)
    agent = BCAgent(model, normalizer, device)
    agent.reset()
    obs = _obs(seed=3)
    agent.act(obs)
    img, state = model.last_inputs
    assert img.shape == (1, 3, 96, 96) and state.shape == (1, 2)
    exp_img = normalizer.normalize_image(obs["observation.image"].unsqueeze(0))
    exp_state = normalizer.normalize_state(obs["observation.state"].unsqueeze(0))
    assert torch.allclose(img.cpu(), exp_img.cpu(), atol=1e-5)
    assert torch.allclose(state.cpu(), exp_state.cpu(), atol=1e-5)

def test_bcagent_applies_image_transform(normalizer, device):
    model = ConstantModel([0.0, 0.0]).to(device)
    agent = BCAgent(model, normalizer, device, image_transform=make_image_transform(84, train=False))
    agent.reset()
    agent.act(_obs())
    assert model.last_inputs[0].shape == (1, 3, 84, 84)

def test_bcagent_through_rollout(normalizer, device):
    agent = BCAgent(ConstantModel([0.0, 0.0]).to(device), normalizer, device)
    env = FakePushTEnv([0.1, 0.2, 0.3], terminate_at_success=False)
    res = rollout(agent, env, seed=0, max_steps=6)
    assert res.steps == 6
    mid = (ACTION_MIN + ACTION_MAX) / 2
    assert all(np.allclose(a, mid, atol=1e-3) for a in env.actions)
