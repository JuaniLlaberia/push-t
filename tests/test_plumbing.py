from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from torch import nn

from fakes import FakePushTEnv, make_synthetic_cache, synthetic_stats
from pusht_bc.config import PROJECT_ROOT, Config, load_config
from pusht_bc.data import (
    CachedFeatureDataset,
    cache_rows_for_episodes,
    env_obs_to_batch,
    frames_for_episodes,
    get_episode_indices,
    make_dataloaders,
    make_image_transform,
)
from pusht_bc.encoders import make_resnet18_encoder
from pusht_bc.evaluate import binomial_ci, parse_seeds, summary_to_dict, EvalSummary, EpisodeResult
from pusht_bc.utils import (
    get_device,
    load_checkpoint,
    plot_coverage_over_time,
    plot_losses,
    save_checkpoint,
    set_seed,
    write_video,
)

# ---------------------------------------------------------------- dataset

def test_dataset_keys_shapes_and_size(pusht_dataset):
    ds = pusht_dataset
    assert ds.num_episodes == 206
    assert len(ds) == 25650
    assert ds.fps == 10
    assert len(get_episode_indices(ds)) == 206
    s = ds[0]
    for key in ["observation.image", "observation.state", "action", "episode_index", "frame_index", "timestamp", "next.reward"]:
        assert key in s, key
    assert s["observation.image"].shape == (3, 96, 96)
    assert s["observation.image"].dtype == torch.float32
    assert 0.0 <= float(s["observation.image"].min()) and float(s["observation.image"].max()) <= 1.0
    assert s["observation.state"].shape == (2,)
    assert s["action"].shape == (2,)
    stats = ds.meta.stats
    for key in ["observation.state", "action"]:
        assert np.asarray(stats[key]["min"]).shape == (2,)
        assert np.asarray(stats[key]["max"]).shape == (2,)
        assert np.all(np.asarray(stats[key]["min"]) >= 0) and np.all(np.asarray(stats[key]["max"]) <= 512)

def test_frames_for_episodes(pusht_dataset):
    ds = pusht_dataset
    frames = frames_for_episodes(ds, [0])
    assert frames == list(range(len(frames)))
    assert all(int(ds.hf_dataset[i]["episode_index"]) == 0 for i in (frames[0], frames[-1]))
    assert int(ds.hf_dataset[frames[-1] + 1]["episode_index"]) == 1
    all_frames = frames_for_episodes(ds, get_episode_indices(ds))
    assert all_frames == list(range(len(ds)))
    a, b = frames_for_episodes(ds, [3, 7]), frames_for_episodes(ds, [3]) + frames_for_episodes(ds, [7])
    assert a == sorted(b)
    with pytest.raises(ValueError):
        frames_for_episodes(ds, [9999])

def test_env_obs_matches_dataset_sample(pusht_dataset, env):
    obs, _ = env.reset(seed=0)
    assert obs["pixels"].dtype == np.uint8 and obs["pixels"].shape == (96, 96, 3)
    assert obs["agent_pos"].shape == (2,)
    b = env_obs_to_batch(obs)
    s = pusht_dataset[0]
    for key in ["observation.image", "observation.state"]:
        assert b[key].dtype == s[key].dtype, key
        assert b[key].shape == s[key].shape, key
    img = b["observation.image"]
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
    assert float(img.max()) > 0.5  # really scaled by 1/255, not left in [0, 255] or squashed
    # same pixels, channels first
    assert torch.allclose(img * 255.0, torch.from_numpy(obs["pixels"]).permute(2, 0, 1).float(), atol=1e-3)
    st = b["observation.state"]
    assert torch.all((st >= 0) & (st <= 512))
    assert torch.allclose(st.double(), torch.from_numpy(obs["agent_pos"]), atol=1e-4)
    # dataset images and env images live in the same range
    assert abs(float(img.mean()) - float(s["observation.image"].mean())) < 0.25

def test_env_obs_to_batch_on_fake_obs():
    env = FakePushTEnv([0.1])
    obs, _ = env.reset(seed=3)
    b = env_obs_to_batch(obs)
    assert set(b) == {"observation.image", "observation.state"}
    assert b["observation.image"].shape == (3, 96, 96)
    assert b["observation.state"].dtype == torch.float32

# ---------------------------------------------------------------- encoder

def _bn_stats(module: nn.Module) -> list[torch.Tensor]:
    return [
        t.clone()
        for m in module.modules()
        if isinstance(m, nn.BatchNorm2d)
        for t in (m.running_mean, m.running_var)
    ]

def test_frozen_encoder_shape_frozen_and_bn_fixed():
    enc = make_resnet18_encoder(pretrained=True, frozen=True)
    assert all(not p.requires_grad for p in enc.parameters())
    before = _bn_stats(enc)
    parent = nn.Sequential(enc)
    parent.train()
    enc.train()
    x = torch.randn(4, 3, 96, 96) * 2 + 0.5
    y = enc(x)
    assert y.shape == (4, 512)
    y84 = enc(torch.randn(2, 3, 84, 84))
    assert y84.shape == (2, 512)
    after = _bn_stats(enc)
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    assert not any(m.training for m in enc.modules() if isinstance(m, nn.BatchNorm2d))

def test_trainable_encoder_is_trainable():
    enc = make_resnet18_encoder(pretrained=False, frozen=False)
    assert all(p.requires_grad for p in enc.parameters())
    enc.train()
    y = enc(torch.randn(2, 3, 96, 96))
    y.sum().backward()
    assert any(p.grad is not None for p in enc.parameters())
    # same state-dict keys as the frozen wrapper (checkpoint loading relies on it)
    assert set(enc.state_dict()) == set(make_resnet18_encoder(pretrained=False, frozen=True).state_dict())

# ---------------------------------------------------------------- feature cache

def test_cached_dataset_no_chunk():
    cache = make_synthetic_cache([5, 12, 3])
    ds = CachedFeatureDataset(cache, list(range(20)))
    item = ds[7]
    assert item["features"].shape == (512,)
    assert item["observation.state"].shape == (2,)
    assert item["action"].shape == (2,)
    assert "action_is_pad" not in item
    assert torch.equal(item["action"], cache["action"][7])

def test_cached_dataset_chunking_padding_and_boundaries():
    lengths = [5, 12, 3]
    cache = make_synthetic_cache(lengths)
    k = 8
    ds = CachedFeatureDataset(cache, list(range(sum(lengths))), chunk_size=k)
    starts = np.cumsum([0] + lengths[:-1])
    for row in range(sum(lengths)):
        item = ds[row]
        assert item["action"].shape == (k, 2)
        assert item["action_is_pad"].shape == (k,)
        assert item["action_is_pad"].dtype == torch.bool
        ep = int(cache["episode_index"][row])
        frame = int(cache["frame_index"][row])
        n_left = lengths[ep] - frame  # real steps from t to the episode end
        exp_pad = torch.arange(k) >= n_left
        assert torch.equal(item["action_is_pad"], exp_pad), row
        codes = item["action"][:, 0]
        # no crossing the episode boundary
        assert torch.all((codes // 1000) == ep), row
        # real steps are t, t+1, ...; padded steps repeat the last real action
        for j in range(k):
            exp_frame = min(frame + j, lengths[ep] - 1)
            assert int(codes[j]) == 1000 * ep + exp_frame, (row, j)
    # padding at episode end vs start
    assert not ds[int(starts[1])]["action_is_pad"].any()
    assert ds[int(starts[1]) + 11]["action_is_pad"].sum() == k - 1

def test_cached_dataset_frame_subset_and_file(tmp_path):
    cache = make_synthetic_cache([5, 12, 3])
    path = tmp_path / "cache.pt"
    torch.save(cache, path)
    rows = cache_rows_for_episodes(cache, [1])
    assert rows == list(range(5, 17))
    ds = CachedFeatureDataset(str(path), rows, chunk_size=4)
    assert len(ds) == 12
    assert int(ds[0]["action"][0, 0]) == 1000
    loader = torch.utils.data.DataLoader(ds, batch_size=5)
    b = next(iter(loader))
    assert b["features"].shape == (5, 512)
    assert b["action"].shape == (5, 4, 2)
    assert b["action_is_pad"].shape == (5, 4)

def _cfg(tmp_path, cache_file, **kw) -> Config:
    base = dict(
        name="t", use_cached_features=True, cache_path=str(cache_file), batch_size=4,
        output_dir=str(tmp_path / "runs"),
    )
    base.update(kw)
    return Config(**base)

def test_make_dataloaders_cached(tmp_path):
    lengths = [6] * 20
    cache = make_synthetic_cache(lengths)
    path = tmp_path / "cache.pt"
    torch.save(cache, path)
    train_eps, val_eps = list(range(2, 20)), [0, 1]
    tl, vl = make_dataloaders(_cfg(tmp_path, path), train_eps, val_eps)
    assert len(tl.dataset) == 18 * 6 and len(vl.dataset) == 2 * 6
    b = next(iter(tl))
    assert set(b) == {"features", "observation.state", "action"}
    val_codes = torch.cat([b["action"][:, 0] for b in vl])
    assert set((val_codes // 1000).long().tolist()) == {0, 1}

    tl_c, _ = make_dataloaders(_cfg(tmp_path, path, chunk_size=8, n_execute=4), train_eps, val_eps)
    b = next(iter(tl_c))
    assert b["action"].shape == (4, 8, 2) and b["action_is_pad"].shape == (4, 8)

    tl_half, _ = make_dataloaders(_cfg(tmp_path, path, train_episode_fraction=0.5), train_eps, val_eps)
    eps_half = set((torch.cat([b["action"][:, 0] for b in tl_half]) // 1000).long().tolist())
    assert len(eps_half) == 9 and eps_half <= set(train_eps)
    tl_half2, _ = make_dataloaders(_cfg(tmp_path, path, train_episode_fraction=0.5), train_eps, val_eps)
    eps_half2 = set((torch.cat([b["action"][:, 0] for b in tl_half2]) // 1000).long().tolist())
    assert eps_half == eps_half2

    with pytest.raises(ValueError):
        make_dataloaders(_cfg(tmp_path, path), [0, 1, 2], [2, 3])

def test_image_transform():
    assert make_image_transform(None, train=True) is None
    img = torch.rand(3, 96, 96)
    assert make_image_transform(84, train=True)(img).shape == (3, 84, 84)
    center = make_image_transform(84, train=False)
    assert torch.equal(center(img), center(img))
    assert torch.equal(center(img), img[:, 6:90, 6:90])
    assert make_image_transform(84, train=False)(torch.rand(5, 3, 96, 96)).shape == (5, 3, 84, 84)

# ---------------------------------------------------------------- config

def test_load_config_v0():
    cfg = load_config(PROJECT_ROOT / "configs" / "bc_v0.yaml")
    assert cfg.name == "bc_v0" and cfg.use_cached_features and cfg.freeze_encoder
    assert cfg.eval.dev_seed_list() == list(range(50))
    assert cfg.eval.final_seed_list() == list(range(1000, 1200))
    assert cfg.eval.record_seeds == [0, 1, 2]
    assert Config.from_dict(cfg.to_dict()) == cfg

def test_load_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\nlearning_rate: 0.1\n")
    with pytest.raises(ValueError, match="learning_rate"):
        load_config(p)
    p.write_text("name: x\neval:\n  seeds: [0, 1]\n")
    with pytest.raises(ValueError, match="seeds"):
        load_config(p)

def test_config_validation():
    with pytest.raises(ValueError):
        Config(loss="huber")
    with pytest.raises(ValueError):
        Config(chunk_size=8)  # n_execute missing
    Config(chunk_size=8, n_execute=4)

# ---------------------------------------------------------------- utils

def test_get_device_and_seed():
    assert isinstance(get_device(), torch.device)
    set_seed(123)
    a = (torch.rand(3), np.random.rand(3))
    set_seed(123)
    b = (torch.rand(3), np.random.rand(3))
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1])

class _TinyPolicy(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(4, 2)

def test_checkpoint_roundtrip_frozen_encoder_not_saved(tmp_path):
    enc = make_resnet18_encoder(pretrained=False, frozen=True)
    model = _TinyPolicy(enc)
    cfg = Config(name="x", freeze_encoder=True)
    p = save_checkpoint(tmp_path / "a.pt", model, synthetic_stats(), cfg, epoch=3, extra={"val_loss": 0.5})
    ck = load_checkpoint(p)
    assert ck["encoder"] is None
    assert ck["epoch"] == 3 and ck["extra"]["val_loss"] == 0.5
    assert ck["config"] == cfg
    assert np.array_equal(ck["normalizer_stats"]["action"]["min"], synthetic_stats()["action"]["min"])
    for k, v in model.head.state_dict().items():
        assert torch.equal(ck["head"][k], v)
    assert p.stat().st_size < 1_000_000  # no ResNet weights inside

def test_checkpoint_trained_encoder_saved(tmp_path):
    enc = make_resnet18_encoder(pretrained=False, frozen=False)
    model = _TinyPolicy(enc)
    cfg = Config(name="x", use_cached_features=False, freeze_encoder=False, crop=84)
    ck = load_checkpoint(save_checkpoint(tmp_path / "b.pt", model, {"a": 1}, cfg, epoch=1))
    assert ck["encoder"] is not None
    fresh = make_resnet18_encoder(pretrained=False, frozen=True)
    fresh.load_state_dict(ck["encoder"])
    for k, v in enc.state_dict().items():
        assert torch.equal(fresh.state_dict()[k], v)

def test_video_and_plots(tmp_path):
    frames = [np.full((96, 96, 3), i * 10, dtype=np.uint8) for i in range(12)]
    p = write_video(frames, tmp_path / "v.mp4", fps=10)
    assert p.exists() and p.stat().st_size > 0
    write_video([np.zeros((81, 81, 3), np.uint8)] * 3, tmp_path / "odd.mp4")
    import imageio.v3 as iio

    assert sum(1 for _ in iio.imiter(p)) == 12
    hist = {"train_loss": list(np.linspace(1, 0.1, 40)), "val_loss": [0.8, 0.5, 0.3, 0.25]}
    assert plot_losses(hist, tmp_path / "loss.png").stat().st_size > 0
    cov = plot_coverage_over_time({"a": [0.1, 0.5, 0.96], "b": [0.2] * 10}, tmp_path / "cov.png")
    assert cov.stat().st_size > 0

# ---------------------------------------------------------------- evaluation helpers

def test_binomial_ci():
    lo, hi = binomial_ci(32, 50)
    assert 0.0 <= lo < 0.64 < hi <= 1.0
    assert 0.49 < lo < 0.51 and 0.75 < hi < 0.77  # Wilson: ~0.501, ~0.759
    assert binomial_ci(0, 10)[0] == 0.0 and binomial_ci(10, 10)[1] == 1.0
    lo200, hi200 = binomial_ci(128, 200)
    assert hi200 - lo200 < hi - lo
    assert binomial_ci(0, 0) == (0.0, 1.0)

def test_parse_seeds_and_summary_json():
    assert parse_seeds("0-49") == list(range(50))
    assert parse_seeds("0,1,2") == [0, 1, 2]
    assert parse_seeds("0-2,10") == [0, 1, 2, 10]
    assert parse_seeds("") == []
    ep = EpisodeResult(seed=0, success=True, max_coverage=0.97, steps=3, rewards=[0.1, 0.5, 0.97],
                       frames=[np.zeros((2, 2, 3), np.uint8)])
    s = EvalSummary(success_rate=1.0, ci_low=0.2, ci_high=1.0, avg_max_coverage=0.97, n_episodes=1, episodes=[ep])
    d = summary_to_dict(s)
    json.dumps(d)
    assert "frames" not in d["episodes"][0]

# ---------------------------------------------------------------- pretrained Diffusion Policy

def test_pretrained_dp_act_on_real_obs(env):
    try:
        from pusht_bc.policies.pretrained_dp import PretrainedDiffusionAgent

        agent = PretrainedDiffusionAgent(device="cpu")
    except (OSError, ConnectionError) as e:  # offline / Hub unavailable
        pytest.skip(f"lerobot/diffusion_pusht unavailable: {e}")
    agent.reset()
    obs, _ = env.reset(seed=0)
    a = agent.act(env_obs_to_batch(obs))
    assert isinstance(a, np.ndarray) and a.shape == (2,)
    assert np.issubdtype(a.dtype, np.floating)
    assert np.all((a >= 0) & (a <= 512)) and np.all(np.isfinite(a))
    obs, *_ = env.step(a)
    a2 = agent.act(env_obs_to_batch(obs))
    assert a2.shape == (2,) and math.dist(a, a2) < 100  # consecutive actions of one chunk stay close
