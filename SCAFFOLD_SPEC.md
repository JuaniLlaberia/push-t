# SCAFFOLD_SPEC.md: Push-T behavior cloning repo (step 5)

You are a coding agent setting up a learning repository. The human will implement the **core** of a behavior-cloning (BC) policy by hand, following a separate guide. Your job is to build **everything else**: environment setup, data plumbing, encoders, checkpointing, video, plotting, the pretrained-policy adapter, notebooks, configs, and the **tests** that define the core's behavior.

**The most important rule:** do not implement, sketch or hint at the core. Every function marked `TODO(core)` below must be a stub that raises `NotImplementedError("TODO(core): see guide section N")`. Its docstring states inputs, outputs, shapes and contract only. No pseudo-code, no "hint:" comments, no helper that does the core work under another name (for example no `minmax_scale` util, no `run_episode` helper, no `mse_masked` helper). The human learns by writing these.

---

## 0. Target environment

- Machine: MacBook with Apple M4, 16 GB. PyTorch device order: `mps`, then `cuda`, then `cpu`.
- Python 3.10, 3.11 or 3.12, in a fresh virtual environment (`uv` or `venv`).
- Install LeRobot with the Push-T extra (`pip install "lerobot[pusht]"`, or `lerobot` + `gym-pusht`). Install `ffmpeg` if video decoding needs it (Homebrew).
- Also: `torch`, `torchvision`, `numpy`, `matplotlib`, `imageio[ffmpeg]`, `pyyaml`, `pytest`, `jupyter`.
- **LeRobot's API has changed across versions** (for example `lerobot.common.*` became `lerobot.*`, and newer versions use policy pre/post-processor pipelines via `make_pre_post_processors`). Check the installed version (`pip show lerobot`) and follow *that version's* docs and examples (`examples/` in the LeRobot repo, especially the pretrained-policy evaluation example). Do not guess import paths.
- After a working install, freeze exact versions into `requirements.lock.txt`.

## 1. Facts the code must respect

| Item | Value |
| --- | --- |
| Dataset | `lerobot/pusht` on the Hugging Face Hub. 206 episodes, 25,650 frames, 10 fps |
| Dataset keys | `observation.image` (96×96 RGB), `observation.state` (2: pusher x, y), `action` (2: target x, y), `episode_index`, `frame_index`, `timestamp`, `next.reward` |
| Environment | `gym_pusht/PushT-v0` via `gymnasium.make`, `obs_type="pixels_agent_pos"`, `render_mode="rgb_array"` |
| Env observation | dict: `"pixels"` uint8 (96, 96, 3), `"agent_pos"` float (2,) |
| Env action | float (2,), absolute target position, range [0, 512] |
| Reward | coverage of the target by the block, in [0, 1] |
| Success | maximum coverage over the episode ≥ 0.95 (the env also terminates at 0.95) |
| Episode cap | 300 steps |
| Pretrained reference | `lerobot/diffusion_pusht` (Diffusion Policy; inputs: image + agent position) |

Verify the dataset keys, image dtype/range/layout and stats structure against the installed LeRobot version, and adapt the plumbing. Report any difference.

## 2. Repository layout

```
pusht-bc/
  README.md
  pyproject.toml            # package "pusht_bc" under src/, editable install
  requirements.lock.txt
  pytest.ini                # registers the "core" marker
  configs/
    bc_v0.yaml
  src/pusht_bc/
    __init__.py
    config.py               # agent
    data.py                 # agent
    split.py                # CORE stub
    normalize.py            # CORE stub
    encoders.py             # agent
    model.py                # CORE stub
    losses.py               # CORE stub
    train.py                # CORE stubs + agent CLI
    evaluate.py             # CORE stubs + agent dataclasses, env, CI, CLI
    utils.py                # agent
    policies/
      __init__.py
      pretrained_dp.py      # agent
  scripts/
    check_env.py            # agent
    cache_features.py       # agent (runs once Normalizer is implemented)
  notebooks/
    00_explore_data.ipynb   # agent: loading cells only
    01_experiments.ipynb    # agent: orchestration cells only
  tests/
    fakes.py                # agent
    test_plumbing.py        # agent, unmarked, must pass
    test_split.py           # core
    test_normalize.py       # core
    test_model.py           # core
    test_losses.py          # core
    test_train.py           # core
    test_evaluate.py        # core
    test_chunking.py        # core
  runs/                     # gitignored outputs
  data/cache/               # gitignored feature cache
```

## 3. Agent-implemented modules (implement fully)

### `config.py`
- Dataclass `Config` loaded from YAML (`load_config(path) -> Config`), with fields shown in section 6. Unknown keys raise an error.

### `utils.py`
- `set_seed(seed: int)`: Python, NumPy, torch (and MPS/CUDA if present).
- `get_device() -> torch.device`: mps > cuda > cpu.
- `save_checkpoint(path, model, normalizer_stats: dict, config: Config, epoch: int, extra: dict | None = None)` and `load_checkpoint(path) -> dict`. Save the head weights, plus the encoder weights only when the encoder was trained (`freeze_encoder: false`), plus metadata. Never save frozen pretrained encoder weights.
- `write_video(frames: list[np.ndarray], path, fps: int = 10)` using imageio.
- `plot_losses(history: dict, path)`: train loss per step, val loss per epoch.
- `plot_coverage_over_time(rewards_by_label: dict[str, list[float]], path)`: coverage curves on one axis, 0.95 line marked.

### `data.py`
- `load_dataset(repo_id="lerobot/pusht", delta_timestamps=None, image_transforms=None) -> LeRobotDataset`.
- `get_episode_indices(dataset) -> list[int]`.
- `frames_for_episodes(dataset, episodes: list[int]) -> list[int]`: frame indices belonging to those episodes.
- `env_obs_to_batch(obs: dict) -> dict[str, torch.Tensor]`: converts an env observation to **the same format, dtype, layout and range as one dataset sample** (no batch dim): `"observation.image"` float32 (3, 96, 96) in [0, 1]; `"observation.state"` float32 (2,). No normalization here.
- `CachedFeatureDataset(cache_path, frame_indices, chunk_size=None)`: returns dicts with `"features"` (512,), `"observation.state"` (2,), `"action"` (2,), or with `chunk_size=K`: `"action"` (K, 2) and `"action_is_pad"` (K,) bool, padded within the episode exactly like LeRobot's `delta_timestamps` behavior (repeat last valid action, mark padded steps True). All values raw (unnormalized).
- `make_dataloaders(cfg, train_episodes, val_episodes) -> (train_loader, val_loader)`: cached path if `cfg.use_cached_features`, else image path through `LeRobotDataset` with `delta_timestamps` when `cfg.chunk_size` is set (`{"action": [i / fps for i in range(K)]}`). If `cfg.train_episode_fraction < 1`, subsample training episodes with `cfg.seed`. If `cfg.crop` is set, image path applies random crop (train) and center crop (val) of that size.
- `make_image_transform(crop: int | None, train: bool)`: the crop transform above, also used at evaluation (center crop).

### `encoders.py`
- `make_resnet18_encoder(pretrained: bool = True, frozen: bool = True) -> nn.Module`: torchvision ResNet-18 with ImageNet weights, classifier removed, global average pooling, output (B, 512) for input (B, 3, H, W) with H, W ≥ 64. When `frozen`: all parameters `requires_grad=False`, and wrap it so that `.train()` on the wrapper or any parent module keeps the encoder in eval mode (BatchNorm statistics must never update). Input is expected already ImageNet-normalized.

### `evaluate.py` (agent parts only)
- `make_env(max_steps: int = 300)`: gymnasium env as in section 1, with `max_episode_steps=max_steps`.
- Dataclasses `EpisodeResult(seed: int, success: bool, max_coverage: float, steps: int, rewards: list[float], frames: list[np.ndarray] | None)` and `EvalSummary(success_rate: float, ci_low: float, ci_high: float, avg_max_coverage: float, n_episodes: int, episodes: list[EpisodeResult])`.
- `binomial_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]`: Wilson interval.
- `load_policy_for_eval(checkpoint_path, device) -> tuple[nn.Module, dict]`: builds `BCPolicy` **with** a frozen encoder attached (`encoder=make_resnet18_encoder()`), loads the head weights saved by `save_checkpoint`, and the encoder weights too when the checkpoint has them (relies on the attribute convention in section 4: `self.encoder`, `self.head`). Returns the model and the normalizer stats. The CLI builds the `BCAgent` with `image_transform=make_image_transform(cfg.crop, train=False)` from the checkpoint's config.
- CLI `python -m pusht_bc.evaluate --checkpoint PATH | --pretrained-dp --seeds 0-49 [--record 0,1,2] [--out runs/NAME/eval]`: builds the agent (`BCAgent` or `PretrainedDiffusionAgent`), calls the human's `evaluate`, writes `summary.json`, videos for recorded seeds, and prints success rate with the interval. It must work unchanged once the human implements the core.

### `train.py` (agent parts only)
- Dataclass `TrainResult(history: dict, checkpoint_paths: list[str], best_val_loss: float)`.
- CLI `python -m pusht_bc.train --config configs/bc_v0.yaml [--name NAME]`: loads config, calls the human's `train(cfg)`, saves the loss plot.
- Factories the human's `train` will call: `build_model(cfg, device) -> BCPolicy` (cached path: `encoder=None`; image path: trainable or frozen encoder per `cfg.freeze_encoder`), `build_optimizer(cfg, model)` (AdamW over parameters with `requires_grad`), `build_normalizer(dataset_stats) -> Normalizer`. These only construct objects; they contain no training logic.

### `policies/pretrained_dp.py`
- `PretrainedDiffusionAgent(repo_id="lerobot/diffusion_pusht", device=None)` with `reset()` and `act(obs_batch: dict) -> np.ndarray` of shape (2,), float, environment units. `obs_batch` is exactly what `env_obs_to_batch` returns. Inside, do whatever the installed LeRobot version requires (stats/normalization, pre/post-processors, `select_action`, policy `reset()` for its action queue). This adapter must expose the same interface as the human's `BCAgent`, so the human's `rollout`/`evaluate` can run it unchanged.

### `scripts/check_env.py`
- Runs one episode with random actions for 50 steps, saves `runs/check_env/random.mp4`, prints observation keys, shapes, dtypes and the reward range. Smoke test for the install.

### `scripts/cache_features.py`
- For every dataset frame: image → `Normalizer.normalize_image` (the human's) → frozen ResNet-18 → (512,). Save `data/cache/features_resnet18.pt` with tensors `features` (N, 512), `state` (N, 2), `action` (N, 2), `episode_index` (N,), `frame_index` (N,), plus the dataset stats. Batched, on `get_device()`, with a progress bar. It is expected to fail with the core's `NotImplementedError` until the human finishes guide section 3; say so in its error message.

## 4. Core stubs (signatures + docstrings only; body raises NotImplementedError)

Guide section numbers in parentheses. Use exactly these names, signatures and attribute names.

### `split.py` (section 2)
```python
def split_episodes(episode_indices: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Return (train_episodes, val_episodes).
    Contract: deterministic for a given seed; disjoint; together cover all inputs;
    len(val) == max(1, round(len(episode_indices) * val_fraction)); both lists sorted, plain ints."""
```

### `normalize.py` (section 3)
```python
class Normalizer:
    """Min-max normalization of state and action to [-1, 1]; ImageNet normalization of images.
    Same object is used in training and evaluation."""
    @classmethod
    def from_stats(cls, stats: dict) -> "Normalizer":
        """Build from LeRobot-style stats: stats['observation.state']['min'|'max'], stats['action']['min'|'max']."""
    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        """(..., 2) raw -> (..., 2) in [-1, 1]. Works on any device."""
    def normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        """(..., 2) raw -> (..., 2) in [-1, 1]."""
    def unnormalize_action(self, a_norm: torch.Tensor) -> torch.Tensor:
        """Inverse of normalize_action."""
    def normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        """(..., 3, H, W) float in [0, 1] -> ImageNet-normalized, same shape."""
    def state_dict(self) -> dict:
        """Stats needed to rebuild this object (used by save_checkpoint)."""
```

### `model.py` (sections 4 and 10)
```python
class BCPolicy(nn.Module):
    """Deterministic BC policy: image features (+ pusher state) -> normalized action(s).
    Attribute convention (used by checkpoint loading): self.encoder (nn.Module or None), self.head (nn.Module)."""
    def __init__(self, encoder: nn.Module | None = None, feat_dim: int = 512, state_dim: int = 2,
                 hidden_dim: int = 256, action_dim: int = 2, chunk_size: int | None = None): ...
    def forward(self, obs_feat_or_image: torch.Tensor, state_norm: torch.Tensor) -> torch.Tensor:
        """obs_feat_or_image: (B, 512) features if self.encoder is None, else (B, 3, H, W) normalized images.
        state_norm: (B, 2). Returns (B, 2), or (B, K, 2) when chunk_size=K."""
```

### `losses.py` (section 5)
```python
def bc_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, kind: str = "mse") -> torch.Tensor:
    """pred, target: (B, 2) or (B, K, 2). mask: (B,) or (B, K) bool, True = real target.
    Mean over real elements only. kind in {"mse", "l1"}. Returns a scalar; 0 if no real elements."""
```

### `train.py` (section 6)
```python
def train_one_epoch(model, loader, optimizer, normalizer, device) -> float:
    """One pass over loader. Batches are raw dicts ("features" or "observation.image",
    "observation.state", "action", optional "action_is_pad"). Returns mean training loss."""

def compute_val_loss(model, loader, normalizer, device) -> float:
    """Mean loss over loader without updating parameters."""

def train(cfg) -> TrainResult:
    """Build data/model/optimizer via the provided factories, run cfg.epochs, log history
    (train loss per step, val loss per epoch), save a checkpoint every epoch. Returns TrainResult."""
```

### `evaluate.py` (sections 7 and 10)
```python
class BCAgent:
    """Wraps a trained BCPolicy for closed-loop use."""
    def __init__(self, model, normalizer, device, image_transform=None, n_execute: int | None = None): ...
    def reset(self) -> None:
        """Clear any per-episode state."""
    def act(self, obs_batch: dict) -> np.ndarray:
        """obs_batch: output of env_obs_to_batch (no batch dim). Returns (2,) float action in env units, clipped to [0, 512]."""

def rollout(agent, env, seed: int, max_steps: int = 300, record: bool = False) -> EpisodeResult:
    """Run one seeded episode. Success = max coverage >= 0.95."""

def evaluate(agent, env, seeds: list[int], record_seeds: tuple[int, ...] = ()) -> EvalSummary:
    """Run rollout for each seed; aggregate success rate (+ binomial_ci) and average max coverage."""
```

## 5. Tests

`pytest.ini` registers marker `core`. Core tests carry `@pytest.mark.core`. Before the human starts: `pytest -m "not core"` passes and `pytest -m core` fails only with `NotImplementedError`. Tests check **behavior**, never implementation details. Keep test code free of hints (no comments explaining how to implement).

### `tests/fakes.py`
- `FakePushTEnv(rewards: list[float], terminate_at_success: bool = True)`: gymnasium-like `reset(seed=None) -> (obs, info)` and `step(action) -> (obs, reward, terminated, truncated, info)` that plays back scripted rewards; observations match section 1 (`"pixels"` uint8 (96, 96, 3), `"agent_pos"` (2,)); records every action received; `terminated=True` when reward ≥ 0.95 if `terminate_at_success`.
- `ConstantModel(output)`: an `nn.Module` with the `BCPolicy` call signature returning a fixed normalized output, with a call counter.
- `synthetic_stats()`: a LeRobot-style stats dict with known min/max.

### `test_plumbing.py` (unmarked, must pass now)
- Dataset loads; keys and shapes as in section 1; 206 episodes.
- `env_obs_to_batch` on a real env observation matches a dataset sample's dtype, shape and value range for image and state.
- `make_resnet18_encoder` output (B, 512); frozen params; BatchNorm running stats unchanged after `.train()` + a forward pass.
- `CachedFeatureDataset` with `chunk_size=8` on a tiny synthetic cache: shapes, padding at episode end, mask correctness, no crossing episode boundaries.
- `binomial_ci(32, 50)` within sensible bounds and containing 0.64.
- `PretrainedDiffusionAgent.act` on one real env observation returns shape (2,) within [0, 512] (skip with a clear reason if the download is unavailable).

### Core tests
- `test_split.py`: deterministic per seed; different seeds give different splits (on ≥ 50 episodes); disjoint; full coverage; validation size rule incl. minimum 1; sorted int lists.
- `test_normalize.py`: min → −1 and max → +1; round trip error < 1e-5 for shapes (B, 2) and (B, K, 2); zero-range dimension gives finite values; image output shape preserved and a constant image equal to the ImageNet mean maps to ≈ 0; works with tensors on the test's device.
- `test_model.py`: cached path (B, 512) + (B, 2) → (B, 2); image path with the real frozen encoder (B, 3, 96, 96) → (B, 2); `chunk_size=8` → (B, 8, 2); encoder params frozen, head params trainable; default cached model has exactly 198,146 trainable parameters; attributes `encoder` and `head` exist.
- `test_losses.py`: 0 for equal inputs; equals `F.mse_loss` without mask; masked-out entries do not change the loss; `kind="l1"` equals `F.l1_loss`; all-False mask returns 0 without NaN; works on (B, K, 2) with (B, K) mask.
- `test_train.py`: on 32 synthetic samples (random 512-d features, states, actions; synthetic stats), repeated `train_one_epoch` calls (lr 1e-3, batch 32) reach normalized loss < 1e-2 within 500 steps; `compute_val_loss` returns a float and leaves parameters unchanged.
- `test_evaluate.py`: `rollout` with `FakePushTEnv([0.2, 0.5, 0.96, 0.4])` → success True, max_coverage 0.96, stops at the terminating step; with rewards all < 0.95 and no termination → success False and `steps == max_steps`; `evaluate` over 4 seeds aggregates success rate and average max coverage correctly and fills the interval; `BCAgent` with a `ConstantModel` returning values far outside [−1, 1] produces actions clipped to [0, 512], shape (2,), dtype float.
- `test_chunking.py`: `BCAgent` with a chunked `ConstantModel` (K = 8) and `n_execute=4` calls the model exactly once every 4 steps over 12 steps; `reset()` forces a new prediction on the next step.

## 6. Config `configs/bc_v0.yaml`

```yaml
name: bc_v0
seed: 0
dataset_repo_id: lerobot/pusht
val_fraction: 0.1
train_episode_fraction: 1.0
use_cached_features: true
cache_path: data/cache/features_resnet18.pt
freeze_encoder: true
crop: null
hidden_dim: 256
chunk_size: null
n_execute: null
loss: mse
batch_size: 256
lr: 1.0e-3
weight_decay: 1.0e-4
epochs: 50
eval:
  dev_seeds: [0, 49]        # inclusive range
  final_seeds: [1000, 1199] # inclusive range
  max_steps: 300
  record_seeds: [0, 1, 2]
output_dir: runs
```

## 7. Notebooks (no core logic in either)

- `00_explore_data.ipynb`: cells that load the dataset and stats, print keys/shapes/fps/episode count, and a helper to display a frame. Then **empty** markdown cells with these headings for the human to fill: "Episode lengths", "One episode: state vs action path", "First, middle and last frames", "Similar starts: same strategy?". Do not answer them.
- `01_experiments.ipynb`: cells that (1) run `scripts/cache_features.py`, (2) train a config via `train(load_config(...))`, (3) evaluate a checkpoint or the pretrained DP on dev seeds and show summary + videos inline, (4) plot loss curves and coverage-over-time. One section per experiment ID (E1–E7) with the config path filled for E1 only.

## 8. README.md

Short: install, `python scripts/check_env.py`, `pytest -m "not core"`, then "implement the core following the guide; `pytest -m core` tracks progress", the train/eval commands, and where outputs go.

## 9. Acceptance (report back when done)

1. `python scripts/check_env.py` saves a video and prints shapes.
2. `pytest -m "not core"` passes; `pytest -m core` fails only with `NotImplementedError`.
3. `grep -rn "TODO(core)" src/` lists exactly: `split_episodes`, `Normalizer` methods, `BCPolicy.__init__`/`forward`, `bc_loss`, `train_one_epoch`, `compute_val_loss`, `train`, `BCAgent` methods, `rollout`, `evaluate`.
4. Report: installed versions (Python, torch, lerobot, gym-pusht), MPS availability, any place the installed LeRobot API differed from this spec and what you did, and anything you could not verify.
