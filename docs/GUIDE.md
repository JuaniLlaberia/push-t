# Step 5 Guide: Behavior Cloning on Push-T

Nov 23, 2026 · @Juan Ignacio Llaberia

You build a behavior-cloning policy for Push-T and measure its success rate in simulation. You write the core (split, normalization, model, loss, training, evaluation); the coding agent writes the plumbing from SCAFFOLD\_SPEC.md. Start after finals, on Dec 4.

## How to use this guide

Every section has the same parts, in this order:

1. **Theory anchor:** the idea from L13, L17, CS 285 L2 or the tutorial that this code implements.
2. **Decision:** what we choose, the alternatives, and why.
3. **Spec:** inputs, outputs and shapes. Never the code.
4. **Your turn:** what you write, and in which file.
5. **Checkpoint:** the test or number that tells you it works. Don't move on until it passes.
6. **Predict, then run:** where it applies, write your prediction under the section before you run anything.

**Rules**

- You write every function marked `TODO(core)`. The agent builds everything else from `SCAFFOLD_SPEC.md` before you start, and its own tests pass on day one.
- Hints live in the Hints tab, three levels per section. Give each part a real try first, about 20–30 minutes.
- Predictions stay in this doc as a record, right or wrong. Wrong predictions are the most useful part.
- Compute: your MacBook M4 is enough. Use Kaggle only if end-to-end training in section 8 is too slow.
- Time cap: about two weeks part-time. If you're over, cut experiments from section 8, never the evaluation check in section 7.

**Done when**

- [ ] Your evaluation loop scores the pretrained Diffusion Policy at roughly 60–65% success.
- [ ] Your BC policy is trained and evaluated on the same 50 fixed seeds: success rate and average max coverage.
- [ ] Three videos saved: one success and two failures, each failure explained.
- [ ] `REPORT.md` written with the results table and the failure analysis.

## At a glance

&#91;embedded content: step 5 pipeline · training and evaluation\]

Training never touches the simulator. Evaluation reuses the same Normalizer and the trained weights in a closed loop, which is where compounding errors show up.

## 1 · Task and data

Push-T is a 2D simulation. A round pusher, which your policy controls, must push a T-shaped block onto a fixed target T. An episode succeeds when the block covers at least 95% of the target.

**Theory anchor.** Behavior cloning learns πθ(aₜ | oₜ) from (observation, action) pairs (CS 285 L2). Here oₜ is a top-down image plus the pusher's position, and aₜ is where the pusher should go next. The pairs come in trajectories, so consecutive rows are not independent (tutorial, section 4).

| Item | Value |
| --- | --- |
| Dataset | `lerobot/pusht`: 206 human episodes, 25,650 frames |
| Control rate | 10 Hz, one action every 0.1 s |
| `observation.image` | 96 × 96 RGB, top-down |
| `observation.state` | pusher (x, y) |
| `action` | target (x, y) for the pusher, range 0–512 |
| Environment | `gym_pusht/PushT-v0`, `obs_type="pixels_agent_pos"`, at most 300 steps |
| Success | block covers ≥ 95% of the target |

**Decision.** Keep the action as the dataset defines it: an absolute target position. The environment consumes absolute targets, so predicting exactly what it consumes avoids a conversion step that can drift. The alternative is a delta (target minus current position), which is sometimes easier to learn. Try it in section 8 only if you have time.

**Your turn.** In `notebooks/00_explore_data.ipynb` the agent provides the loading cells; you write the analysis and answer here:

- [x] Episode lengths: minimum, mean and maximum number of frames.
- [x] One episode plotted on a 512 × 512 canvas: the pusher's path (state) and the action path together. How far ahead of the state is the action?
- [x] Frames at the start, middle and end of three episodes. What do all the first frames share, and what differs?
- [x] Two episodes whose blocks start in similar positions. Do the demonstrators push from the same side?

**Checkpoint.** You can say in one sentence what one row of the dataset is, and why consecutive rows are not independent.

> Answer: Each row represents one frame in the dataset, meaning that each row is not one episode but one is divided into multiple frames. Because of that each row is not independent, example episode 0 covers 160 rows.

**Predict, then run.** Before the last question: will similar starts lead to the same strategy? If not, what will a policy trained with MSE do in that situation?

> Answer: Different strategies may simply follow from where the pusher was, which the policy can see. That would be a sensible response to a different observation, not multimodality.

## 2 · Train/validation split by episode

Hold out whole episodes, never individual frames.

**Theory anchor.** Supervised learning assumes independent samples. At 10 Hz, consecutive frames of one episode are near-duplicates. If frame 41 is in training and frame 42 in validation, the validation loss measures memory, not generalization.

**Decision.** Hold out about 10% of episodes (about 20 of 206), chosen at random with a fixed seed. Alternatives:

- A random split of frames: rejected, because it leaks (above).
- The last 20 episodes in recording order: workable, but demonstrators drift over a recording session, so the held-out set may differ systematically.

Validation loss is only a sanity check here. The number that counts is the success rate in section 7, and section 8 tests how well the two agree.

**Spec.** In `src/pusht_bc/split.py`:

- `split_episodes(episode_indices, val_fraction, seed) -> (train_episodes, val_episodes)`
- Deterministic for a given seed. The two lists are disjoint and together cover every episode. The validation list has `round(n × val_fraction)` episodes, at least 1.
- The agent's `frames_for_episodes(dataset, episodes)` turns episodes into frame indices. You don't write that part.

**Your turn.** Implement `split_episodes`.

**Checkpoint.** `pytest -m core tests/test_split.py` passes.

**Predict, then run (optional, 10 minutes).** Once section 6 works, train once with a random frame split. Before you do, write down whether its validation loss will be lower or higher than with the episode split, and why.

## 3 · Normalization

Normalization is the most common source of silent bugs in robot learning. Get it right once, and use the same object at training and at evaluation.

**Theory anchor.** Positions run from 0 to 512 while image pixels run from 0 to 1. Regressing raw pixel coordinates gives large losses and gradients, and the output layer has to produce large numbers. Scaling inputs and targets to similar ranges makes optimization well-conditioned. It's the same reason you normalized activations in CS336.

**Decision.**

- **State and action: min-max to \[−1, 1\]**, using the dataset's statistics (`dataset.meta.stats`). The range is known and bounded, outputs can be clipped to \[−1, 1\], and it matches LeRobot's own default. The alternative, mean and standard deviation, works too, but has no fixed bounds.
- **Images: ImageNet mean and standard deviation** on \[0, 1\] floats, channels first. The encoder in section 4 was pretrained with exactly that input, so anything else shifts what it sees.
- **Statistics come from the dataset as a whole**, not from your training split. That avoids a second set of numbers to keep consistent. The leakage from 20 held-out episodes into min and max is negligible.

**Spec.** In `src/pusht_bc/normalize.py`, a `Normalizer` class:

- `Normalizer.from_stats(stats)` builds it from LeRobot's stats dict.
- `normalize_state(x)`, `normalize_action(a)` and `unnormalize_action(a_norm)` take tensors with any leading batch shape and a last dimension of 2.
- `normalize_image(img)` takes float images, channels first, in \[0, 1\], and returns ImageNet-normalized ones.
- Guard against division by zero when a dimension's min equals its max. Also add a state\_dict() method that returns the stats, so checkpoints can rebuild the object.
- The agent's `env_obs_to_batch` turns an environment observation (uint8, channels last) into the dataset's format. Your `Normalizer` then treats training and evaluation inputs identically.

**Your turn.** Implement `Normalizer`.

**Checkpoint.** `pytest -m core tests/test_normalize.py` passes: a normalize/unnormalize round trip within 1e−5, min maps to −1 and max to +1, and a zero-range dimension stays finite. The agent's plumbing test already checks that an environment frame and a dataset frame match in shape, dtype and range.

**Predict, then run.** If evaluation forgets `unnormalize_action`, where on the 512 × 512 table will the pusher go? Check your answer once section 7 works.

## 4 · The policy network

Version 0 is a frozen pretrained image encoder plus a small MLP you write. It's deliberately simple, so that every failure can be traced to behavior cloning itself.

**Theory anchor.** A deterministic network a = fθ(o) is the mean of a Gaussian policy πθ(a | o) = N(fθ(o), σ²I). That single Gaussian is what makes MSE the right loss in section 5. It's also unimodal by construction, which is the multimodality limitation from CS 285 L2. You're building the baseline that ACT and Diffusion Policy were designed to beat.

**Decision.**

- **Encoder: ResNet-18, ImageNet weights, frozen,** global-average-pooled to 512 features. Frozen means the features can be computed once and cached (the agent's script does this), training takes minutes, and any bug is in your head network, not the encoder. The alternative is training the encoder end to end with random-crop augmentation, as Diffusion Policy does. That's experiment E4 in section 8.
- **Pusher position concatenated to the image features.** At 96 × 96 the image shows the pusher only approximately; two exact numbers cost nothing.
- **Head: MLP 514 → 256 → 256 → 2** with ReLU and no output activation. A tanh would bound outputs to \[−1, 1\] but saturates near the edges, where the table's borders are. Clip at inference instead (section 7).
- **The encoder stays in eval mode** while frozen, so BatchNorm uses its stored statistics. The agent's encoder factory handles this; know why it matters.

**Spec.** In `src/pusht_bc/model.py`, `BCPolicy(nn.Module)`:

- `__init__(encoder=None, feat_dim=512, state_dim=2, hidden_dim=256, action_dim=2)`. With `encoder=None`, the model expects cached features.
- `forward(obs_feat_or_image, state_norm) -> action_norm` of shape (B, 2). The first input is either an image batch (B, 3, 96, 96) or cached features (B, 512).
- The agent provides `make_resnet18_encoder(pretrained=True, frozen=True)`, which returns (B, 512) features.

**Your turn.** Implement `BCPolicy`.

**Checkpoint.** `pytest -m core tests/test_model.py` passes: output shape (B, 2) on both input paths, encoder parameters frozen, head parameters trainable.

**Predict, then run.** Work out the head's trainable parameter count by hand before the test prints it.

## 5 · The behavior cloning loss

MSE on normalized actions, which is maximum likelihood under the Gaussian policy from section 4.

**Theory anchor.** With πθ(a | o) = N(fθ(o), σ²I) and fixed σ, the negative log-likelihood of one expert action is:

```latex
-\log \pi_\theta(a \mid o) = \frac{1}{2\sigma^2}\,\lVert a - f_\theta(o) \rVert^2 + \text{const}
```

Minimizing MSE over the dataset is therefore behavior cloning as maximum likelihood (your notes from the MLE summary and CS 285 L2).

**Decision.**

- **MSE, averaged over the batch and the 2 action dimensions.**
- Alternative: **L1**, which ACT uses. It's more robust to sloppy demonstrations and predicts a median rather than a mean.
- Alternative: **NLL with a learned σ**. It changes nothing about the predicted mean, so it doesn't help here.
- Support an optional **mask** now. Section 10 needs it to ignore padded actions at the end of an episode.

**Spec.** In `src/pusht_bc/losses.py`:

- `bc_loss(pred, target, mask=None, kind="mse") -> scalar tensor`
- `pred` and `target` have shape (B, 2), or (B, K, 2) in section 10. `mask` is (B,) or (B, K), True where the target is real. The loss averages over real elements only.
- `kind` is `"mse"` or `"l1"`.

**Your turn.** Implement `bc_loss`.

**Checkpoint.** `pytest -m core tests/test_losses.py` passes: zero for equal inputs, equal to PyTorch's MSE without a mask, masked entries ignored, L1 option correct.

**Predict, then run.** In one state, half the demonstrations go to −0.5 and half to +0.5. Which single output minimizes MSE? Which minimizes L1? What does that mean for the pusher?

## 6 · The training loop

The same structure as your TinyForge loop, on a much smaller problem. It runs offline, with no simulator involved.

**Theory anchor.** Behavior cloning is plain supervised learning on a fixed dataset. That's its main practical advantage over RL (tutorial, section 4): no rewards, no exploration, no resets. The simulator only appears at evaluation.

**Decision.**

| Setting | v0 (cached features) | E4 (end-to-end encoder) |
| --- | --- | --- |
| Optimizer | AdamW, weight decay 1e−4 | AdamW, weight decay 1e−4 |
| Learning rate | 1e−3, constant | 1e−4 |
| Batch size | 256 | 64 |
| Length | 50 epochs to start; adjust from the curves | from the curves |
| Device | MPS on the M4 | MPS, or a Kaggle GPU if slow |

These are starting points, not tuned values. Change one at a time and write down why.

- **Normalize inside the loop**, not in the dataset. The batch arrives raw, so you can see exactly where normalization happens.
- **Log** train loss every step and validation loss every epoch. **Save** a checkpoint every epoch with the agent's `save_checkpoint`. Section 8 evaluates several of them.
- **Seed** everything with the agent's `set_seed` so runs are repeatable.

**Spec.** In `src/pusht_bc/train.py`:

- `train_one_epoch(model, loader, optimizer, normalizer, device) -> mean_loss`
- `compute_val_loss(model, loader, normalizer, device) -> mean_loss`, under `torch.no_grad()`, with the model in eval mode.
- `train(cfg) -> TrainResult`: builds data, model and optimizer from the agent's factories, runs the epochs, logs and saves. The command-line entry and config parsing are the agent's.
- A batch is a dict with `"features"` (or `"observation.image"`), `"observation.state"` and `"action"`, all raw.

**Your turn.** Implement the three functions.

**Checkpoint.**

- `pytest -m core tests/test_train.py` passes. The key test overfits 32 samples to a normalized loss below 1e−2 within 500 steps. If it can't, there's a bug; more data won't fix it.
- A full v0 run finishes, and `notebooks/01_experiments.ipynb` plots both curves.

**Predict, then run.** Before the full run: will validation loss keep falling with training loss, or flatten first? At roughly what epoch?

## 7 · The evaluation loop

Verify your evaluation code on a policy whose score you already know, before you trust it with your own.

**Theory anchor.** Compounding errors only appear in closed loop, when the policy's own actions pick its next observation (CS 285 L2). Offline validation loss can't see them. A rollout in the simulator is the only honest test of a policy.

**Decision.**

- **Success** = the block's maximum coverage of the target during the episode reaches 95%, the same definition LeRobot uses. Also report **average maximum coverage**, which shows partial progress that success alone hides.
- **Fixed seeds, the same for every policy:** seeds 0–49 while developing, and 200 fresh seeds (1000–1199) for the final report. The same seeds mean the same starting positions, so comparisons are paired.
- **Report a 95% confidence interval.** With 50 episodes and a true rate near 64%, the standard error is about 7 points, so the interval is roughly ±13. Fifty episodes can't separate 60% from 70%; 200 can roughly halve that.
- **At most 300 steps** per episode. **Clip** unnormalized actions to \[0, 512\].
- **Sanity check first:** run the pretrained `lerobot/diffusion_pusht` through *your* loop, using the agent's adapter, which exposes the same interface as your policy. Its model card reports 63.8–65.4% success over 500 episodes, and a 2026 paper measured 62.5% over 200 seeds. On 50 seeds, expect roughly 50–75%. Far outside that, the bug is in your evaluation (seeds, the success definition or the observation format), not in any policy.

**Spec.** In `src/pusht_bc/evaluate.py`:

- `BCAgent(model, normalizer, device, image_transform=None, n_execute=None)` wraps your `Normalizer` and a `BCPolicy` with the encoder attached; the agent's `load_policy_for_eval` builds that model from a checkpoint. `reset()` and `act(obs) -> np.ndarray` of shape (2,), in environment units and clipped. `obs` is what the agent's `env_obs_to_batch` returns.
- `rollout(agent, env, seed, max_steps=300, record=False) -> EpisodeResult` with `success`, `max_coverage`, `steps` and, if recording, `frames`. In gym-pusht the reward at each step is the current coverage.
- `evaluate(agent, env, seeds, record_seeds=()) -> EvalSummary` with success rate, average max coverage, the interval from the agent's `binomial_ci`, and per-episode results.
- Creating the environment and writing videos are the agent's.

**Your turn.** Implement `BCAgent`, `rollout` and `evaluate`.

**Checkpoint.**

- [ ] `pytest -m core tests/test_evaluate.py` passes, using the agent's fake environment with scripted rewards.
- [ ] Pretrained Diffusion Policy on seeds 0–49 through your loop: within roughly 50–75%.
- [ ] Your v0 policy on seeds 0–49, with videos for three seeds.

**Predict, then run.** Before your v0's first evaluation, write down its success rate and average max coverage.

## 8 · Experiments

E1 and E2 are required, E3 is recommended, the rest are optional. Change one thing per experiment, evaluate on seeds 0–49, and fill in your prediction before you run.

**Theory anchor.** Each experiment tests one claim from CS 285 L2 or the tutorial on your own policy: offline loss vs. closed-loop success, how data quantity affects coverage of the states the policy visits, and what the encoder and loss choices change.

| ID | Question | Change from v0 | Your prediction | Result |
| --- | --- | --- | --- | --- |
| E1 | How good is plain behavior cloning? | none: v0 is the baseline |  |  |
| E2 | Does lower validation loss mean higher success? | evaluate checkpoints from epochs 5, 10, 25 and 50 |  |  |
| E3 | How much does more data help? | train on 25%, 50% and 100% of the training episodes |  |  |
| E4 | Does a trainable encoder help? | train ResNet-18 end to end, random 84 × 84 crops in training, center crop at evaluation |  |  |
| E5 | Does L1 change behavior? | `kind="l1"` in `bc_loss` |  |  |
| E6 | Are deltas easier than absolute targets? | predict target minus current position |  |  |

**Your turn.** Each experiment is a config file in `configs/` plus a cell in `notebooks/01_experiments.ipynb` that trains, evaluates and logs. The agent provides the v0 config; copy and edit it for each experiment.

**Checkpoint.** E1 and E2 filled in, with a plot of validation loss against success rate for E2.

**Predict, then run.** The table's prediction column. For E2, write the shape you expect before plotting it.

## 9 · Failure analysis

Watch about 10 failed episodes, label each with one failure mode, and count. These counts explain your success rate, and they're what the rest of 4b is built to fix.

**Theory anchor.** CS 285 L2 gave two reasons a policy fails to match the expert (multimodality and missing history) plus compounding errors in closed loop. Your videos show which ones dominate.

| Failure mode | What you see | Likely cause (L2) | What fixes it later in 4b |
| --- | --- | --- | --- |
| Hesitation | the pusher wavers between sides of the block, or stops between them | averaging of multimodal demonstrations | Diffusion Policy (step 7), ACT's latent z (step 6) |
| Drift | a good start, then one bad push and no recovery | compounding errors, states missing from the data | action chunking (step 6), more varied data |
| Stall | the pusher stops moving mid-task | a single frame can't show the task's phase | observation history, action chunks |
| Near miss | coverage reaches 85–94% but the final alignment fails | few demonstrations of fine corrections | more data, higher-resolution images |
| Never reaches the block | aimless motion from the first step | usually a bug, not BC | recheck sections 3 and 7 |

**Your turn.**

- [ ] Record videos of 10 failed seeds with the agent's video writer, watch each, and label it.
- [ ] Plot coverage over time for one success and one failure (the agent's plotting helper) and describe where they diverge.
- [ ] Count the labels and add one line on what the most common mode says about behavior cloning.

**Checkpoint.** Counts per failure mode, plus two failure videos you could explain to someone in one sentence each.

**Predict, then run.** Before you watch: which mode will be most common?

## 10 · Optional: action chunking

Predict the next 8 actions at once and execute 4 before predicting again. It's the smallest step from your BC policy toward ACT.

**Theory anchor.** A policy that decides every step has 300 chances per episode to make a small error that compounds. Predicting a chunk means fewer decisions, and within a chunk the policy commits to one strategy instead of re-averaging every frame. In ACT's ablation, as reported in the Robot Learning tutorial, chunking raised success from 1% to 44%.

**Decision.**

- **Chunk size K = 8**, which is 0.8 s at 10 Hz. **Execute n = 4** actions, then predict again.
- **Data:** set LeRobotDataset's `delta_timestamps` so `"action"` covers offsets 0.0 to 0.7 s. Each sample then has actions of shape (8, 2) plus an `"action_is_pad"` mask for chunks that run past the episode's end. The agent's config option `chunk_size` sets this up.
- **No temporal ensembling yet.** That's ACT's refinement; you'll read about it in step 6.

**Spec.**

- `BCPolicy` gets `chunk_size`: the head outputs K × 2 values, reshaped to (B, K, 2).
- `bc_loss` with the mask (already supported from section 5).
- `BCAgent` keeps a queue of actions: it predicts a chunk when the queue is empty, takes the first n, and clears the queue on `reset()`.

**Your turn.** Make those three changes and run it as experiment E7.

**Checkpoint.** `pytest -m core tests/test_chunking.py` passes, and E7 is evaluated on seeds 0–49 next to v0.

**Predict, then run.** Executing n = 1, 4 or 8 actions per chunk: which works best, and why? Try all three if time allows.

## 11 · Write-up

`REPORT.md` becomes the first row of the 4b table. Steps 8 and 12 add the ACT, Diffusion Policy and SmolVLA rows to the same file, evaluated on the same 200 seeds.

**Your turn.** `REPORT.md` in the repo, in this order:

1. **Results table:** one row per policy (pretrained Diffusion Policy, your v0, your best variant) with seeds, success rate and its 95% interval, and average max coverage, all on seeds 1000–1199.
2. **E2 plot:** validation loss against success, with one sentence on what it shows.
3. **Failure counts** from section 9, and the two explained failure videos as GIFs.
4. **What you'd change**, in three bullets at most.

**Checkpoint: explain it cold.** Without notes, answer aloud:

- [ ] Why is MSE maximum likelihood here, and what does that assume about the expert?
- [ ] Why didn't validation loss predict success in E2 (or, if it did, why that isn't guaranteed)?
- [ ] Why does behavior cloning hesitate on Push-T, and which 4b method addresses it?

If any answer stalls, reread that section of this guide before you mark step 5 done.

## Sources

- [lerobot/pusht dataset](https://huggingface.co/datasets/lerobot/pusht): features, 206 episodes, 25,650 frames, 10 Hz.
- [gym-pusht](https://github.com/huggingface/gym-pusht): action range, observation types, coverage reward, 95% success threshold.
- [lerobot/diffusion\_pusht model card](https://huggingface.co/lerobot/diffusion_pusht): success definition and its 500-episode results.
- [Revisiting the Push-T task (2026)](https://arxiv.org/pdf/2608.18227): 62.5% for the LeRobot checkpoint over 200 seeds.
- [Robot Learning: A Tutorial](https://arxiv.org/html/2510.12403): behavior cloning section, ACT chunking ablation.
- [LeRobot dataset docs](https://huggingface.co/docs/lerobot/en/index): `LeRobotDataset`, `delta_timestamps`, dataset stats.
