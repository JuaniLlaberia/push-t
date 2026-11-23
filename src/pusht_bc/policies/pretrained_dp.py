from __future__ import annotations

import logging

import numpy as np
import torch

from ..utils import get_device

log = logging.getLogger(__name__)

ACTION_LOW, ACTION_HIGH = 0.0, 512.0

class PretrainedDiffusionAgent:
    """
    LeRobot's pretrained Diffusion Policy behind the same interface as BCAgent.

    Written for lerobot 0.7.0. The Hub checkpoint lerobot/diffusion_pusht predates
    LeRobot's processor pipelines: it has no policy_preprocessor.json /
    policy_postprocessor.json and keeps its normalization stats as buffers inside
    model.safetensors. When the Hub files are missing, the pipelines are rebuilt from
    those buffers with LeRobot's own migration helper (extract_normalization_stats).
    The policy keeps its own observation history (n_obs_steps=2) and action queue
    (n_action_steps=8); reset() clears both, so call it at the start of every episode.

    Args:
        repo_id: pretrained policy on the Hub.
        device: target device (None = get_device()).
    """

    def __init__(self, repo_id: str = "lerobot/diffusion_pusht", device=None):
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        self.repo_id = repo_id
        self.device = torch.device(device) if device is not None else get_device()

        policy = DiffusionPolicy.from_pretrained(repo_id)
        policy.config.device = str(self.device)
        policy.to(self.device)
        policy.eval()
        self.policy = policy

        try:
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy.config,
                pretrained_path=repo_id,
                preprocessor_overrides={"device_processor": {"device": str(self.device)}},
            )
            self.processor_source = "hub"
        except Exception as e:  # ProcessorMigrationError or missing files on the Hub
            log.info("no processor files for %s (%s); rebuilding from checkpoint buffers", repo_id, type(e).__name__)
            stats = self._stats_from_checkpoint(repo_id)
            self.preprocessor, self.postprocessor = make_pre_post_processors(policy.config, dataset_stats=stats)
            self.processor_source = "checkpoint_buffers"
        self.reset()

    @staticmethod
    def _stats_from_checkpoint(repo_id: str) -> dict:
        from huggingface_hub import hf_hub_download
        from lerobot.processor.migrate_policy_normalization import extract_normalization_stats
        from safetensors.torch import load_file

        state_dict = load_file(hf_hub_download(repo_id, "model.safetensors"))
        stats = extract_normalization_stats(state_dict)
        for key in ("observation.image", "observation.state", "action"):
            if key not in stats:
                raise RuntimeError(f"{repo_id}: no normalization stats for {key!r} in the checkpoint")
        return stats

    def reset(self) -> None:
        """
        Clear the policy's observation history and action queue.
        """
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    @torch.inference_mode()
    def act(self, obs_batch: dict) -> np.ndarray:
        """
        Choose the next action.

        Args:
            obs_batch: output of env_obs_to_batch (no batch dim, unnormalized).
        Returns:
            (2,) float action in env units, clipped to [0, 512].
        """
        batch = {
            "observation.image": obs_batch["observation.image"].unsqueeze(0),
            "observation.state": obs_batch["observation.state"].unsqueeze(0),
            "task": [""],
        }
        batch = self.preprocessor(batch)
        action = self.policy.select_action(batch)
        action = self.postprocessor(action)
        a = action.reshape(-1, 2)[0].detach().to("cpu", torch.float32).numpy()
        return np.clip(a, ACTION_LOW, ACTION_HIGH)
