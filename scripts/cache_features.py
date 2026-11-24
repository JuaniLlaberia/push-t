from __future__ import annotations

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pusht_bc.config import resolve_path
from pusht_bc.data import IMAGE_KEY, STATE_KEY, ACTION_KEY, load_dataset
from pusht_bc.encoders import make_resnet18_encoder
from pusht_bc.normalize import Normalizer
from pusht_bc.utils import get_device

def main() -> None:
    """
    Precompute frozen ResNet-18 features for every frame of lerobot/pusht.

    Usage: python scripts/cache_features.py [--batch-size 128] [--out data/cache/features_resnet18.pt]
    Pipeline per frame: image -> Normalizer.normalize_image -> frozen
    ResNet-18 -> (512,). Saves features (N, 512) float32, state (N, 2), action (N, 2),
    episode_index (N,), frame_index (N,), index (N,), stats (dataset stats) and meta.
    Row i = dataset frame i.
    """
    ap = argparse.ArgumentParser(description="Cache frozen ResNet-18 features of every lerobot/pusht frame.")
    ap.add_argument("--repo-id", default="lerobot/pusht")
    ap.add_argument("--out", default="data/cache/features_resnet18.pt")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 4)
    args = ap.parse_args()

    dataset = load_dataset(args.repo_id)
    stats = dataset.meta.stats
    normalizer = Normalizer.from_stats(stats)

    device = get_device()
    encoder = make_resnet18_encoder(pretrained=True, frozen=True).to(device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    feats, state, action, ep_idx, fr_idx, idx = [], [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"encoding on {device}", unit="batch"):
            img = batch[IMAGE_KEY].to(device, non_blocking=True)
            f = encoder(normalizer.normalize_image(img))
            feats.append(f.float().cpu())
            state.append(batch[STATE_KEY].float())
            action.append(batch[ACTION_KEY].float())
            ep_idx.append(batch["episode_index"].long().reshape(-1))
            fr_idx.append(batch["frame_index"].long().reshape(-1))
            idx.append(batch["index"].long().reshape(-1))

    index = torch.cat(idx)
    if not torch.equal(index, torch.arange(len(dataset))):
        raise RuntimeError("cache rows are not in dataset order")
    cache = {
        "features": torch.cat(feats),
        "state": torch.cat(state),
        "action": torch.cat(action),
        "episode_index": torch.cat(ep_idx),
        "frame_index": torch.cat(fr_idx),
        "index": index,
        "stats": stats,
        "meta": {"repo_id": args.repo_id, "encoder": "resnet18_imagenet_frozen", "image_size": (96, 96), "fps": dataset.fps},
    }
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out)
    print(f"saved {out} | features {tuple(cache['features'].shape)} | {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
