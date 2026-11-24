import argparse
import json

from pusht_bc.config import load_config, resolve_path
from pusht_bc.evaluate import build_agent_from_checkpoint, format_summary, run_and_save
from pusht_bc.train import save_train_outputs, train
from pusht_bc.utils import checkpoint_path, get_device, run_dir

CONFIGS = ["configs/e3_frac25.yaml", "configs/e3_frac50.yaml"]

def main() -> None:
    """
    Experiment E3 (data quantity): train v0 on 25% and 50% of the training episodes, evaluate the
    last checkpoint of each on the dev seeds, and print a table next to v0 (100%).

    Usage: python scripts/run_e3.py [--skip-train]
    Outputs: runs/e3_frac25/, runs/e3_frac50/ (checkpoints, history.json, eval/epoch_050/)
    and runs/e3_summary.json.
    """
    ap = argparse.ArgumentParser(description="Run experiment E3 (data quantity).")
    ap.add_argument("--skip-train", action="store_true", help="only evaluate existing runs")
    args = ap.parse_args()

    device = get_device()
    rows = []
    for path in CONFIGS:
        cfg = load_config(path)
        if not args.skip_train:
            print(f"training {cfg.name} (train_episode_fraction={cfg.train_episode_fraction})")
            result = train(cfg)
            save_train_outputs(cfg, result)

        ckpt = checkpoint_path(cfg, cfg.epochs)
        out = run_dir(cfg) / "eval" / ckpt.stem
        agent = build_agent_from_checkpoint(ckpt, device)
        summary = run_and_save(agent, cfg.eval.dev_seed_list(), (), out, cfg.eval.max_steps, label=f"{cfg.name}/{ckpt.stem}")
        print(format_summary(summary, cfg.name))

        hist = json.loads((run_dir(cfg) / "history.json").read_text())
        rows.append({
            "name": cfg.name,
            "fraction": cfg.train_episode_fraction,
            "best_val_loss": hist["best_val_loss"],
            "success_rate": summary.success_rate,
            "avg_max_coverage": summary.avg_max_coverage,
        })

    v0_eval = resolve_path("runs/bc_v0/eval/epoch_050/summary.json")
    if v0_eval.exists():
        v0 = json.loads(v0_eval.read_text())
        v0_hist = json.loads(resolve_path("runs/bc_v0/history.json").read_text())
        rows.append({
            "name": "bc_v0",
            "fraction": 1.0,
            "best_val_loss": v0_hist["best_val_loss"],
            "success_rate": v0["success_rate"],
            "avg_max_coverage": v0["avg_max_coverage"],
        })

    print("\nE3 (seeds 0-49, last epoch)")
    print(f"{'run':<12} {'fraction':>8} {'best val':>9} {'success':>8} {'avg max cov':>12}")
    for r in rows:
        print(f"{r['name']:<12} {r['fraction']:>8.2f} {r['best_val_loss']:>9.5f} {r['success_rate']:>8.1%} {r['avg_max_coverage']:>12.3f}")

    out_json = resolve_path("runs/e3_summary.json")
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved {out_json}")

if __name__ == "__main__":
    main()
