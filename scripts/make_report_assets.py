import argparse
import json
import re
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

from pusht_bc.config import resolve_path

MEDIA = resolve_path("docs/media")
REPORT = resolve_path("REPORT.md")
README = resolve_path("README.md")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

FINAL_RUNS = [
    ("Pretrained Diffusion Policy (`lerobot/diffusion_pusht`)", "runs/diffusion_pusht/eval_final/summary.json"),
    ("BC v0: frozen ResNet-18 features + MLP", "runs/bc_v0/eval/final_epoch_050/summary.json"),
    ("BC E4: ResNet-18 trained end to end, 84×84 crops", "runs/e4_end_to_end/eval/final_epoch_020/summary.json"),
    ("BC E7: E4 + action chunking (8 predicted, 4 executed)", "runs/e7_chunk_e4/eval/final_epoch_020/summary.json"),
]

DEV_RUNS = [
    ("Pretrained Diffusion Policy", "runs/diffusion_pusht/eval/summary.json", "-"),
    ("v0 (100% of train episodes)", "runs/bc_v0/eval/epoch_050/summary.json", "runs/bc_v0/history.json"),
    ("E3: 50% of train episodes", "runs/e3_frac50/eval/epoch_050/summary.json", "runs/e3_frac50/history.json"),
    ("E3: 25% of train episodes", "runs/e3_frac25/eval/epoch_050/summary.json", "runs/e3_frac25/history.json"),
    ("E4: end-to-end encoder", "runs/e4_end_to_end/eval/epoch_020/summary.json", "runs/e4_end_to_end/history.json"),
    ("E7: v0 + action chunking", "runs/e7_chunk_v0/eval/epoch_050/summary.json", "runs/e7_chunk_v0/history.json"),
    ("E7: E4 + action chunking", "runs/e7_chunk_e4/eval/epoch_020/summary.json", "runs/e7_chunk_e4/history.json"),
]

COMPARE_SEED = 0
DP_DEV = "runs/diffusion_pusht/eval"
E4_DEV = "runs/e4_end_to_end/eval/epoch_020"
E4_FAILURES = "runs/e4_end_to_end/eval/failures_0_9"
E7_REPLAY = "runs/e7_chunk_e4/eval/cpu_replay_0_49"
E7_SEED = 2

# Failure labels from watching the failures_0_9 videos: seed -> (failure mode, what happens)
FAILURE_LABELS = {
    0: ("Stall", "pushes the T smoothly into the target area, then stops as if it were done"),
    1: ("Drift", "pushes the T the wrong way, into the top wall, and never recovers"),
    2: ("Stall", "good start, then stops with about 40% of the T inside the target"),
    3: ("Near miss", "gets the T almost in place (94.4%), then stays still without the final correction"),
    4: ("Stall", "pushes the T toward the target, then leaves it next to the target without overlap"),
    5: ("Drift", "starts placing the T, then pushes it away from the target"),
    6: ("Drift", "easy start with the T already partly on the target, then pushes it away"),
    7: ("Never reaches the block", "the pusher goes to the bottom edge and stays there; the T is never touched"),
    8: ("Drift", "moves the T around, away from the target, then parks in a corner"),
    9: ("Stall", "lines up to push from the top and moves the T next to the target, then gets stuck without overlap"),
}

# Extra E4 examples for the README: (video, gif name). The near miss comes from the failures_0_9 recording.
EXAMPLES = [
    (f"{E4_FAILURES}/seed_3.mp4", "e4_seed3_near_miss.gif"),
    (f"{E4_DEV}/seed_1.mp4", "e4_seed1_complete_failure.gif"),
    (f"{E7_REPLAY}/seed_{E7_SEED}.mp4", f"e7_seed{E7_SEED}_success.gif"),
]

def load_summary(path: str) -> dict | None:
    """
    Load a summary.json written by run_and_save.

    Args:
        path: repo-relative path.
    Returns:
        The dict, or None if the file is not there yet.
    """
    p = resolve_path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

def episode(summary: dict, seed: int) -> dict | None:
    """
    One episode of a summary.

    Args:
        summary: loaded summary.json.
        seed: env seed.
    Returns:
        The episode dict, or None if the seed was not evaluated.
    """
    return next((e for e in summary["episodes"] if e["seed"] == seed), None)

def seed_range(summary: dict) -> str:
    """
    Seeds of a summary as 'a–b' (or a comma list when not contiguous).

    Args:
        summary: loaded summary.json.
    Returns:
        The text.
    """
    seeds = [e["seed"] for e in summary["episodes"]]
    if seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}–{seeds[-1]}"
    return ", ".join(str(s) for s in seeds)

def final_table() -> str:
    """
    Markdown results table on the final seeds (pending rows for missing runs).

    Returns:
        The table.
    """
    lines = [
        "| Policy | Seeds | Success | 95% CI | Avg max coverage |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, path in FINAL_RUNS:
        s = load_summary(path)
        if s is None:
            lines.append(f"| {name} | 1000–1199 | *pending* | | |")
            continue
        lines.append(
            f"| {name} | {seed_range(s)} (n={s['n_episodes']}) | {s['success_rate']:.1%} | "
            f"{s['ci_low']:.1%}–{s['ci_high']:.1%} | {s['avg_max_coverage']:.3f} |"
        )
    return "\n".join(lines)

def dev_table() -> str:
    """
    Markdown table of the development runs on seeds 0–49.

    Returns:
        The table.
    """
    lines = [
        "| Run | Best val loss | Success | Avg max coverage |",
        "| --- | --- | --- | --- |",
    ]
    for name, path, hist_path in DEV_RUNS:
        s = load_summary(path)
        hist = load_summary(hist_path) if hist_path != "-" else None
        if s is None and hist is None and not hist_path.startswith("runs/bc_v0"):
            lines.append(f"| {name} | - | *pending* | |")
            continue
        val = f"{hist['best_val_loss']:.5f}" if hist else "-"
        if s is None:
            lines.append(f"| {name} | {val} | *pending* | |")
            continue
        lines.append(f"| {name} | {val} | {s['success_rate']:.0%} | {s['avg_max_coverage']:.3f} |")
    return "\n".join(lines)

def failure_table() -> str:
    """
    Failure-labeling table for the recorded E4 failures (labels are filled in by hand).

    Returns:
        The table, or a pending note if the videos are not downloaded yet.
    """
    s = load_summary(f"{E4_FAILURES}/summary.json")
    if s is None:
        return f"*pending: download `{E4_FAILURES}/` from Colab.*"
    lines = [
        "| Seed | Start coverage | Max coverage | Failure mode | What happens |",
        "| --- | --- | --- | --- | --- |",
    ]
    for e in s["episodes"]:
        mode, note = FAILURE_LABELS.get(e["seed"], ("TODO", "TODO"))
        lines.append(f"| {e['seed']} | {e['rewards'][0]:.3f} | {e['max_coverage']:.3f} | {mode} | {note} |")
    return "\n".join(lines)

def style_axes(ax) -> None:
    """
    Recessive grid and axes on the chart surface.

    Args:
        ax: matplotlib axes.
    """
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2)

def coverage_plot(seed: int, runs: list[tuple[str, str, str]], name: str) -> Path | None:
    """
    Coverage over time of several policies on the same seed.

    Args:
        seed: env seed present in every summary.
        runs: (label, folder with summary.json, color) per policy.
        name: output file name in docs/media.
    Returns:
        The .png written, or None if a summary or the seed is missing.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = []
    for label, folder, color in runs:
        s = load_summary(f"{folder}/summary.json")
        if s is None or episode(s, seed) is None:
            return None
        curves.append((label, episode(s, seed), color))

    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    style_axes(ax)
    ax.axhline(0.95, color=INK_2, linestyle="--", linewidth=1, zorder=1)
    ax.annotate("success (0.95)", (0, 0.95), textcoords="offset points", xytext=(4, 5), ha="left", fontsize=9, color=INK_2)
    for label, ep, color in curves:
        steps = np.arange(1, len(ep["rewards"]) + 1)
        ax.plot(steps, ep["rewards"], color=color, linewidth=2, label=label, zorder=2)
        outcome = "success" if ep["success"] else f"max {ep['max_coverage']:.2f}"
        ax.annotate(f"{label}: {outcome}", (steps[-1], ep["rewards"][-1]), textcoords="offset points",
                    xytext=(6, -12 if ep["success"] else 0), va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("step", color=INK_2)
    ax.set_ylabel("coverage of the target", color=INK_2)
    ax.set_title(f"Coverage over time, seed {seed} (same starting position)", loc="left", fontsize=11, color=INK)
    ax.legend(loc="lower right", frameon=False, labelcolor=INK_2)
    fig.tight_layout()

    out = MEDIA / name
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out

def make_gif(src: str, name: str, speed: int = 1, width: int = 256, fps: int = 10) -> Path | None:
    """
    Downscaled GIF from an evaluation video.

    Args:
        src: repo-relative .mp4.
        name: output file name in docs/media.
        speed: keep every speed-th frame (2 = twice as fast).
        width: output width in pixels.
        fps: playback frame rate.
    Returns:
        The .gif written, or None if the video is missing.
    """
    p = resolve_path(src)
    if not p.exists():
        return None
    frames = iio.imread(p)[::speed]
    h = round(frames.shape[1] * width / frames.shape[2])
    images = [Image.fromarray(f).resize((width, h), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    images += [images[-1]] * fps  # hold the last frame for a second
    out = MEDIA / name
    images[0].save(out, save_all=True, append_images=images[1:], duration=round(1000 / fps), loop=0, optimize=True)
    return out

def fill_block(path: Path, key: str, content: str) -> bool:
    """
    Replace the text between <!-- BEGIN:key --> and <!-- END:key --> in a Markdown file.

    Args:
        path: Markdown file.
        key: block name.
        content: new text for the block.
    Returns:
        True if the block was found.
    """
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(rf"(<!-- BEGIN:{key} -->\n).*?(<!-- END:{key} -->)", re.S)
    if not pattern.search(text):
        return False
    path.write_text(pattern.sub(lambda m: m.group(1) + content + "\n" + m.group(2), text), encoding="utf-8", newline="\n")
    return True

def main() -> None:
    """
    Build the report media and tables, and fill the generated blocks of REPORT.md and README.md.

    Usage: python scripts/make_report_assets.py [--failures 3 7]
    Safe to re-run: runs that are not downloaded yet show as pending.
    """
    ap = argparse.ArgumentParser(description="Build REPORT.md / README.md assets from the run summaries.")
    ap.add_argument("--failures", type=int, nargs="*", default=[], help="two E4 failure seeds from failures_0_9 to turn into GIFs")
    args = ap.parse_args()
    MEDIA.mkdir(parents=True, exist_ok=True)

    written = []
    written.append(coverage_plot(COMPARE_SEED, [("Diffusion Policy", DP_DEV, BLUE), ("BC E4", E4_DEV, ORANGE)],
                                 f"coverage_seed{COMPARE_SEED}_dp_vs_e4.png"))
    written.append(coverage_plot(E7_SEED, [("BC E4", E4_DEV, ORANGE), ("BC E4 + chunking", E7_REPLAY, AQUA)],
                                 f"coverage_seed{E7_SEED}_e4_vs_e7.png"))
    written.append(make_gif(f"{DP_DEV}/seed_{COMPARE_SEED}.mp4", f"dp_seed{COMPARE_SEED}_success.gif"))
    written.append(make_gif(f"{E4_DEV}/seed_{COMPARE_SEED}.mp4", f"e4_seed{COMPARE_SEED}_failure.gif", speed=2))
    for src, name in EXAMPLES:
        gif = make_gif(src, name, speed=2)
        if gif is None:
            print(f"skipped {name}: {src} not found")
        written.append(gif)
    for seed in args.failures:
        written.append(make_gif(f"{E4_FAILURES}/seed_{seed}.mp4", f"e4_failure_seed{seed}.gif", speed=2))
    for p in written:
        if p is not None:
            print(f"wrote {p.relative_to(resolve_path('.'))} ({p.stat().st_size / 1e6:.1f} MB)")

    blocks = {
        "final-table": final_table(),
        "dev-table": dev_table(),
        "failure-table": failure_table(),
    }
    for doc in (REPORT, README):
        for key, content in blocks.items():
            if fill_block(doc, key, content):
                print(f"filled {key} in {doc.name}")

    print("\n" + final_table())

if __name__ == "__main__":
    main()
