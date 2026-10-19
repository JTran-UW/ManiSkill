"""Eval success (default resets) vs env steps from tensorboard event dirs; one line per run, direct-labelled.

    python reinforcement_learning/scripts/plot_success_curves.py --dirs runs/a runs/b --labels A B --out runs/viz_2goal/curves
"""
from dataclasses import dataclass
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


@dataclass
class Args:
    dirs: List[str]
    labels: List[str]
    out: str
    tag: str = "eval/success_once"
    title: str = "PushT2Goal-v1: eval success from default resets (either goal)"
    subtitle: str = "128 episodes per point; FastSAC eval = deterministic mean, MaxEntDP eval = sampled actions"


def main(args: Args):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#b5b4ad", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.set_facecolor("#fcfcfb")
    rows = []
    for d, label, color in zip(args.dirs, args.labels, SERIES):
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        ev = {e.step: e.value for e in ea.Scalars(args.tag)}
        x = np.array(sorted(ev)) / 1e6
        y = np.array([ev[s] for s in sorted(ev)])
        ax.plot(x, y, color=color, lw=2)
        rows.append((label, x, y))
    ax.set_xlabel("env steps (M)")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, max(r[1][-1] for r in rows) * 1.02)
    ax.grid(color="#ecebe7", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend([plt.Line2D([], [], color=c, lw=2) for c in SERIES[: len(rows)]], [f"{r[0]}  (latest {r[2][-1]:.2f} @ {r[1][-1]:.0f}M)" for r in rows], frameon=False, loc="lower right")
    ax.set_title(args.title, loc="left", fontsize=11, color="#0b0b0b", pad=18)
    ax.text(0, 1.02, args.subtitle, transform=ax.transAxes, fontsize=8.5, color="#52514e")
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    with open(f"{args.out}.csv", "w") as f:
        f.write("run,step_M,success\n")
        for label, x, y in rows:
            f.writelines(f"{label},{a:.3f},{b:.4f}\n" for a, b in zip(x, y))
    print(f"wrote {args.out}.png/.csv")


if __name__ == "__main__":
    main(tyro.cli(Args))
