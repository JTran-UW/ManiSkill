"""2x2 trajectory grid from a viz_2goal.py .npz/.json (no re-rollout): rows = algorithm, cols = expert / no expert.

    python reinforcement_learning/scripts/plot_2goal_grid.py --src runs/viz_2goal/stoch100_1050 --cells 3 1 5 4 \
        --titles "MaxEntDP + expert data" "MaxEntDP" "FastSAC + expert data" "FastSAC" --out runs/viz_2goal/grid
"""
import json
import math
from dataclasses import dataclass
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import tyro

COLORS = {"A": "#2a78d6", "B": "#eb6834", "neither": "#8a8985"}
TEE_OUTLINE = np.array([[-0.1, -0.0625], [0.1, -0.0625], [0.1, -0.0125], [0.025, -0.0125], [0.025, 0.1375],
                        [-0.025, 0.1375], [-0.025, -0.0125], [-0.1, -0.0125]])
START_XY, START_YAW, GOAL_DIST = np.array([-0.156, 0.0]), -math.pi / 2, 0.15


def tee_poly(xy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return (TEE_OUTLINE @ np.array([[c, s], [-s, c]]) + xy)[:, ::-1]


@dataclass
class Args:
    src: str
    cells: List[int]
    """row-major indices into the src rows (top-left, top-right, bottom-left, bottom-right)"""
    titles: List[str]
    out: str


def main(args: Args):
    data = np.load(f"{args.src}.npz", allow_pickle=True)
    stats = json.load(open(f"{args.src}.json"))
    goals = {"A": START_XY + [0, GOAL_DIST], "B": START_XY - [0, GOAL_DIST]}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#b5b4ad", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True, sharey=True)
    for ax, i, title in zip(axes.flat, args.cells, args.titles):
        traj, outcome, st = data[f"traj_{i}"], data[f"outcome_{i}"], stats[i]
        ax.set_facecolor("#fcfcfb")
        for key, g in goals.items():
            ax.add_patch(Polygon(tee_poly(g, START_YAW), closed=True, fc=COLORS[key], alpha=0.15, ec=COLORS[key], lw=1.5))
            ax.text(g[1], g[0] + 0.165, f"goal {key}", ha="center", color="#0b0b0b")
        ax.add_patch(Polygon(tee_poly(START_XY, START_YAW), closed=True, fill=False, ec="#52514e", lw=1, ls="--"))
        ax.text(START_XY[1], START_XY[0] + 0.165, "start", ha="center", color="#52514e", fontsize=9)
        for k in range(len(traj)):
            ax.plot(traj[k, :, 1], traj[k, :, 0], color=COLORS[outcome[k]], lw=0.8, alpha=0.4)
        for key in ("neither", "A", "B"):
            m = outcome == key
            ax.scatter(traj[m, -1, 1], traj[m, -1, 0], s=12, color=COLORS[key], edgecolors="#fcfcfb", linewidths=0.4, zorder=3,
                       label=f"{'goal ' + key if key != 'neither' else 'neither'}: {100 * m.mean():.0f}%")
        ax.set_aspect("equal")
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.27, 0.06)
        ax.grid(color="#ecebe7", lw=0.6)
        ax.legend(loc="lower center", fontsize=8.5, frameon=False, ncol=3)
        ax.set_title(f"{title}  ({st['episodes']} episodes)", fontsize=11, color="#0b0b0b", loc="left")
    for ax in axes[1]:
        ax.set_xlabel("world y (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("world x (m)")
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    print(f"wrote {args.out}.png")


if __name__ == "__main__":
    main(tyro.cli(Args))
