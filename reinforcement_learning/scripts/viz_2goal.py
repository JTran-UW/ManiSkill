"""Roll out FastSAC / MaxEntDP checkpoints on PushT2Goal-v1 default resets and plot where the T goes.

One row per checkpoint: top-down T-center trajectories colored by outcome (goal A / goal B / neither) and a histogram
of the final T y. Saves <out>.png, <out>.npz (trajectories) and <out>.json (goal fractions).

    python reinforcement_learning/scripts/viz_2goal.py --ckpts runs/a/final_ckpt.pt runs/b/final_ckpt.pt \
        --labels "MaxEntDP" "FastSAC" --out runs/viz/compare
"""
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import torch
import tyro

import mani_skill.envs  # noqa: F401
from reinforcement_learning.utils.args import parse_env_kwargs
from reinforcement_learning.utils.utils import EmpiricalNormalization

COLORS = {"A": "#2a78d6", "B": "#eb6834", "neither": "#8a8985"}
TEE_OUTLINE = np.array([[-0.1, -0.0625], [0.1, -0.0625], [0.1, -0.0125], [0.025, -0.0125], [0.025, 0.1375],
                        [-0.025, 0.1375], [-0.025, -0.0125], [-0.1, -0.0125]])


@dataclass
class Args:
    ckpts: List[str]
    labels: Optional[List[str]] = None
    out: str = "runs/viz_2goal/compare"
    num_episodes: int = 512
    mode: Optional[List[str]] = None
    """per ckpt: fastsac mean|sample, maxentdp sample|select (default: fastsac mean, maxentdp sample)"""
    seed: int = 0
    env_kwargs: List[str] = field(default_factory=list)
    """extra KEY VALUE pairs overriding the checkpoint's env kwargs (reset_datasets is always dropped)"""
    max_lines: int = 200
    num_steps: int = 100
    """episode length (PushT2Goal-v1 max_episode_steps)"""


def make_policy(ckpt, envs, device, mode):
    algo = ckpt["config"]["algo"]
    if "diffusion_steps" in algo:
        from reinforcement_learning.maxentdp.maxentdp import AlgoArgs, build_actor
        from reinforcement_learning.fastsac.networks.fastsac_critic import DistributionalQNetwork
        a = AlgoArgs(**{k: v for k, v in algo.items() if k in AlgoArgs.__dataclass_fields__})
        a.actor_hidden_dims = tuple(a.actor_hidden_dims)
        actor = build_actor(envs, a, device)
        actor.load_state_dict(ckpt["actor"])
        qfs = []
        for k in ("qf1", "qf2"):
            q = DistributionalQNetwork(envs, v_min=a.v_min, v_max=a.v_max).to(device)
            q.load_state_dict(ckpt[k])
            qfs.append(q)
        mode = mode or "sample"

        def act(o):
            if mode == "sample":
                return actor.sample(o)
            n, B = a.eval_candidates, o.shape[0]
            rep = o.repeat_interleave(n, 0)
            c = actor.sample(rep)
            qv = torch.min(*[q.get_value(torch.softmax(q(rep, c), -1)) for q in qfs]).view(B, n)
            return c.view(B, n, -1)[torch.arange(B, device=o.device), qv.argmax(1)]
        name = f"MaxEntDP ({mode})"
    else:
        from reinforcement_learning.fastsac.networks.fastsac_actor import Actor
        actor = Actor(envs, action_scale=torch.ones(envs.single_action_space.shape), action_bias=torch.zeros(envs.single_action_space.shape),
                      use_tanh=algo.get("use_tanh", True)).to(device)
        actor.load_state_dict(ckpt["actor"])
        mode = mode or "mean"
        act = lambda o: actor.get_action(o)[2 if mode == "mean" else 0]
        name = f"FastSAC ({mode})"
    norm = None
    if algo.get("obs_normalization", True):
        norm = EmpiricalNormalization(shape=envs.single_observation_space.shape, device=device)
        norm.load_state_dict(ckpt["obs_normalizer"])
    return (lambda o: act(norm(o, update=False) if norm is not None else o)), name


@torch.no_grad()
def rollout(path, args, mode):
    ckpt = torch.load(path, weights_only=False)
    env_cfg = ckpt["config"]["env"]
    kwargs = parse_env_kwargs(env_cfg.get("kwargs", {}))
    kwargs.pop("reset_datasets", None)
    kwargs.pop("reset_dataset_weights", None)
    kwargs.pop("valid_goals", None)
    kwargs.update(parse_env_kwargs(dict(zip(args.env_kwargs[::2], args.env_kwargs[1::2]))))
    env = gym.make("PushT2Goal-v1", num_envs=args.num_episodes, obs_mode=env_cfg["obs_mode"], control_mode=env_cfg["control_mode"],
                   reward_mode="sparse", **kwargs)
    e = env.unwrapped  # raw batched env: no auto-reset, so the last recorded pose is the episode's final pose
    policy, name = make_policy(ckpt, e, torch.device("cuda"), mode)
    obs, _ = env.reset(seed=args.seed)
    traj = [e.tee.pose.p[:, :2].cpu().numpy()]
    hit_a = torch.zeros(args.num_episodes, dtype=torch.bool, device=e.device)
    hit_b = torch.zeros_like(hit_a)
    for _ in range(args.num_steps):
        obs, _, _, _, info = env.step(policy(obs))
        traj.append(e.tee.pose.p[:, :2].cpu().numpy())
        hit_a |= info["success_a"]
        hit_b |= info["success_b"]
    env.close()
    traj = np.stack(traj, 1)
    a, b = hit_a.cpu().numpy(), hit_b.cpu().numpy()
    outcome = np.where(a & ~b, "A", np.where(b & ~a, "B", np.where(a & b, "A", "neither")))
    stats = dict(checkpoint=path, policy=name, episodes=int(len(a)), goal_a=float(a.mean()), goal_b=float(b.mean()),
                 both=float((a & b).mean()), neither=float((~a & ~b).mean()), success_once=float((a | b).mean()),
                 global_step=ckpt.get("global_step"))
    return traj, outcome, stats, e


def tee_poly(xy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    pts = TEE_OUTLINE @ np.array([[c, s], [-s, c]]) + xy
    return pts[:, ::-1]  # plot (y, x)


def main(args: Args):
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    labels = args.labels or [os.path.basename(os.path.dirname(p)) for p in args.ckpts]
    modes = args.mode or [None] * len(args.ckpts)
    rows = []
    for path, label, mode in zip(args.ckpts, labels, modes):
        traj, outcome, stats, e = rollout(path, args, mode)
        stats["label"] = label
        rows.append((traj, outcome, stats))
        print(json.dumps(stats), flush=True)
    goals = e._goal_xy.cpu().numpy()
    start, yaw0, dist = np.array(e.START_XY), e.START_YAW, e.goal_dist

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#b5b4ad", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 4.2 * len(rows)), gridspec_kw=dict(width_ratios=[1.35, 1]), squeeze=False)
    rng = np.random.default_rng(0)
    for (traj, outcome, st), (ax, axh) in zip(rows, axes):
        ax.set_facecolor("#fcfcfb")
        for g, key in ((goals[0], "A"), (goals[1], "B")):
            ax.add_patch(Polygon(tee_poly(g, yaw0), closed=True, fill=True, fc=COLORS[key], alpha=0.15, ec=COLORS[key], lw=1.5))
            ax.text(g[1], g[0] + 0.17, f"goal {key}", ha="center", color="#0b0b0b", fontsize=10)
        ax.add_patch(Polygon(tee_poly(start, yaw0), closed=True, fill=False, ec="#52514e", lw=1, ls="--"))
        ax.text(start[1], start[0] + 0.17, "start", ha="center", color="#52514e", fontsize=9)
        idx = rng.permutation(len(traj))[: args.max_lines]
        for i in idx:
            ax.plot(traj[i, :, 1], traj[i, :, 0], color=COLORS[outcome[i]], lw=0.8, alpha=0.35)
        for key in ("neither", "A", "B"):
            m = outcome == key
            ax.scatter(traj[m, -1, 1], traj[m, -1, 0], s=10, color=COLORS[key], edgecolors="#fcfcfb", linewidths=0.4, zorder=3,
                       label=f"{'goal ' + key if key != 'neither' else 'neither'}: {100 * m.mean():.1f}%")
        ax.set_aspect("equal")
        ax.set_xlim(-0.33, 0.33)
        ax.set_ylim(-0.32, 0.12)
        ax.set_xlabel("world y (m)")
        ax.set_ylabel("world x (m)")
        ax.grid(color="#ecebe7", lw=0.6)
        ax.legend(loc="lower center", fontsize=8, frameon=False, ncol=3)
        step = f" @ {st['global_step'] / 1e6:.1f}M" if st.get("global_step") else ""
        ax.set_title(f"{st['label']}{step} — {st['policy']}, {st['episodes']} eps", fontsize=10, color="#0b0b0b", loc="left")
        bins = np.linspace(-0.3, 0.3, 61)
        axh.hist(traj[:, -1, 1], bins=bins, color="#52514e", alpha=0.85, rwidth=0.85)
        for g, key in ((goals[0], "A"), (goals[1], "B")):
            axh.axvline(g[1], color=COLORS[key], lw=1.5, ls="--")
            axh.text(g[1], axh.get_ylim()[1] * 0.95, f" {key}", color="#0b0b0b", fontsize=9)
        axh.set_xlabel("final T y (m)")
        axh.set_ylabel("episodes")
        axh.grid(axis="y", color="#ecebe7", lw=0.6)
        axh.set_title(f"A {100 * st['goal_a']:.1f}%  B {100 * st['goal_b']:.1f}%  neither {100 * st['neither']:.1f}%", fontsize=10, color="#0b0b0b", loc="left")
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=130)
    np.savez_compressed(f"{args.out}.npz", **{f"traj_{i}": r[0] for i, r in enumerate(rows)}, **{f"outcome_{i}": r[1] for i, r in enumerate(rows)})
    with open(f"{args.out}.json", "w") as f:
        json.dump([r[2] for r in rows], f, indent=1)
    print(f"wrote {args.out}.png/.npz/.json")


if __name__ == "__main__":
    main(tyro.cli(Args))
