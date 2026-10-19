"""Video of one successful episode next to live per-dimension action distributions of a FastSAC policy.

Rolls out num_envs episodes (sampled actions) from a saved initial state, picks a successful one, and re-renders it
from its recorded sim states. Each action dim is tanh(u), u ~ N(mu, sigma), so its density on (-1, 1) is
N(atanh(a); mu, sigma) / (1 - a^2); the action actually taken at that step is marked.

    python reinforcement_learning/scripts/action_distribution_video.py --checkpoint runs/fastsac_pusht_control/final_ckpt.pt \
        --init-state-path runs/init_states/PushT-v1_10_seed0/init_states.pt --init-state-index 9 \
        --env.kwargs tee_ballast_pos_x 0.0 tee_ballast_pos_y 0.1375 tee_ballast_mass_ratio 0.9 tee_show_ballast True
"""
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import gymnasium as gym
import imageio.v2 as imageio
import mani_skill.envs  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from mani_skill.utils import common, gym_utils

from reinforcement_learning.utils.args import EnvArgs, gym_make_kwargs
from reinforcement_learning.utils.init_states import FixedInitStateWrapper, load_init_state
from reinforcement_learning.utils.policies import load_fastsac_actor


@dataclass
class Args:
    checkpoint: str
    init_state_path: str
    init_state_index: int = 9
    env: EnvArgs = field(default_factory=lambda: EnvArgs(id="PushT-v1"))
    num_envs: int = 64
    seed: int = 0
    episode: Optional[int] = None
    """which env's episode to render (default: the first successful one)"""
    dim_names: List[str] = field(default_factory=lambda: ["Δx", "Δy", "Δz", "Δrot x", "Δrot y", "Δrot z"])
    out: Optional[str] = None


def tanh_gaussian_pdf(a, mu, sigma):
    u = np.arctanh(a)
    return np.exp(-0.5 * ((u - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)) / (1 - a ** 2)


def main(args: Args):
    torch.manual_seed(args.seed)
    env = gym.make(args.env.id, num_envs=args.num_envs, **gym_make_kwargs(args.env))
    base = env.unwrapped
    env = FixedInitStateWrapper(env, load_init_state(args.init_state_path, args.init_state_index))
    actor, normalizer = load_fastsac_actor(args.checkpoint, base)
    steps = gym_utils.find_max_episode_steps_value(env)

    obs, _ = env.reset(seed=args.seed)
    states, mus, sigmas, actions, success = [], [], [], [], torch.zeros(args.num_envs, dtype=torch.bool, device=base.device)
    first_success = torch.full((args.num_envs,), -1, device=base.device)
    for t in range(steps + 1):
        states.append(common.to_cpu_tensor(base.get_state_dict()))
        with torch.no_grad():
            x = normalizer(obs, update=False) if normalizer is not None else obs
            mu, log_std = actor(x)
            u = mu + log_std.exp() * torch.randn_like(mu)
            action = torch.tanh(u) * actor.action_scale + actor.action_bias
        mus.append(mu.cpu().numpy()); sigmas.append(log_std.exp().cpu().numpy()); actions.append(action.cpu().numpy())
        if t == steps:
            break
        obs, _, _, _, info = env.step(action)
        newly = info["success"] & ~success
        first_success[newly] = t + 1
        success |= info["success"]
    success = success.cpu().numpy()
    print(f"{success.sum()}/{args.num_envs} episodes from state {args.init_state_index} succeeded")
    k = args.episode if args.episode is not None else int(np.flatnonzero(success)[0])
    k_first = int(first_success[k])
    print(f"rendering env {k} (success={bool(success[k])}, first success at step {k_first})")
    mus, sigmas, actions = np.stack(mus)[:, k], np.stack(sigmas)[:, k], np.stack(actions)[:, k]

    grid = np.linspace(-0.999, 0.999, 800)
    dens = np.stack([[tanh_gaussian_pdf(grid, mus[t, d], sigmas[t, d]) for d in range(mus.shape[1])] for t in range(len(mus))])
    ymax = np.percentile(dens.max(axis=2), 75, axis=0) * 1.5  # per-dim fixed axis; rare near-bound spikes are clipped and labeled

    one = gym.make(args.env.id, num_envs=1, **gym_make_kwargs(args.env))
    one.reset(seed=args.seed)
    ob = one.unwrapped
    out = args.out or os.path.join(os.path.dirname(args.init_state_path),
                                   f"action_dist_state{args.init_state_index}_{os.path.basename(os.path.dirname(args.checkpoint))}_env{k}.mp4")
    writer = imageio.get_writer(out, fps=15, codec="libx264", quality=8)
    blue, ink, muted = "#2a78d6", "#0b0b0b", "#52514e"
    for t in range(len(mus)):
        ob.set_state_dict({g: {n: v[k:k + 1].to(ob.device) for n, v in d.items()} for g, d in states[t].items() if g in ("actors", "articulations")})
        frame = ob.render().cpu().numpy()[0].astype(np.uint8)
        fig = plt.figure(figsize=(14.08, 5.12), dpi=100)
        axf = fig.add_axes([0.0, 0.0, 0.366, 1.0]); axf.imshow(frame); axf.axis("off")
        done = 0 < k_first <= t
        axf.text(8, 22, f"step {t}/{steps}", color="white", fontsize=12, weight="bold", bbox=dict(facecolor="black", alpha=0.55, lw=0))
        axf.text(8, 52, "SUCCESS" if done else "pushing", color="#33ff33" if done else "white", fontsize=12, weight="bold", bbox=dict(facecolor="black", alpha=0.55, lw=0))
        for d in range(mus.shape[1]):
            ax = fig.add_axes([0.40 + (d % 3) * 0.2, 0.56 - (d // 3) * 0.46, 0.17, 0.34])
            ax.fill_between(grid, dens[t, d], color=blue, alpha=0.25, lw=0)
            ax.plot(grid, dens[t, d], color=blue, lw=2)
            if t < steps:
                ax.axvline(actions[t, d], color=ink, lw=1.2, ls="--")
            ax.set_xlim(-1, 1); ax.set_ylim(0, ymax[d])
            if dens[t, d].max() > ymax[d]:
                ax.text(0.98, 0.92, f"peak {dens[t, d].max():.0f} (clipped)", transform=ax.transAxes, ha="right", fontsize=8, color=muted)
            ax.set_title(f"{args.dim_names[d]}  μ={np.tanh(mus[t, d]):+.2f} σ={sigmas[t, d]:.2f}", fontsize=10, color=ink)
            ax.tick_params(labelsize=8, colors=muted)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if d // 3 == 1:
                ax.set_xlabel("action value", fontsize=9, color=muted)
            if d % 3 == 0:
                ax.set_ylabel("probability density", fontsize=9, color=muted)
        fig.text(0.40, 0.95, "policy action distribution per dimension (dashed = action taken)", fontsize=11, color=ink)
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)
    writer.close()
    json.dump(dict(env=k, success=bool(success[k]), first_success_step=k_first, n_success=int(success.sum()), num_envs=args.num_envs,
                   mu=mus.tolist(), sigma=sigmas.tolist(), action=actions.tolist()), open(out.replace(".mp4", ".json"), "w"))
    print("wrote", out)
    os._exit(0)


if __name__ == "__main__":
    main(tyro.cli(Args))
