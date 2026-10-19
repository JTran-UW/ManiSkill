"""Per-state action-outcome map for a FastSAC policy: which sampled actions lead to success.

1. Roll out num_rollouts episodes (sampled actions) from a saved initial state; keep the first successful one and save
   its per-step sim states / obs / actions / policy mu, sigma.
2. For every decision state s_t (t = 0..T-1) of that trajectory, restore `branches` envs to s_t with elapsed steps = t
   (same time remaining), sample a first action from pi(.|s_t), then follow the (stochastic) policy until the episode
   ends. A branch succeeds if the episode has a success at any step (success_once, history included).
3. Video: scene + per-dim tanh-Gaussian pdf of pi(.|s_t), with one dot per branch at (its first action, pdf there),
   green = success, red = failure.

    python reinforcement_learning/scripts/action_success_video.py --checkpoint runs/fastsac_pusht_control/final_ckpt.pt \
        --init-state-path runs/init_states/PushT-v1_10_seed0/init_states.pt --init-state-index 9 \
        --env.kwargs tee_ballast_pos_x 0.0 tee_ballast_pos_y 0.1375 tee_ballast_mass_ratio 0.9 tee_show_ballast True
"""
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

from reinforcement_learning.scripts.action_distribution_video import tanh_gaussian_pdf
from reinforcement_learning.utils.args import EnvArgs, gym_make_kwargs
from reinforcement_learning.utils.init_states import load_init_state
from reinforcement_learning.utils.policies import load_fastsac_actor


@dataclass
class Args:
    checkpoint: str
    init_state_path: str
    init_state_index: int = 9
    env: EnvArgs = field(default_factory=lambda: EnvArgs(id="PushT-v1"))
    branches: int = 100
    """sampled first actions (rollouts) per state"""
    repeats: int = 1
    """continuations per sampled first action; > 1 colors each dot by its success fraction"""
    states_per_batch: int = 10
    """states branched in parallel (num_envs = branches * repeats * states_per_batch; also used for the base rollouts)"""
    max_steps: Optional[int] = None
    """cap the episode at this many steps (default: the env's max_episode_steps)"""
    base_attempts: int = 1
    """batches of base rollouts to try before giving up on finding a successful trajectory"""
    count_history: bool = True
    """a branch also counts as successful if the base trajectory already succeeded before s_t"""
    seed: int = 0
    dim_names: List[str] = field(default_factory=lambda: ["Δx", "Δy", "Δz", "Δrot x", "Δrot y", "Δrot z"])
    out: Optional[str] = None
    render_only: Optional[str] = None
    """skip the rollouts and re-render the video from a saved results .pt"""
    base_trajectory: Optional[str] = None
    """reuse the trajectory (same states) from a saved results .pt instead of rolling out a new one; plots keep its y-axes"""
    label: Optional[str] = None
    """policy name shown in the video"""
    overlay_ref: bool = True
    """with --base-trajectory results, also draw the reference trajectory's (e.g. pre-finetune) distribution in gray, without dots"""


def sim_state(state, rows):
    return {g: {n: v[rows] for n, v in d.items()} for g, d in state.items() if g in ("actors", "articulations")}


def main(args: Args):
    if args.render_only is not None:
        d = torch.load(args.render_only, weights_only=False)
        T = d["outcome"].shape[0]
        render(args, d["trajectory"], d["first_actions"], d["outcome"], T, args.out or args.render_only.replace(".pt", ".mp4"))
        os._exit(0)
    torch.manual_seed(args.seed)
    n = args.branches * args.repeats * args.states_per_batch
    make_kwargs = gym_make_kwargs(args.env)
    make_kwargs.pop("render_mode", None)
    env = gym.make(args.env.id, num_envs=n, **make_kwargs)
    base = env.unwrapped
    dev = base.device
    actor, normalizer = load_fastsac_actor(args.checkpoint, base)
    T = gym_utils.find_max_episode_steps_value(env)
    if args.max_steps is not None:
        T = min(T, args.max_steps)

    def dist(obs):
        with torch.no_grad():
            x = normalizer(obs, update=False) if normalizer is not None else obs
            mu, log_std = actor(x)
        return mu, log_std.exp()

    def sample(mu, sigma):
        return torch.tanh(mu + sigma * torch.randn_like(mu)) * actor.action_scale + actor.action_bias

    def reset_to(states, elapsed):
        obs, _ = env.reset(options=dict(reset_to_env_states=dict(env_states={g: {k: v.to(dev) for k, v in d.items()} for g, d in states.items()})))
        base._elapsed_steps[:] = elapsed
        return obs

    # 1. base trajectory: reuse a saved one, or roll out from the saved initial state and keep the first success
    if args.base_trajectory is not None:
        ref = torch.load(args.base_trajectory, weights_only=False)["trajectory"]
        T = len(ref["states"]) - 1
        mu, sigma = dist(ref["obs"].to(dev))
        traj = dict(ref, mu=mu.cpu(), sigma=sigma.cpu(), mu_ref=ref.get("mu_ref", ref["mu"]), sigma_ref=ref.get("sigma_ref", ref["sigma"]))
        k_first = traj["first_success_step"]
        print(f"reusing trajectory from {args.base_trajectory} ({T} steps, first success at step {k_first})", flush=True)
    else:
        init = load_init_state(args.init_state_path, args.init_state_index)["state"]
        for attempt in range(args.base_attempts):
            obs = reset_to({g: {k: v.expand(n, *v.shape[1:]) for k, v in d.items()} for g, d in init.items()}, 0)
            traj_states, traj_obs, traj_act, traj_mu, traj_sigma = [], [], [], [], []
            success = torch.zeros(n, dtype=torch.bool, device=dev)
            first = torch.full((n,), -1, device=dev)
            for t in range(T + 1):
                traj_states.append(common.to_cpu_tensor(base.get_state_dict()))
                mu, sigma = dist(obs)
                a = sample(mu, sigma)
                traj_obs.append(obs.cpu()); traj_mu.append(mu.cpu()); traj_sigma.append(sigma.cpu()); traj_act.append(a.cpu())
                if t == T:
                    break
                obs, _, _, _, info = env.step(a)
                first[info["success"] & ~success] = t + 1
                success |= info["success"]
            print(f"base rollouts (batch {attempt + 1}): {int(success.sum())}/{n} succeeded within {T} steps from state {args.init_state_index}", flush=True)
            if success.any():
                break
        else:
            raise RuntimeError(f"no successful base rollout within {T} steps after {args.base_attempts * n} attempts")
        k = int(torch.nonzero(success)[0])
        k_first = int(first[k])
        print(f"using env {k} (first success at step {k_first})", flush=True)
        traj = dict(states=[sim_state(s, slice(k, k + 1)) for s in traj_states], obs=torch.stack([o[k] for o in traj_obs]),
                    actions=torch.stack([a[k] for a in traj_act]), mu=torch.stack([m[k] for m in traj_mu]),
                    sigma=torch.stack([s[k] for s in traj_sigma]), first_success_step=k_first)

    # 2. branches from every decision state
    B, R, S = args.branches, args.repeats, args.states_per_batch
    first_actions = torch.zeros(T, B, traj["mu"].shape[1])
    outcome = torch.zeros(T, B)
    max_dobs = max_dmu = 0.0
    for t0 in range(0, T, S):
        ts = list(range(t0, min(t0 + S, T)))
        rows = [t for t in ts for _ in range(B * R)] + [ts[-1]] * (n - len(ts) * B * R)
        state = {g: {name: torch.cat([traj["states"][t][g][name] for t in rows]) for name in traj["states"][0][g]} for g in traj["states"][0]}
        t_env = torch.tensor(rows, device=dev)
        obs = reset_to(state, t_env)
        max_dobs = max(max_dobs, (obs.cpu() - traj["obs"][rows]).abs().max().item())
        mu, sigma = dist(obs)
        max_dmu = max(max_dmu, (mu.cpu() - traj["mu"][rows]).abs().max().item())
        a = sample(mu[::R], sigma[::R]).repeat_interleave(R, dim=0)
        a0 = a[::R].cpu()
        won = torch.tensor([args.count_history and 0 < k_first <= t for t in rows], device=dev)
        for step in range(T - ts[0]):
            active = step < (T - t_env)
            obs, _, _, _, info = env.step(a)
            won |= info["success"] & active
            mu, sigma = dist(obs)
            a = sample(mu, sigma)
        won = won.float().cpu()
        for j, t in enumerate(ts):
            first_actions[t] = a0[j * B:(j + 1) * B]
            outcome[t] = won[j * B * R:(j + 1) * B * R].view(B, R).mean(1)
        print(f"states {ts[0]}-{ts[-1]}: mean branch success " + " ".join(f"{outcome[t].mean():.2f}" for t in ts), flush=True)
    print(f"restore check: max |obs - original obs| {max_dobs:.2e}, max |mu - original mu| {max_dmu:.2e}", flush=True)

    tag = f"_r{R}" if R > 1 else ""
    tag += f"_T{T}" if args.max_steps is not None else ""
    tag += "" if args.count_history else "_nohist"
    out = args.out or os.path.join(os.path.dirname(args.init_state_path),
                                   f"action_success_state{args.init_state_index}_{os.path.basename(os.path.dirname(args.checkpoint))}{tag}.mp4")
    torch.save(dict(trajectory=traj, first_actions=first_actions, outcome=outcome, args=vars(args)), out.replace(".mp4", ".pt"))
    render(args, traj, first_actions, outcome, T, out)
    print("wrote", out, "and", out.replace(".mp4", ".pt"))
    os._exit(0)


def render(args, traj, first_actions, outcome, T, out, vmin=0.0, vmax=1.0, cbar_label="p(success | s, a)", header=None, summary=None):
    """outcome[t, i] colors dot i at state t on [vmin, vmax]; header / summary(t) override the default success-rate texts."""
    B, R = first_actions.shape[1], args.repeats
    k_first = traj["first_success_step"]
    mus, sigmas = traj["mu"].numpy(), traj["sigma"].numpy()
    grid = np.linspace(-0.999, 0.999, 800)
    dens = np.stack([[tanh_gaussian_pdf(grid, mus[t, d], sigmas[t, d]) for d in range(mus.shape[1])] for t in range(T + 1)])
    ref_mu, ref_sigma = traj.get("mu_ref", traj["mu"]).numpy(), traj.get("sigma_ref", traj["sigma"]).numpy()
    ref_dens = np.stack([[tanh_gaussian_pdf(grid, ref_mu[t, d], ref_sigma[t, d]) for d in range(ref_mu.shape[1])] for t in range(T + 1)])
    ymax = np.percentile(ref_dens.max(axis=2), 75, axis=0) * 1.5
    overlay = args.overlay_ref and "mu_ref" in traj
    expert = "mu_exp" in traj
    if expert:
        exp_dens = np.stack([[tanh_gaussian_pdf(grid, traj["mu_exp"][t, d].item(), traj["sigma_exp"][t, d].item()) for d in range(mus.shape[1])]
                             for t in range(T + 1)])
    one = gym.make(args.env.id, num_envs=1, **gym_make_kwargs(args.env))
    one.reset(seed=args.seed)
    ob = one.unwrapped
    writer = imageio.get_writer(out, fps=10 if T > 30 else 2, codec="libx264", quality=8)
    blue, ink, muted, green, red, gray, purple = "#2a78d6", "#0b0b0b", "#52514e", "#1a9e4b", "#d6372a", "#8a8984", "#8a4fd0"
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("fail_success", [red, "#b8b6b0", green])
    for t in range(T + 1):
        ob.set_state_dict({g: {name: v.to(ob.device) for name, v in d.items()} for g, d in traj["states"][t].items()})
        frame = ob.render().cpu().numpy()[0].astype(np.uint8)
        fig = plt.figure(figsize=(14.08, 5.12), dpi=100)
        axf = fig.add_axes([0.0, 0.0, 0.366, 1.0]); axf.imshow(frame); axf.axis("off")
        box = dict(facecolor="black", alpha=0.55, lw=0)
        axf.text(8, 22, f"step {t}/{T}", color="white", fontsize=12, weight="bold", bbox=box)
        done = 0 < k_first <= t
        axf.text(8, 52, "SUCCESS" if done else "pushing", color="#33ff33" if done else "white", fontsize=12, weight="bold", bbox=box)
        if t < T:
            text = summary(t) if summary is not None else f"branch success {outcome[t].mean():.0%}"
            axf.text(8, 82, text, color="white", fontsize=12, weight="bold", bbox=box)
        if args.label:
            axf.text(8, 500, f"policy: {args.label}", color="white", fontsize=12, weight="bold", bbox=box)
        for d in range(mus.shape[1]):
            ax = fig.add_axes([0.40 + (d % 3) * 0.19, 0.56 - (d // 3) * 0.46, 0.16, 0.34])
            if overlay:
                ax.fill_between(grid, ref_dens[t, d], color=gray, alpha=0.18, lw=0)
                ax.plot(grid, ref_dens[t, d], color=gray, lw=1.6)
            if expert:
                ax.fill_between(grid, exp_dens[t, d], color=purple, alpha=0.12, lw=0)
                ax.plot(grid, exp_dens[t, d], color=purple, lw=1.6)
            ax.fill_between(grid, dens[t, d], color=blue, alpha=0.2, lw=0)
            ax.plot(grid, dens[t, d], color=blue, lw=2)
            if t < T:
                x = first_actions[t, :, d].numpy().clip(-0.999, 0.999)
                y = np.minimum(tanh_gaussian_pdf(x, mus[t, d], sigmas[t, d]), ymax[d] * 0.97)
                p = outcome[t].numpy()
                order = np.argsort(np.abs(p - (vmin + vmax) / 2))
                ax.scatter(x[order], y[order], s=18, c=p[order], cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.9, lw=0, zorder=3)
            ax.set_xlim(-1, 1); ax.set_ylim(0, ymax[d])
            if dens[t, d].max() > ymax[d]:
                ax.text(0.98, 0.92, f"peak {dens[t, d].max():.0f} (clipped)", transform=ax.transAxes, ha="right", fontsize=8, color=muted)
            ax.set_title(f"{args.dim_names[d]}  μ={np.tanh(mus[t, d]):+.2f} σ={sigmas[t, d]:.2f}", fontsize=10, color=ink)
            ax.tick_params(labelsize=8, colors=muted)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if d // 3 == 1 and not (overlay or expert):
                ax.set_xlabel("action value", fontsize=9, color=muted)
            if d % 3 == 0:
                ax.set_ylabel("probability density", fontsize=9, color=muted)
        if overlay or expert:
            handles = [matplotlib.lines.Line2D([], [], color=gray, lw=2, label="pre-finetune π(a|s)")] if overlay else []
            handles.append(matplotlib.lines.Line2D([], [], color=blue, lw=2, label=f"{args.label or 'policy'} π(a|s)"))
            if expert:
                handles.append(matplotlib.lines.Line2D([], [], color=purple, lw=2, label="target expert π(a|s)"))
            fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.95, -0.01), ncol=len(handles), frameon=False, fontsize=9)
        if R > 1 or header is not None:
            fig.text(0.40, 0.95, header or (f"π(a|s_t) per dimension{' (x = action value)' if overlay else ''}; dots = {B} sampled first actions, "
                     f"color = success rate of {R} rollouts each"), fontsize=10.5 if not overlay else 10, color=ink)
            cax = fig.add_axes([0.952, 0.12, 0.007, 0.76])
            cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin, vmax), cmap=cmap), cax=cax)
            cb.ax.tick_params(labelsize=8, colors=muted); cb.set_label(cbar_label, fontsize=9, color=muted)
        else:
            fig.text(0.40, 0.95, f"π(a|s_t) per dimension; dots = {B} sampled first actions, green = episode succeeds, red = fails",
                     fontsize=10.5, color=ink)
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)
    writer.close()
    print("wrote", out)


if __name__ == "__main__":
    main(tyro.cli(Args))
