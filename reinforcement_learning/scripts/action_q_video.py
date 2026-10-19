"""Policy action distributions along a saved trajectory, with sampled actions colored by a fixed critic's Q(s, a).

For each policy checkpoint and each state s_t of a saved trajectory (results .pt from action_success_video.py), sample
`branches` actions from pi(.|s_t) and color them by Q_ref(s_t, a): the mean expected value of the reference checkpoint's
two distributional critics (what its actor maximizes), on obs normalized with the reference's own normalizer. One color
scale is shared by all videos. No rollouts.

    python reinforcement_learning/scripts/action_q_video.py --q-checkpoint runs/pusht_stem_tip_ballast/final_ckpt.pt \
        --base-trajectory runs/init_states/.../step0_prefinetune.pt --out-dir runs/init_states/.../q_videos \
        --policies "pre-finetune=runs/fastsac_pusht_control/final_ckpt.pt" "finetune +9.3k steps=runs/.../ckpt_20010068.pt" \
        --env.kwargs tee_ballast_pos_x 0.0 tee_ballast_pos_y 0.1375 tee_ballast_mass_ratio 0.9 tee_show_ballast True
"""
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import tyro

from reinforcement_learning.fastsac.networks.fastsac_critic import DistributionalQNetwork
from reinforcement_learning.scripts.action_success_video import render
from reinforcement_learning.utils.args import EnvArgs
from reinforcement_learning.utils.policies import load_fastsac_actor
from reinforcement_learning.utils.utils import EmpiricalNormalization


@dataclass
class Args:
    q_checkpoint: str
    base_trajectory: str
    policies: List[str]
    """LABEL=CHECKPOINT, one video each"""
    out_dir: str
    env: EnvArgs = field(default_factory=lambda: EnvArgs(id="PushT-v1"))
    branches: int = 100
    seed: int = 0
    color_by: str = "q_per_state"
    """'q_per_state': raw Q, color scale stretched per state over this policy's own samples at that state;
    'q': raw Q on one global scale (dominated by state-to-state differences);
    'advantage': Q(s, a) - Q(s, a_ref) with a_ref = the reference checkpoint's own mean action"""
    color_percentiles: List[float] = field(default_factory=lambda: [2.0, 98.0])
    """color_by=q: shared range = these percentiles of all Q values; color_by=advantage: +-(upper percentile of |A|)"""
    dim_names: List[str] = field(default_factory=lambda: ["Δx", "Δy", "Δz", "Δrot x", "Δrot y", "Δrot z"])
    overlay_ref: bool = True
    overlay_expert: bool = True
    q_source: str = "expert"
    """'expert': color by the --q-checkpoint critic; 'own': color by each policy's own critic (from its checkpoint)"""
    """also draw the q-checkpoint (target expert) policy's distribution"""
    repeats: int = 1
    label: str = ""


def main(args: Args):
    torch.manual_seed(args.seed)
    ref = torch.load(args.base_trajectory, weights_only=False)["trajectory"]
    obs = ref["obs"].cuda()
    T = obs.shape[0] - 1
    spaces = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, obs.shape[1:]),
                             single_action_space=gym.spaces.Box(-1.0, 1.0, (ref["mu"].shape[1],)))

    def load_q(checkpoint):
        qc = torch.load(checkpoint, map_location="cuda", weights_only=False)
        algo = qc["config"]["algo"]
        critics = []
        for name in ("qf1", "qf2"):
            q = DistributionalQNetwork(spaces, v_min=algo["v_min"], v_max=algo["v_max"]).cuda()
            q.load_state_dict(qc[name]); q.eval(); critics.append(q)
        qnorm = EmpiricalNormalization(shape=obs.shape[1:], device="cuda")
        qnorm.load_state_dict(qc["obs_normalizer"]); qnorm.eval()

        @torch.no_grad()
        def q_value(o, a):
            x = qnorm(o, update=False)
            return torch.stack([q.get_value(F.softmax(q(x, a), dim=-1)) for q in critics]).mean(0)
        return q_value

    q_value = load_q(args.q_checkpoint)

    ref_actor, ref_norm = load_fastsac_actor(args.q_checkpoint, spaces)
    with torch.no_grad():
        mu_ref, log_std_ref = ref_actor(ref_norm(obs, update=False) if ref_norm is not None else obs)
    baseline = q_value(obs, torch.tanh(mu_ref) * ref_actor.action_scale + ref_actor.action_bias)[:T].cpu()
    adv = args.color_by == "advantage"

    results = []
    for spec in args.policies:
        label, ckpt = spec.split("=", 1)
        actor, norm = load_fastsac_actor(ckpt, spaces)
        q_value = load_q(ckpt if args.q_source == "own" else args.q_checkpoint)
        with torch.no_grad():
            mu, log_std = actor(norm(obs, update=False) if norm is not None else obs)
        sigma = log_std.exp()
        B = args.branches
        u = mu[:T, None] + sigma[:T, None] * torch.randn(T, B, mu.shape[1], device="cuda")
        a = torch.tanh(u) * actor.action_scale + actor.action_bias
        q = q_value(obs[:T, None].expand(T, B, -1).reshape(T * B, -1), a.reshape(T * B, -1)).view(T, B)
        q_mean_action = q_value(obs, torch.tanh(mu) * actor.action_scale + actor.action_bias)
        traj = dict(ref, mu=mu.cpu(), sigma=sigma.cpu(), mu_ref=ref.get("mu_ref", ref["mu"]), sigma_ref=ref.get("sigma_ref", ref["sigma"]))
        results.append((label, ckpt, traj, a.cpu(), q.cpu(), q_mean_action.cpu()))
        print(f"{label}: mean Q of sampled actions {q.mean():.2f}, of the policy mean action {q_mean_action[:T].mean():.2f}", flush=True)

    per_state = args.color_by == "q_per_state"
    if per_state:
        def per_state_color(q):
            lo, hi = np.percentile(q.numpy(), args.color_percentiles, axis=1)
            return torch.from_numpy(np.clip((q.numpy() - lo[:, None]) / np.maximum(hi - lo, 1e-6)[:, None], 0, 1))
        results = [(l, c, tr, a, q, qm, per_state_color(q)) for l, c, tr, a, q, qm in results]
        vmin, vmax = 0.0, 1.0
    elif adv:
        results = [(l, c, tr, a, q - baseline[:, None], qm[:T] - baseline) for l, c, tr, a, q, qm in results]
        m = np.percentile(np.abs(torch.cat([r[4].flatten() for r in results]).numpy()), args.color_percentiles[1])
        vmin, vmax = -m, m
    else:
        vmin, vmax = np.percentile(torch.cat([r[4].flatten() for r in results]).numpy(), args.color_percentiles)
    if not per_state:
        print(f"shared color range ({args.color_by}): [{vmin:.2f}, {vmax:.2f}]", flush=True)
        results = [r + (r[4],) for r in results]
    os.makedirs(args.out_dir, exist_ok=True)
    qname = "own" if args.q_source == "own" else os.path.basename(os.path.dirname(args.q_checkpoint))
    qtag = "Q_π" if args.q_source == "own" else "Q_exp"
    qwho = "policy's own" if args.q_source == "own" else "expert"
    overlay_default = args.overlay_ref
    for label, ckpt, traj, a, q, qm, color in results:
        args.label = label
        args.overlay_ref = overlay_default and not label.startswith("pre-finetune")
        if args.overlay_expert and os.path.abspath(ckpt) != os.path.abspath(args.q_checkpoint):
            traj = dict(traj, mu_exp=mu_ref.cpu(), sigma_exp=log_std_ref.exp().cpu())
        q, qm, color = q.float(), qm.float(), color.float()
        name = label.replace(" ", "_").replace("+", "")
        out = os.path.join(args.out_dir, f"{dict(q_per_state='qstate', advantage='adv').get(args.color_by, 'q')}_{qname}__{name}.mp4")
        torch.save(dict(policy=ckpt, q_checkpoint=args.q_checkpoint, actions=a, q=q, q_mean_action=qm, mu=traj["mu"], sigma=traj["sigma"],
                        vmin=vmin, vmax=vmax), out.replace(".mp4", ".pt"))
        if per_state:
            render(args, traj, a, color, T, out, vmin=0.0, vmax=1.0, cbar_label=f"{qtag}(s,a): low → high among this policy's samples at this state",
                   header=f"π(a|s_t) per dim (x = action value); dots = {args.branches} sampled actions, color = {qwho} Q(s,a), per-state scale",
                   summary=lambda t, q=q: f"mean {qtag} {q[t].mean():.2f}")
        elif adv:
            render(args, traj, a, q, T, out, vmin=vmin, vmax=vmax, cbar_label="Q_exp(s,a) − Q_exp(s, a_expert)",
                   header=f"π(a|s_t) per dimension (x = action value); dots = {args.branches} sampled actions, color = expert-critic advantage vs expert's action",
                   summary=lambda t, q=q, qm=qm: f"mean A {q[t].mean():+.2f} (π mean {qm[t]:+.2f})")
        else:
            render(args, traj, a, q, T, out, vmin=vmin, vmax=vmax, cbar_label="Q_expert(s, a)",
                   header=f"π(a|s_t) per dimension (x = action value); dots = {args.branches} sampled actions, color = stem-tip expert's Q(s, a)",
                   summary=lambda t, q=q, qm=qm: f"mean Q {q[t].mean():.1f} (π mean action {qm[t]:.1f})")
    os._exit(0)


if __name__ == "__main__":
    main(tyro.cli(Args))
