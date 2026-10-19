"""Collect PushTGoal-v1 reset datasets (full sim states) for three reset distributions.

    near_goal:            T = goal pose + small offset, stick tip near the T
    t_anywhere_arm_near:  T from the PushT spawn distribution, stick tip near the T
    anywhere:             T from the spawn distribution, stick tip anywhere in the workspace (not on the T)

Goals follow PushTGoal-v1's goal distribution in all three. Candidates are rejected if the T already satisfies its goal,
IK misses the tip target, or the T moves while the arm holds still for a few steps (arm-T collision / bad IK).

    python reinforcement_learning/scripts/collect_pusht_goal_resets.py --kinds near_goal t_anywhere_arm_near anywhere
"""
import math
import os
from dataclasses import dataclass, field
from typing import List

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw, ImageFont

from mani_skill.agents.controllers.utils.kinematics import Kinematics
from mani_skill.envs.tasks.tabletop.push_t_goal import PushTGoalEnv
from mani_skill.utils.structs.pose import Pose

# T footprint in the T frame (two boxes: [xmin, xmax, ymin, ymax]), matching push_t.py's create_tee
TEE_BOXES = torch.tensor([[-0.1, 0.1, -0.0625, -0.0125], [-0.025, 0.025, -0.0125, 0.1375]])
KINDS = ("near_goal", "t_anywhere_arm_near", "anywhere")


@dataclass
class Args:
    kinds: List[str] = field(default_factory=lambda: list(KINDS))
    num_states: int = 4096
    """states to keep per dataset"""
    batch: int = 4096
    """candidates simulated in parallel per round"""
    seed: int = 0
    near_goal_xy: float = 0.04
    """near_goal: T xy offset from the goal, uniform in a disk of this radius (m)"""
    near_goal_yaw_deg: float = 30.0
    """near_goal: T yaw offset from the goal, uniform in +-this (deg)"""
    near_goal_solved_frac: float = 0.0
    """near_goal: fraction of states that start in the success configuration (T covers >= 90% of the goal)"""
    solved_margin: float = 0.01
    """overlap margin around the success threshold: solved states need >= thresh + margin, others <= thresh - margin"""
    solved_xy: float = 0.015
    solved_yaw_deg: float = 10.0
    """near_goal solved states: candidate T offsets (disk radius m, +-deg) before keeping only those that satisfy the goal"""
    arm_near_min: float = 0.01
    arm_near_max: float = 0.05
    """arm-near datasets: stick tip distance from the T outline (m)"""
    workspace_x: List[float] = field(default_factory=lambda: [-0.36, 0.05])
    workspace_y: List[float] = field(default_factory=lambda: [-0.30, 0.30])
    """anywhere: stick tip xy range (world frame)"""
    tip_clearance: float = 0.01
    """anywhere: minimum stick tip distance from the T outline (m)"""
    ik_iters: int = 100
    ik_tol: float = 0.005
    settle_steps: int = 10
    tee_move_tol: float = 0.002
    tee_yaw_tol_deg: float = 1.0
    out_root: str = "runs/reset_datasets"


def dist_to_tee(xy_local: torch.Tensor) -> torch.Tensor:
    """Distance from points (..., 2) in the T frame to the T footprint (0 inside)."""
    boxes = TEE_BOXES.to(xy_local.device)
    x, y = xy_local[..., 0:1], xy_local[..., 1:2]
    dx = torch.clamp(torch.maximum(boxes[:, 0] - x, x - boxes[:, 1]), min=0)
    dy = torch.clamp(torch.maximum(boxes[:, 2] - y, y - boxes[:, 3]), min=0)
    return torch.sqrt(dx ** 2 + dy ** 2).min(dim=-1).values


def to_local(xy: torch.Tensor, tee_xy: torch.Tensor, tee_yaw: torch.Tensor) -> torch.Tensor:
    """World xy (B, K, 2) -> T frame for per-row T poses (B, 2), (B,)."""
    d = xy - tee_xy[:, None]
    c, s = tee_yaw.cos()[:, None], tee_yaw.sin()[:, None]
    return torch.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], dim=-1)


def sample_tip(kind, tee_xy, tee_yaw, args, k=256):
    """Per-row stick tip xy satisfying the kind's distance rule; rows with no hit get NaN."""
    b, dev = tee_xy.shape[0], tee_xy.device
    if kind == "anywhere":
        lo = torch.tensor([args.workspace_x[0], args.workspace_y[0]], device=dev)
        hi = torch.tensor([args.workspace_x[1], args.workspace_y[1]], device=dev)
        cand = lo + torch.rand(b, k, 2, device=dev) * (hi - lo)
        ok = dist_to_tee(to_local(cand, tee_xy, tee_yaw)) >= args.tip_clearance
    else:
        r = args.arm_near_max + 0.2
        local = (torch.rand(b, k, 2, device=dev) * 2 - 1) * r
        d = dist_to_tee(local)
        ok = (d >= args.arm_near_min) & (d <= args.arm_near_max)
        c, s = tee_yaw.cos()[:, None], tee_yaw.sin()[:, None]
        cand = tee_xy[:, None] + torch.stack([c * local[..., 0] - s * local[..., 1], s * local[..., 0] + c * local[..., 1]], -1)
    first = torch.where(ok.any(1), ok.float().argmax(1), torch.full((b,), -1, device=dev))
    tip = cand[torch.arange(b, device=dev), first.clamp(min=0)]
    tip[first < 0] = float("nan")
    return tip


def sample_tee(kind, goal_xy, goal_yaw, e: PushTGoalEnv, args, want_solved=False):
    b, dev = goal_xy.shape[0], goal_xy.device
    if kind == "near_goal":
        max_xy, max_yaw = (args.solved_xy, args.solved_yaw_deg) if want_solved else (args.near_goal_xy, args.near_goal_yaw_deg)
        r = max_xy * torch.rand(b, device=dev).sqrt()
        th = torch.rand(b, device=dev) * 2 * math.pi
        xy = goal_xy + torch.stack([r * th.cos(), r * th.sin()], -1)
        yaw = goal_yaw + (torch.rand(b, device=dev) * 2 - 1) * math.radians(max_yaw)
        return xy, torch.remainder(yaw, 2 * math.pi)
    return e._sample_tee_xy_yaw(b)


def solve_ik(e, kin, tip_xy, rest_qpos, tip_z, tip_quat, args):
    root = e.agent.robot.pose.p[:1]
    b = tip_xy.shape[0]
    target_p = torch.cat([tip_xy, torch.full((b, 1), tip_z, device=tip_xy.device)], 1) - root
    target = Pose.create_from_pq(target_p, tip_quat.expand(b, 4))
    lim = torch.tensor(kin.pk_chain.get_joint_limits(), device=tip_xy.device)
    q = rest_qpos.expand(b, -1).clone()
    for _ in range(args.ik_iters):
        q = torch.clamp(kin.compute_ik(target, q), lim[0], lim[1])
    tf = kin.pk_chain.forward_kinematics(q).get_matrix()
    err = torch.linalg.norm(tf[:, :3, 3] - target_p, dim=1)
    return q, err


def collect(kind, args, env, kin, rest_qpos, tip_z, tip_quat, num_states, want_solved=False):
    e: PushTGoalEnv = env.unwrapped
    dev = e.device
    kept, stats = [], dict(candidates=0, solved_at_start=0, not_solved_at_start=0, no_tip=0, ik_miss=0, moved=0, tip_drift=0, kept=0)
    n_kept = 0
    while n_kept < num_states:
        env.reset()
        b = args.batch
        goal_xy, goal_yaw = e._sample_tee_xy_yaw(b)
        tee_xy, tee_yaw = sample_tee(kind, goal_xy, goal_yaw, e, args, want_solved)
        w2g = e._world_to_goal(goal_xy, goal_yaw)
        tee_p = torch.cat([tee_xy, torch.full((b, 1), 0.04 / 2 + 1e-3, device=dev)], 1)
        margin = args.solved_margin if want_solved else -args.solved_margin
        solved = e.pseudo_render_intersection(tee_p, e._yaw_to_quat(tee_yaw), w2g) >= e.intersection_thresh + margin
        tip = sample_tip(kind, tee_xy, tee_yaw, args)
        no_tip = torch.isnan(tip).any(1)
        q, ik_err = solve_ik(e, kin, torch.nan_to_num(tip), rest_qpos, tip_z, tip_quat, args)
        ik_miss = ik_err > args.ik_tol

        e.goal_tee.set_pose(Pose.create_from_pq(torch.cat([goal_xy, torch.full((b, 1), 1e-3, device=dev)], 1), e._yaw_to_quat(goal_yaw)))
        e.tee.set_pose(Pose.create_from_pq(tee_p, e._yaw_to_quat(tee_yaw)))
        e.tee.set_linear_velocity(torch.zeros((b, 3), device=dev))
        e.tee.set_angular_velocity(torch.zeros((b, 3), device=dev))
        e.agent.robot.set_qpos(q)
        e.agent.robot.set_qvel(torch.zeros_like(q))
        if e.gpu_sim_enabled:
            e.scene._gpu_apply_all()
            e.scene.px.gpu_update_articulation_kinematics()
            e.scene._gpu_fetch_all()
        for _ in range(args.settle_steps):
            env.step(q)
        moved_xy = torch.linalg.norm(e.tee.pose.p[:, :2] - tee_xy, dim=1)
        moved_yaw = ((e.quat_to_z_euler(e.tee.pose.q) - tee_yaw + math.pi) % (2 * math.pi) - math.pi).abs()
        moved = (moved_xy > args.tee_move_tol) | (moved_yaw > math.radians(args.tee_yaw_tol_deg))
        tip_drift = torch.linalg.norm(e.agent.tcp.pose.p[:, :2] - torch.nan_to_num(tip), dim=1) > args.ik_tol

        solved_after = e.pseudo_render_intersection(e.tee.pose.p, e.tee.pose.q, w2g) >= e.intersection_thresh + margin
        start_check = ("not_solved_at_start", ~(solved & solved_after)) if want_solved else ("solved_at_start", solved | solved_after)
        bad = [start_check, ("no_tip", no_tip), ("ik_miss", ik_miss), ("moved", moved), ("tip_drift", tip_drift)]
        ok = torch.ones(b, dtype=torch.bool, device=dev)
        for name, m in bad:
            stats[name] += int((m & ok).sum())
            ok &= ~m
        stats["candidates"] += b
        idx = ok.nonzero().squeeze(1)[: num_states - n_kept]
        if len(idx):
            state = e.get_state_dict()
            kept.append(dict(
                env_state={g: {k: v[idx].cpu() for k, v in d.items()} for g, d in state.items()},
                tee_xy=tee_xy[idx].cpu(), tee_yaw=tee_yaw[idx].cpu(),
                goal_xy=goal_xy[idx].cpu(), goal_yaw=goal_yaw[idx].cpu(),
                tip_xy=tip[idx].cpu(), robot_qpos=q[idx].cpu(),
            ))
            n_kept += len(idx)
        print(f"[{kind}{' solved' if want_solved else ''}] round: kept {len(idx)}/{b}  total {n_kept}/{num_states}", flush=True)
    stats["kept"] = n_kept

    def cat(key):
        return torch.cat([k[key] for k in kept])
    env_state = {g: {k: torch.cat([c["env_state"][g][k] for c in kept]) for k in kept[0]["env_state"][g]} for g in kept[0]["env_state"]}
    out = dict(env_state=env_state, **{k: cat(k) for k in ("tee_xy", "tee_yaw", "goal_xy", "goal_yaw", "tip_xy", "robot_qpos")})
    out["start_solved"] = torch.full((n_kept,), want_solved)
    return out, stats


def concat_shuffled(parts, seed):
    n = sum(len(p["tee_xy"]) for p in parts)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    out = {}
    for k in parts[0]:
        if k == "env_state":
            out[k] = {g: {kk: torch.cat([p[k][g][kk] for p in parts])[perm] for kk in parts[0][k][g]} for g in parts[0][k]}
        else:
            out[k] = torch.cat([p[k] for p in parts])[perm]
    return out


def dataset_dir(args, kind):
    tag = f"_succ{round(100 * args.near_goal_solved_frac)}" if kind == "near_goal" and args.near_goal_solved_frac > 0 else ""
    return f"{args.out_root}/PushTGoal-v1_{kind}{tag}_{args.num_states}_seed{args.seed}"


def render_grid(path, data, n=16, seed=0):
    n = min(n, len(data["tee_xy"]))
    env = gym.make("PushTGoal-v1", num_envs=n, obs_mode="state", control_mode="pd_joint_pos", render_mode="rgb_array")
    states = {g: {k: v[:n] for k, v in d.items()} for g, d in data["env_state"].items()}
    env.reset(seed=seed, options=dict(reset_to_env_states=dict(env_states=states)))
    frames = env.render().cpu().numpy().astype(np.uint8)
    env.close()
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    h, w = frames.shape[1:3]
    cols = 4
    grid = Image.new("RGB", (cols * w, math.ceil(n / cols) * h))
    for i, f in enumerate(frames):
        tile = Image.fromarray(f)
        ImageDraw.Draw(tile).text((8, 8), f"{i}", fill=(0, 0, 0), font=font)
        grid.paste(tile, ((i % cols) * w, (i // cols) * h))
    grid.save(path)


def main(args: Args):
    torch.manual_seed(args.seed)
    env = gym.make("PushTGoal-v1", num_envs=args.batch, obs_mode="state", control_mode="pd_joint_pos")
    env.reset(seed=args.seed)
    e: PushTGoalEnv = env.unwrapped
    rest_qpos = e.agent.robot.get_qpos()[:1].clone()
    tip_z = float(e.agent.tcp.pose.p[0, 2])
    tip_quat = e.agent.tcp.pose.q[:1].clone()
    arm = e.agent.controller.controllers["arm"]
    kin = Kinematics(e.agent.urdf_path, e.agent.ee_link_name, e.agent.robot, arm.active_joint_indices)
    for kind in args.kinds:
        assert kind in KINDS, kind
        n_solved = round(args.near_goal_solved_frac * args.num_states) if kind == "near_goal" else 0
        data, stats = collect(kind, args, env, kin, rest_qpos, tip_z, tip_quat, args.num_states - n_solved)
        if n_solved:
            solved_data, solved_stats = collect(kind, args, env, kin, rest_qpos, tip_z, tip_quat, n_solved, want_solved=True)
            data = concat_shuffled([data, solved_data], args.seed)
            stats = dict(unsolved=stats, solved=solved_stats)
        out_dir = dataset_dir(args, kind)
        os.makedirs(out_dir, exist_ok=True)
        data.update(kind=kind, stats=stats, config=vars(args))
        torch.save(data, f"{out_dir}/reset_states.pt")
        print(f"[{kind}] saved {out_dir}/reset_states.pt  stats {stats}", flush=True)
    env.close()
    for kind in args.kinds:
        out_dir = dataset_dir(args, kind)
        render_grid(f"{out_dir}/grid.png", torch.load(f"{out_dir}/reset_states.pt", weights_only=False))
        print(f"[{kind}] wrote {out_dir}/grid.png", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
