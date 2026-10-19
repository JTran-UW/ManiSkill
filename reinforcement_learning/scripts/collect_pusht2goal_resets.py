"""Collect PushT2Goal-v1 reset datasets (tee_xy, tee_yaw, robot_qpos + full sim states).

    near_goal_a / near_goal_b:  T = goal A / B pose + small offset (optionally a solved fraction), tip near the T
    between_arm_near:           T anywhere between the goals (x = start +- between_x, |y| <= between_y, yaw = start +- between_yaw), tip near the T
    between_anywhere:           same T distribution, tip anywhere in the workspace (not on the T)

Rejects candidates that start solved (either goal; for --valid solved states: the target goal), miss IK, or move the
T while the arm holds still. Also writes default_resets.png and grid.png per dataset and prints default-reset stats.

    python reinforcement_learning/scripts/collect_pusht2goal_resets.py
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
from PIL import Image, ImageDraw

from mani_skill.envs.tasks.tabletop.push_t_2goal import PushT2GoalEnv, solve_tip_ik
from mani_skill.utils.structs.pose import Pose
from reinforcement_learning.scripts.collect_pusht_goal_resets import dist_to_tee, sample_tip, to_local  # noqa: F401

KINDS = ("near_goal_a", "near_goal_b", "between_arm_near", "between_anywhere")


@dataclass
class Args:
    kinds: List[str] = field(default_factory=lambda: list(KINDS))
    num_states: int = 4096
    batch: int = 4096
    seed: int = 0
    near_goal_xy: float = 0.04
    near_goal_yaw_deg: float = 30.0
    near_goal_solved_frac: float = 0.25
    solved_margin: float = 0.01
    solved_xy: float = 0.015
    solved_yaw_deg: float = 10.0
    between_x: float = 0.05
    between_y: float = 0.18
    between_yaw_deg: float = 45.0
    arm_near_min: float = 0.01
    arm_near_max: float = 0.05
    workspace_x: List[float] = field(default_factory=lambda: [-0.36, 0.05])
    workspace_y: List[float] = field(default_factory=lambda: [-0.30, 0.30])
    tip_clearance: float = 0.01
    ik_tol: float = 0.005
    settle_steps: int = 10
    tee_move_tol: float = 0.002
    tee_yaw_tol_deg: float = 1.0
    out_root: str = "runs/reset_datasets_2goal"


def sample_tee(kind, e: PushT2GoalEnv, b, args, want_solved):
    dev = e.device
    if kind.startswith("near_goal"):
        g = e._goal_xy[0 if kind.endswith("a") else 1]
        max_xy, max_yaw = (args.solved_xy, args.solved_yaw_deg) if want_solved else (args.near_goal_xy, args.near_goal_yaw_deg)
        r = max_xy * torch.rand(b, device=dev).sqrt()
        th = torch.rand(b, device=dev) * 2 * math.pi
        xy = g + torch.stack([r * th.cos(), r * th.sin()], -1)
    else:
        max_yaw = args.between_yaw_deg
        xy = e._start_xy + (torch.rand(b, 2, device=dev) * 2 - 1) * torch.tensor([args.between_x, args.between_y], device=dev)
    yaw = e.START_YAW + (torch.rand(b, device=dev) * 2 - 1) * math.radians(max_yaw)
    return xy, yaw


def collect(kind, args, env, num_states, want_solved=False):
    e: PushT2GoalEnv = env.unwrapped
    dev, b = e.device, args.batch
    tip_kind = "anywhere" if kind == "between_anywhere" else "near"
    kept, n_kept = [], 0
    stats = dict(candidates=0, start_check=0, no_tip=0, ik_miss=0, moved=0, tip_drift=0)
    target = 0 if kind.endswith("_a") else 1
    while n_kept < num_states:
        env.reset()
        tee_xy, tee_yaw = sample_tee(kind, e, b, args, want_solved)
        tee_p = torch.cat([tee_xy, torch.full((b, 1), 0.04 / 2 + 1e-3, device=dev)], 1)
        tee_q = e._yaw_to_quat(tee_yaw)
        tip = sample_tip(tip_kind, tee_xy, tee_yaw, args)
        no_tip = torch.isnan(tip).any(1)
        q, ik_err = solve_tip_ik(e, torch.nan_to_num(tip))
        e.tee.set_pose(Pose.create_from_pq(tee_p, tee_q))
        e.tee.set_linear_velocity(torch.zeros((b, 3), device=dev))
        e.tee.set_angular_velocity(torch.zeros((b, 3), device=dev))
        e.agent.robot.set_qpos(q)
        e.agent.robot.set_qvel(torch.zeros_like(q))
        e.scene._gpu_apply_all()
        e.scene.px.gpu_update_articulation_kinematics()
        e.scene._gpu_fetch_all()
        for _ in range(args.settle_steps):
            env.step(q)
        moved_xy = torch.linalg.norm(e.tee.pose.p[:, :2] - tee_xy, dim=1)
        moved_yaw = ((e.quat_to_z_euler(e.tee.pose.q) - tee_yaw + math.pi) % (2 * math.pi) - math.pi).abs()
        moved = (moved_xy > args.tee_move_tol) | (moved_yaw > math.radians(args.tee_yaw_tol_deg))
        tip_drift = torch.linalg.norm(e.agent.tcp.pose.p[:, :2] - torch.nan_to_num(tip), dim=1) > args.ik_tol
        ov0, ov1 = e.goal_overlaps(tee_p, tee_q), e.goal_overlaps()
        if want_solved:
            bad_start = (ov0[target] < e.intersection_thresh + args.solved_margin) | (ov1[target] < e.intersection_thresh + args.solved_margin)
        else:
            bad_start = torch.stack([o >= e.intersection_thresh - args.solved_margin for o in ov0 + ov1]).any(0)
        ok = torch.ones(b, dtype=torch.bool, device=dev)
        for name, m in (("start_check", bad_start), ("no_tip", no_tip), ("ik_miss", ik_err > args.ik_tol), ("moved", moved), ("tip_drift", tip_drift)):
            stats[name] += int((m & ok).sum())
            ok &= ~m
        stats["candidates"] += b
        idx = ok.nonzero().squeeze(1)[: num_states - n_kept]
        if len(idx):
            state = e.get_state_dict()
            kept.append(dict(env_state={g: {k: v[idx].cpu() for k, v in d.items()} for g, d in state.items()},
                             tee_xy=tee_xy[idx].cpu(), tee_yaw=tee_yaw[idx].cpu(), tip_xy=tip[idx].cpu(), robot_qpos=q[idx].cpu()))
            n_kept += len(idx)
        print(f"[{kind}{' solved' if want_solved else ''}] kept {len(idx)}/{b} total {n_kept}/{num_states}", flush=True)
    out = {k: torch.cat([c[k] for c in kept]) for k in ("tee_xy", "tee_yaw", "tip_xy", "robot_qpos")}
    out["env_state"] = {g: {k: torch.cat([c["env_state"][g][k] for c in kept]) for k in kept[0]["env_state"][g]} for g in kept[0]["env_state"]}
    out["start_solved"] = torch.full((n_kept,), want_solved)
    return out, stats


def shuffle_cat(parts, seed):
    n = sum(len(p["tee_xy"]) for p in parts)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    out = {}
    for k in parts[0]:
        if k == "env_state":
            out[k] = {g: {kk: torch.cat([p[k][g][kk] for p in parts])[perm] for kk in parts[0][k][g]} for g in parts[0][k]}
        else:
            out[k] = torch.cat([p[k] for p in parts])[perm]
    return out


def render_tiles(env, path, n=16, states=None):
    env.reset(seed=0, options=dict(reset_to_env_states=dict(env_states=states)) if states is not None else None)
    frames = env.render().cpu().numpy().astype(np.uint8)[:n]
    h, w = frames.shape[1:3]
    grid = Image.new("RGB", (4 * w, math.ceil(len(frames) / 4) * h))
    for i, f in enumerate(frames):
        tile = Image.fromarray(f)
        ImageDraw.Draw(tile).text((8, 8), f"{i}", fill=(0, 0, 0))
        grid.paste(tile, ((i % 4) * w, (i // 4) * h))
    grid.save(path)


def main(args: Args):
    torch.manual_seed(args.seed)
    os.makedirs(args.out_root, exist_ok=True)
    # default-reset sanity check
    env = gym.make("PushT2Goal-v1", num_envs=256, obs_mode="state", control_mode="pd_ee_delta_pose", reward_mode="sparse", render_mode="rgb_array")
    obs, info = env.reset(seed=args.seed)
    e = env.unwrapped
    _, ik_err = solve_tip_ik(e, torch.tensor([[-0.30, 0.0]], device=e.device))
    print(f"[default] obs dim {obs.shape[-1]}, default qpos {e._default_qpos.tolist()}, IK err {ik_err.item():.2e}")
    print(f"[default] tip xy mean {e.agent.tcp.pose.p[:, :2].mean(0).tolist()} std {e.agent.tcp.pose.p[:, :2].std(0).tolist()}")
    ov = e.goal_overlaps()
    print(f"[default] start overlap A {ov[0].mean():.3f} B {ov[1].mean():.3f}, success {info['success'].float().mean():.3f}")
    zero = torch.zeros(env.action_space.shape, device=e.device)
    for _ in range(20):
        obs, r, *_ , info = env.step(zero)
    print(f"[default] after 20 zero-action steps: T moved {torch.linalg.norm(e.tee.pose.p[:, :2] - e._start_xy, dim=1).mean():.4f} m (incl. start noise), reward {r.mean():.3f}")
    render_tiles(env, f"{args.out_root}/default_resets.png")
    env.close()

    env = gym.make("PushT2Goal-v1", num_envs=args.batch, obs_mode="state", control_mode="pd_joint_pos", reward_mode="sparse")
    env.reset(seed=args.seed)
    paths = []
    for kind in args.kinds:
        assert kind in KINDS, kind
        n_solved = round(args.near_goal_solved_frac * args.num_states) if kind.startswith("near_goal") else 0
        data, stats = collect(kind, args, env, args.num_states - n_solved)
        if n_solved:
            sdata, sstats = collect(kind, args, env, n_solved, want_solved=True)
            data, stats = shuffle_cat([data, sdata], args.seed), dict(unsolved=stats, solved=sstats)
        tag = f"_succ{round(100 * args.near_goal_solved_frac)}" if n_solved else ""
        out_dir = f"{args.out_root}/PushT2Goal-v1_{kind}{tag}_{args.num_states}_seed{args.seed}"
        os.makedirs(out_dir, exist_ok=True)
        data.update(kind=kind, stats=stats, config=vars(args))
        torch.save(data, f"{out_dir}/reset_states.pt")
        paths.append(out_dir)
        print(f"[{kind}] saved {out_dir}/reset_states.pt stats {stats}", flush=True)
    env.close()
    env = gym.make("PushT2Goal-v1", num_envs=16, obs_mode="state", control_mode="pd_joint_pos", reward_mode="sparse", render_mode="rgb_array")
    for out_dir in paths:
        d = torch.load(f"{out_dir}/reset_states.pt", weights_only=False)
        render_tiles(env, f"{out_dir}/grid.png", states={g: {k: v[:16] for k, v in s.items()} for g, s in d["env_state"].items()})
    env.close()
    print("done", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
