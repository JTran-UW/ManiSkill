"""Roll out reference policies from every recorded training episode's initial state; compose a side-by-side video.

Needs a training run made with --capture-video --save-train-video-freq 1 --save-train-trajectory.

    python reinforcement_learning/scripts/counterfactual_rollouts.py --run-dir runs/<exp> \
        --policies "pre-finetune=runs/fastsac_pusht_control/final_ckpt.pt" "target expert=runs/pusht_stem_tip_ballast/final_ckpt.pt"
"""
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import List

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import tyro

from reinforcement_learning.utils.policies import load_fastsac_policy

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BOX = "fontsize=20:box=1:boxcolor=black@0.55:boxborderw=6"


@dataclass
class Args:
    run_dir: str
    policies: List[str] = field(default_factory=lambda: [
        "pre-finetune=runs/fastsac_pusht_control/final_ckpt.pt",
        "target expert=runs/pusht_stem_tip_ballast/final_ckpt.pt"])
    """LABEL=CHECKPOINT pairs, one panel each (after the training panel)"""
    train_label: str = "training (online)"
    deterministic: bool = True
    compose: bool = True


def initial_states(h5):
    keys = sorted(h5.keys(), key=lambda k: int(k.split("_")[1]))
    states = {}
    for group in ("actors", "articulations"):
        if group in h5[keys[0]]["env_states"]:
            names = h5[keys[0]]["env_states"][group].keys()
            states[group] = {n: torch.from_numpy(np.stack([h5[k]["env_states"][group][n][0] for k in keys])) for n in names}
    success = [bool(np.any(h5[k]["success"][()])) for k in keys]
    return keys, states, success


def rollout(env, states, policy, steps, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    base = env.unwrapped
    obs, _ = env.reset(options=dict(reset_to_env_states=dict(env_states={g: {n: v.to(base.device) for n, v in d.items()} for g, d in states.items()})))
    tee = states["actors"]["Tee"][:, :2].to(base.device)
    assert (base.tee.pose.p[:, :2] - tee).abs().max() < 1e-5, "initial state not restored"
    writers = [imageio.get_writer(f"{out_dir}/{i}.mp4", fps=30, codec="libx264", quality=8) for i in range(base.num_envs)]
    success = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)

    def write():
        for w, frame in zip(writers, env.render().cpu().numpy().astype(np.uint8)):
            w.append_data(frame)

    write()
    for _ in range(steps):
        obs, _, _, _, info = env.step(policy(obs))
        success |= info["success"]
        write()
    for w in writers:
        w.close()
    return success.cpu().tolist()


def compose(args, labels, successes, n, out):
    tallies = [np.cumsum(s) for s in successes]
    parts = []
    for i in range(n):
        inputs, filters = [], []
        for j, label in enumerate(labels):
            path = f"{args.run_dir}/train_videos/{i}.mp4" if j == 0 else f"{args.run_dir}/counterfactual/{label.replace(' ', '_')}/{i}.mp4"
            s = successes[j][i]
            inputs += ["-i", path]
            filters.append(
                f"[{j}:v]drawtext=fontfile={FONT}:text='{label}':fontcolor=white:{BOX}:x=12:y=12,"
                f"drawtext=fontfile={FONT}:text='{'SUCCESS' if s else 'FAIL'}':fontcolor={'0x33ff33' if s else '0xff4444'}:{BOX}:x=12:y=48,"
                f"drawtext=fontfile={FONT}:text='{tallies[j][i]}/{i + 1} so far':fontcolor=white:{BOX}:x=w-tw-12:y=12"
                + (f",drawtext=fontfile={FONT}:text='episode {i + 1}/{n}':fontcolor=white:{BOX}:x=12:y=h-th-18" if j == 0 else "")
                + f"[v{j}]")
        graph = ";".join(filters) + ";" + "".join(f"[v{j}]" for j in range(len(labels))) + f"hstack=inputs={len(labels)}[out]"
        part = f"{args.run_dir}/counterfactual/part_{i}.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[out]",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", part], check=True)
        parts.append(part)
    lst = f"{args.run_dir}/counterfactual/concat.txt"
    with open(lst, "w") as f:
        f.writelines(f"file '{os.path.abspath(p)}'\n" for p in parts)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out], check=True)
    for p in parts + [lst]:
        os.remove(p)


def main(args: Args):
    traj = f"{args.run_dir}/train_videos/trajectory"
    env_info = json.load(open(f"{traj}.json"))["env_info"]
    with h5py.File(f"{traj}.h5") as h5:
        keys, states, train_success = initial_states(h5)
    n = len(keys)
    videos = len([f for f in os.listdir(f"{args.run_dir}/train_videos") if f.endswith(".mp4")])
    assert videos == n, f"{videos} training videos vs {n} recorded trajectories"
    kwargs = {k: v for k, v in env_info["env_kwargs"].items() if k not in ("num_envs", "render_mode", "sim_backend")}
    env = gym.make(env_info["env_id"], num_envs=n, render_mode="rgb_array", sim_backend="gpu", **kwargs)
    steps = env_info["max_episode_steps"]

    labels, successes = [args.train_label], [train_success]
    for spec in args.policies:
        label, ckpt = spec.split("=", 1)
        policy = load_fastsac_policy(ckpt, env.unwrapped, deterministic=args.deterministic)
        s = rollout(env, states, policy, steps, f"{args.run_dir}/counterfactual/{label.replace(' ', '_')}")
        labels.append(label)
        successes.append(s)
        print(f"{label}: {sum(s)}/{n} episodes succeeded ({ckpt})", flush=True)
    print(f"{args.train_label}: {sum(train_success)}/{n} episodes succeeded", flush=True)
    json.dump(dict(labels=labels, success=successes, policies=args.policies, deterministic=args.deterministic),
              open(f"{args.run_dir}/counterfactual/success.json", "w"), indent=1)
    env.close()
    if args.compose:
        out = f"{args.run_dir}/counterfactual_side_by_side_{n}ep.mp4"
        compose(args, labels, successes, n, out)
        print("wrote", out)
    os._exit(0)


if __name__ == "__main__":
    main(tyro.cli(Args))
