"""Side-by-side PushT2Goal-v1 video: each panel plays N stochastic eval episodes of one checkpoint back to back
(same start states via the same env seed), labelled with episode index and outcome.

    python reinforcement_learning/scripts/video_2goal.py --ckpts runs/a/ckpt.pt runs/b/ckpt.pt --labels MaxEntDP FastSAC \
        --modes sample sample --num-episodes 10 --out runs/viz_2goal/side_by_side.mp4
"""
from dataclasses import dataclass
from typing import List

import gymnasium as gym
import imageio
import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw, ImageFont

import mani_skill.envs  # noqa: F401
from reinforcement_learning.scripts.viz_2goal import make_policy
from reinforcement_learning.utils.args import parse_env_kwargs

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
OUTCOME_COLOR = {"goal A": (42, 120, 214), "goal B": (235, 104, 52), "neither": (110, 110, 105)}


@dataclass
class Args:
    ckpts: List[str]
    labels: List[str]
    modes: List[str]
    num_episodes: int = 10
    num_steps: int = 100
    seed: int = 0
    out: str = "runs/viz_2goal/side_by_side.mp4"
    fps: int = 30


@torch.no_grad()
def rollout_frames(path, mode, args):
    torch.manual_seed(args.seed)
    ckpt = torch.load(path, weights_only=False)
    cfg = ckpt["config"]["env"]
    kwargs = {k: v for k, v in parse_env_kwargs(cfg.get("kwargs", {})).items() if k not in ("reset_datasets", "reset_dataset_weights", "valid_goals")}
    env = gym.make("PushT2Goal-v1", num_envs=args.num_episodes, obs_mode=cfg["obs_mode"], control_mode=cfg["control_mode"],
                   reward_mode="sparse", render_mode="rgb_array", **kwargs)
    e = env.unwrapped
    policy, _ = make_policy(ckpt, e, torch.device("cuda"), mode)
    obs, _ = env.reset(seed=args.seed)
    frames = [env.render().cpu().numpy()]
    hit_a = torch.zeros(args.num_episodes, dtype=torch.bool, device=e.device)
    hit_b = torch.zeros_like(hit_a)
    for _ in range(args.num_steps):
        obs, _, _, _, info = env.step(policy(obs))
        hit_a |= info["success_a"]
        hit_b |= info["success_b"]
        frames.append(env.render().cpu().numpy())
    env.close()
    outcome = ["goal A" if a else "goal B" if b else "neither" for a, b in zip(hit_a.tolist(), hit_b.tolist())]
    return np.stack(frames, 1), outcome  # (N, T+1, H, W, 3)


def annotate(frame, label, ep, n, outcome, show_outcome, font, small):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 34], fill=(252, 252, 251))
    d.text((10, 6), label, fill=(11, 11, 11), font=font)
    d.text((img.width - 150, 9), f"episode {ep + 1}/{n}", fill=(82, 81, 78), font=small)
    if not show_outcome:
        d.rectangle([0, img.height - 30, img.width, img.height], fill=(252, 252, 251))
        d.text((10, img.height - 25), "goal B = blue (left)   goal A = gray (right)", fill=(82, 81, 78), font=small)
    else:
        d.rectangle([0, img.height - 34, img.width, img.height], fill=OUTCOME_COLOR[outcome])
        d.text((10, img.height - 28), f"reached: {outcome}", fill=(255, 255, 255), font=font)
    return np.asarray(img)


def main(args: Args):
    font, small = ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT, 16)
    panels = []
    for path, label, mode in zip(args.ckpts, args.labels, args.modes):
        frames, outcome = rollout_frames(path, mode, args)
        print(f"{label}: {outcome}", flush=True)
        seq = []
        for ep in range(args.num_episodes):
            for t in range(frames.shape[1]):
                seq.append(annotate(frames[ep, t], label, ep, args.num_episodes, outcome[ep], t >= frames.shape[1] - 1, font, small))
            seq.extend([seq[-1]] * (args.fps // 2))  # hold the final frame with the outcome banner
        panels.append(seq)
    gap = np.full((panels[0][0].shape[0], 8, 3), 255, np.uint8)
    with imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8) as w:
        for fr in zip(*panels):
            w.append_data(np.concatenate([fr[0], gap, fr[1]], axis=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
