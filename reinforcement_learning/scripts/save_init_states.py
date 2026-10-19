"""Sample and save PushT initial states (full sim state + tee pose / robot qpos), plus a labeled image of each.

    python reinforcement_learning/scripts/save_init_states.py --env.id PushT-v1 --num-states 10 --seed 0 --name pusht_10
"""
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw, ImageFont

from reinforcement_learning.utils.args import EnvArgs, config_dict, gym_make_kwargs


@dataclass
class Args:
    env: EnvArgs = field(default_factory=lambda: EnvArgs(id="PushT-v1"))
    num_states: int = 10
    """number of initial states to sample"""
    seed: int = 0
    """reset seed; the same seed and num_states give the same states"""
    name: Optional[str] = None
    """output folder name under runs/init_states (default: <env id>_<num_states>_seed<seed>)"""


def to_cpu(tree):
    if isinstance(tree, dict):
        return {k: to_cpu(v) for k, v in tree.items()}
    return tree.cpu()


def main(args: Args):
    out_dir = f"runs/init_states/{args.name or f'{args.env.id}_{args.num_states}_seed{args.seed}'}"
    os.makedirs(out_dir, exist_ok=True)
    env = gym.make(args.env.id, num_envs=args.num_states, **gym_make_kwargs(args.env))
    env.reset(seed=args.seed)
    base = env.unwrapped

    tee_pose = base.tee.pose
    tee_xy = tee_pose.p[:, :2].cpu()
    q = tee_pose.q.cpu()
    tee_yaw = 2 * torch.atan2(q[:, 3], q[:, 0])
    tee_yaw = torch.remainder(tee_yaw, 2 * math.pi)
    data = dict(
        env_state=to_cpu(base.get_state_dict()),
        tee_xy=tee_xy,
        tee_yaw=tee_yaw,
        robot_qpos=base.agent.robot.get_qpos().cpu(),
        config=dict(config_dict(env=args.env), seed=args.seed, num_states=args.num_states),
    )
    torch.save(data, f"{out_dir}/init_states.pt")

    frames = env.render().cpu().numpy().astype(np.uint8)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    tiles = []
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        text = f"state {i}\nx={tee_xy[i, 0]:.3f} y={tee_xy[i, 1]:.3f}\nyaw={math.degrees(tee_yaw[i]):.0f}deg"
        draw.multiline_text((8, 8), text, font=font, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
        img.save(f"{out_dir}/state_{i}.png")
        tiles.append(np.asarray(img))
    cols = min(5, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    h, w, _ = tiles[0].shape
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        grid[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = t
    Image.fromarray(grid).save(f"{out_dir}/grid.png")

    for i in range(args.num_states):
        print(f"state {i}: tee x={tee_xy[i, 0]:.4f} y={tee_xy[i, 1]:.4f} yaw={math.degrees(tee_yaw[i]):.1f}deg")
    print(f"saved {out_dir}/init_states.pt, state_*.png, grid.png")
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
