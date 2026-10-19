"""Eval video for ppo_rgb.py / sac_rgbd.py (scene | policy RGB input) and flashsac.py (scene only) checkpoints, episodes back to back."""
import argparse
import os
import sys

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
BASELINES = os.path.join(REPO, "examples/baselines")

PANE = 512
HEADER = 56
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--algo", choices=["ppo", "sac", "flashsac"], default="ppo")
    p.add_argument("--camera-size", type=int, default=None)
    p.add_argument("--env-id", default="PushCube-v1")
    p.add_argument("--num-episodes", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--control-mode", default="pd_joint_delta_pos")
    p.add_argument("--no-include-state", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--output", required=True)
    return p.parse_args()


def header(width, text, color):
    img = Image.new("RGB", (width, HEADER), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 26)
    tw = draw.textlength(text, font=font)
    draw.text(((width - tw) / 2, (HEADER - 30) / 2), text, fill=color, font=font)
    return np.asarray(img)


def main():
    args = parse_args()
    device = torch.device("cuda")
    use_rgb = args.algo != "flashsac"
    env = gym.make(
        args.env_id,
        num_envs=args.num_episodes,
        obs_mode="rgb" if use_rgb else "state",
        render_mode="rgb_array",
        sim_backend="physx_cuda",
        control_mode=args.control_mode,
        reconfiguration_freq=1,
        sensor_configs=dict(width=args.camera_size, height=args.camera_size) if args.camera_size else dict(),
    )
    if use_rgb:
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=not args.no_include_state)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, args.num_episodes, ignore_terminations=True, auto_reset=False, record_metrics=True)

    obs, _ = env.reset(seed=args.seed)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if args.algo == "ppo":
        sys.path.insert(0, os.path.join(BASELINES, "ppo"))
        from ppo_rgb import Agent

        agent = Agent(env, sample_obs=obs).to(device)
        agent.load_state_dict(ckpt)
        act = lambda o: agent.get_action(o, deterministic=True)
    elif args.algo == "sac":
        sys.path.insert(0, os.path.join(BASELINES, "sac"))
        from sac_rgbd import Actor

        agent = Actor(env, sample_obs=obs).to(device)
        agent.load_state_dict(ckpt["actor"])
        act = agent.get_eval_action
    else:
        sys.path.insert(0, REPO)
        from reinforcement_learning.flashsac.networks.flashsac_actor import FlashSACActor

        shape = env.single_action_space.shape
        agent = FlashSACActor(env, action_scale=torch.ones(shape), action_bias=torch.zeros(shape), use_tanh=True).to(device)
        agent.load_state_dict(ckpt["actor"])
        act = lambda o: agent.get_action(o, training=False)[2]
    agent.eval()

    scene_frames = [env.render().cpu().numpy()]
    policy_frames = [obs["rgb"].cpu().numpy()] if use_rgb else None
    successes = []
    for _ in range(args.num_steps):
        with torch.no_grad():
            obs, _, _, _, info = env.step(act(obs))
        scene_frames.append(env.render().cpu().numpy())
        if use_rgb:
            policy_frames.append(obs["rgb"].cpu().numpy())
        successes.append(info["success"].cpu().numpy())
    env.close()

    successes = np.stack(successes)  # (T, N)
    success_at_end = successes[-1]
    success_once = successes.any(axis=0)
    if use_rgb:
        h, w = policy_frames[0].shape[1:3]
        right_header = header(PANE, f"Policy input ({w}x{h} camera)", (255, 255, 255))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    writer = imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
    for ep in range(args.num_episodes):
        ok = bool(success_at_end[ep])
        left_header = header(
            PANE,
            f"Episode {ep + 1}/{args.num_episodes}: {'SUCCESS' if ok else 'FAILURE'}",
            (90, 220, 110) if ok else (240, 80, 80),
        )
        for t in range(len(scene_frames)):
            left = np.asarray(Image.fromarray(scene_frames[t][ep]).resize((PANE, PANE), Image.BILINEAR))
            frame = np.vstack([left_header, left])
            if use_rgb:
                right = np.asarray(Image.fromarray(policy_frames[t][ep][..., :3]).resize((PANE, PANE), Image.NEAREST))
                frame = np.concatenate([frame, np.vstack([right_header, right])], axis=1)
            writer.append_data(frame)
    writer.close()

    for ep in range(args.num_episodes):
        print(f"episode {ep + 1}: success_at_end={bool(success_at_end[ep])} success_once={bool(success_once[ep])}")
    print(f"success_at_end={success_at_end.mean():.3f} success_once={success_once.mean():.3f}")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
