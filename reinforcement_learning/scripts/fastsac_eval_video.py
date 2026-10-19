"""Back-to-back eval episodes of a state-based fastsac.py checkpoint as one video (human render camera), labeled like
flashsac_rgb_eval_video.py so the two can be placed side by side."""

import argparse
import os
from types import SimpleNamespace

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

import mani_skill.envs  # noqa: F401
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from reinforcement_learning.fastsac.networks.fastsac_actor import Actor
from reinforcement_learning.utils.args import parse_env_kwargs
from reinforcement_learning.utils.viz import ballast_pixels, draw_marker
from reinforcement_learning.utils.init_states import FixedInitStateWrapper, load_init_state
from reinforcement_learning.utils.utils import EmpiricalNormalization


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--output", required=True)
    p.add_argument("--env-kwargs", nargs="*", default=[], help="extra gym.make kwargs as key value pairs")
    p.add_argument("--init-state-path", default=None)
    p.add_argument("--init-state-index", type=int, default=0)
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of the policy mean")
    p.add_argument("--title", default=None, help="first label line on every frame")
    p.add_argument("--mark-ballast", action="store_true", help="draw the ballast position on the rendered frames (not on policy inputs)")
    return p.parse_args()


def label(img, lines):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for i, s in enumerate(lines):
        d.text((6, 4 + 14 * i), s, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(im)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    env_cfg, algo = ckpt["config"]["env"], ckpt["config"]["algo"]
    n = args.num_episodes
    if len(args.env_kwargs) % 2:
        raise ValueError("--env-kwargs takes key value pairs")
    env_kwargs = {**env_cfg.get("kwargs", {}), **dict(zip(args.env_kwargs[::2], args.env_kwargs[1::2]))}

    env = gym.make(
        env_cfg["id"],
        num_envs=n,
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend=env_cfg.get("sim_backend", "gpu"),
        control_mode=env_cfg["control_mode"],
        reconfiguration_freq=env_cfg.get("eval_reconfiguration_freq", 1),
        human_render_camera_configs=dict(shader_pack="default"),
        **parse_env_kwargs(env_kwargs),
    )
    if args.init_state_path:
        env = FixedInitStateWrapper(env, load_init_state(args.init_state_path, args.init_state_index))
    horizon = gym_utils.find_max_episode_steps_value(env)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, n, ignore_terminations=True, record_metrics=True)

    shape = env.single_action_space.shape
    actor = Actor(env, action_scale=torch.ones(shape), action_bias=torch.zeros(shape), use_tanh=algo["use_tanh"]).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    normalize = lambda x: x
    if algo.get("obs_normalization") and "obs_normalizer" in ckpt:
        norm = EmpiricalNormalization(shape=env.single_observation_space.shape, device=device)
        norm.load_state_dict(ckpt["obs_normalizer"])
        norm.eval()
        normalize = lambda x: norm(x, update=False)

    obs, _ = env.reset(seed=args.seed)
    u = env.unwrapped
    has_overlap = hasattr(u, "pseudo_render_intersection")
    renders, success, overlap = [], [], []
    for _ in range(horizon):
        frame = env.render().cpu().numpy()
        marks = ballast_pixels(u) if args.mark_ballast else None
        renders.append(frame if marks is None else np.stack([draw_marker(f, m) for f, m in zip(frame, marks)]))
        if has_overlap:
            overlap.append(u.pseudo_render_intersection().cpu().numpy())
        with torch.no_grad():
            action = actor.get_action(normalize(obs))[0 if args.stochastic else 2]
        obs, _, _, _, info = env.step(action)
        success.append(info["success"].cpu().numpy())
    success = np.stack(success, 1)
    overlap = np.stack(overlap, 1) if has_overlap else None

    frames = []
    for e in range(n):
        succ_once = False
        for t in range(horizon):
            succ_once |= bool(success[e, t])
            lines = ([args.title] if args.title else []) + [
                f"episode {e + 1}/{n}  t={t + 1}/{horizon}",
                f"success {'YES' if success[e, t] else 'no'}  (once: {'YES' if succ_once else 'no'})",
            ] + ([f"goal overlap {overlap[e, t]:.0%} (success >= {u.intersection_thresh:.0%})"] if has_overlap else [])
            frames.append(label(renders[t][e], lines))
        print(f"episode {e + 1}: success_once={succ_once}" + (f" max_overlap={overlap[e].max():.3f}" if has_overlap else ""))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    imageio.mimsave(args.output, frames, fps=args.fps, macro_block_size=1)
    print(f"success_once {success.any(1).mean():.2f}; saved {args.output}")
    env.close()


if __name__ == "__main__":
    main()
