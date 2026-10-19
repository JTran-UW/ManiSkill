"""Back-to-back eval episodes of a flashsac_rgb.py checkpoint as one video: left = human render camera, right = the
camera images the policy receives (cameras stacked vertically, newest frame of the stack), nearest-upscaled."""

import argparse
import csv
import os

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import mani_skill.envs  # noqa: F401
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import FrameStack
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from reinforcement_learning.flashsac.networks.flashsac_actor import FlashSACActor
from reinforcement_learning.flashsac.networks.flashsac_utils import FlashSACEncoder
from reinforcement_learning.utils.args import parse_env_kwargs
from reinforcement_learning.utils.viz import ballast_pixels, draw_marker
from reinforcement_learning.utils.init_states import FixedInitStateWrapper, load_init_state, per_env_init_state
from reinforcement_learning.utils.wrappers import GrayscaleObservationWrapper, camera_sensor_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--output", default=None, help="video path; omit to only print success stats")
    p.add_argument("--env-kwargs", nargs="*", default=[], help="extra gym.make kwargs as key value pairs, e.g. tee_ballast_mass_ratio 0.9")
    p.add_argument("--init-state-path", default=None, help="saved init states (save_init_states.py); --num-episodes envs per state")
    p.add_argument("--init-states", type=int, nargs="*", default=None, help="state indices (default: all in the file)")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of the policy mean")
    p.add_argument("--rounds", type=int, default=1, help="episodes per env (stats only)")
    p.add_argument("--csv", default=None, help="write per-state success rates here")
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
    groups = [None]
    if args.init_state_path:
        data = torch.load(args.init_state_path, map_location="cpu", weights_only=False)
        total = len(data["tee_xy"]) if "tee_xy" in data else next(iter(data["env_state"]["actors"].values())).shape[0]
        groups = args.init_states if args.init_states else list(range(total))
    per_group = n
    n = per_group * len(groups)
    if args.output and (args.rounds != 1 or len(groups) > 1):
        raise ValueError("videos support a single round and at most one init state")

    kwargs = dict(obs_mode="rgb", render_mode="rgb_array", sim_backend=env_cfg.get("sim_backend", "gpu"))
    if env_cfg.get("control_mode"):
        kwargs["control_mode"] = env_cfg["control_mode"]
    env = gym.make(
        env_cfg["id"],
        num_envs=n,
        sensor_configs=camera_sensor_config(algo["image_size"], algo.get("camera_eye"), algo.get("camera_target"), algo.get("camera_fov")),
        reconfiguration_freq=env_cfg.get("eval_reconfiguration_freq", 1),
        human_render_camera_configs=dict(shader_pack="default"),
        **kwargs,
        **parse_env_kwargs(env_kwargs),
    )
    if args.init_state_path:
        env = FixedInitStateWrapper(env, per_env_init_state(args.init_state_path, [g for g in groups for _ in range(per_group)]))
    horizon = gym_utils.find_max_episode_steps_value(env)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if algo.get("grayscale"):
        env = GrayscaleObservationWrapper(env)
    env = FrameStack(env, num_stack=algo["n_frame_stack"])
    env = ManiSkillVectorEnv(env, n, ignore_terminations=True, record_metrics=True)

    channels_last = bool(algo.get("bf16"))
    encoder = FlashSACEncoder(
        env, image_embed_dim=algo["image_embed_dim"], state_embed_dim=algo["state_embed_dim"], channels_last=channels_last
    ).to(device)
    shape = env.single_action_space.shape
    actor = FlashSACActor(
        env,
        input_dim=algo["image_embed_dim"] + algo["state_embed_dim"],
        action_scale=torch.ones(shape),
        action_bias=torch.zeros(shape),
        use_tanh=algo["use_tanh"],
    ).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    encoder.eval()
    actor.eval()
    if channels_last:
        encoder.to(memory_format=torch.channels_last)

    gray = bool(algo.get("grayscale"))
    obs, _ = env.reset(seed=args.seed)
    renders, views, success, overlap = [], [], [], []
    u = env.unwrapped
    has_overlap = hasattr(u, "pseudo_render_intersection")
    for _ in range(horizon * args.rounds):
        if args.output:
            frame = env.render().cpu().numpy()
            marks = ballast_pixels(u) if args.mark_ballast else None
            renders.append(frame if marks is None else np.stack([draw_marker(f, m) for f, m in zip(frame, marks)]))
            views.append(obs["rgb"][:, -1].cpu().numpy())
        if has_overlap:
            overlap.append(u.pseudo_render_intersection().cpu().numpy())
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=channels_last):
                embed = encoder(obs).float()
            action = actor.get_action(embed, training=False)[0 if args.stochastic else 2]
        obs, _, _, _, info = env.step(action)
        success.append(info["success"].cpu().numpy())

    success = np.stack(success, 1)
    overlap = np.stack(overlap, 1) if has_overlap else None
    if not args.output:
        per_episode = success.reshape(n, args.rounds, horizon).any(2)
        mode = "stochastic" if args.stochastic else "deterministic"
        print(f"[{mode}] kwargs {env_kwargs}: success_once {per_episode.mean():.3f} over {per_episode.size} episodes")
        rows = []
        if args.init_state_path:
            for i, g in enumerate(groups):
                rate = per_episode[i * per_group:(i + 1) * per_group].mean()
                rows.append(dict(index=g, success_rate=round(float(rate), 4), episodes=per_group * args.rounds))
                print(f"  state {g}: {rate:.3f}")
        if args.csv:
            os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
            with open(args.csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["index", "success_rate", "episodes"])
                w.writeheader()
                w.writerows(rows or [dict(index="all", success_rate=round(float(per_episode.mean()), 4), episodes=per_episode.size)])
        env.close()
        return

    H, W = renders[0].shape[1:3]
    cams = views[0].shape[-1] if gray else views[0].shape[-1] // 3
    cell = H // cams
    frames = []
    for e in range(n):
        succ_once = False
        for t in range(horizon):
            v = views[t][e]
            imgs = [v[..., c : c + 1].repeat(3, -1) for c in range(cams)] if gray else [v[..., 3 * c : 3 * c + 3] for c in range(cams)]
            right = np.concatenate(
                [F.interpolate(torch.from_numpy(i).permute(2, 0, 1)[None].float(), size=(cell, cell), mode="nearest")[0]
                 .permute(1, 2, 0).byte().numpy() for i in imgs], 0)
            right = np.pad(right, ((0, H - right.shape[0]), (0, 0), (0, 0)))
            succ_once |= bool(success[e, t])
            left = label(renders[t][e], ([args.title] if args.title else []) + [f"episode {e + 1}/{n}  t={t + 1}/{horizon}",
                                         f"success {'YES' if success[e, t] else 'no'}  (once: {'YES' if succ_once else 'no'})"]
                         + ([f"goal overlap {overlap[e, t]:.0%} (success >= {u.intersection_thresh:.0%})"] if has_overlap else []))
            right = label(right, [f"policy input {imgs[0].shape[0]}x{imgs[0].shape[1]} {'gray' if gray else 'rgb'} x{cams} cam"])
            frames.append(np.concatenate([left, np.full((H, 4, 3), 255, np.uint8), right], 1))
        print(f"episode {e + 1}: success_once={succ_once}" + (f" max_overlap={overlap[e].max():.3f}" if has_overlap else ""))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    imageio.mimsave(args.output, frames, fps=args.fps, macro_block_size=1)
    print(f"success_once {success.any(1).mean():.2f}; saved {args.output}")
    env.close()


if __name__ == "__main__":
    main()
