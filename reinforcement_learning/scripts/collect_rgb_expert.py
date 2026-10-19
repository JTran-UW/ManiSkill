"""Roll out a state-based FlashSAC checkpoint while rendering the RGB observations a vision policy sees, and save them as
an ImageReplayBuffer (same layout as flashsac_rgb.py's replay buffer) for RLPD-style expert mixing.

The env runs obs_mode="rgb+state": the expert acts on the full privileged state, while the stored state is its
non-privileged prefix, i.e. exactly what obs_mode="rgb" returns (checked at startup).
"""

import argparse
import os
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.envs.utils.observations import parse_obs_mode_to_struct
from mani_skill.utils import common, gym_utils
from mani_skill.utils.wrappers import FrameStack
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from reinforcement_learning.fastsac.networks.fastsac_actor import Actor as FastSACActor
from reinforcement_learning.flashsac.networks.flashsac_actor import FlashSACActor
from reinforcement_learning.utils.buffers import ImageReplayBuffer
from reinforcement_learning.utils.utils import EmpiricalNormalization
from reinforcement_learning.utils.wrappers import GrayscaleObservationWrapper, camera_metadata, camera_sensor_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="state-based flashsac.py or fastsac.py checkpoint")
    p.add_argument("--expert-algo", choices=("flashsac", "fastsac"), default="flashsac")
    p.add_argument("--env-id", default="PickCube-v1")
    p.add_argument("--control-mode", default="pd_ee_delta_pose")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--num-rounds", type=int, default=4, help="episodes collected per env")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--grayscale", action="store_true", help="store 1-channel luminance per camera")
    p.add_argument("--camera-eye", type=float, nargs=3, default=None, help="world-frame camera eye (with --camera-target)")
    p.add_argument("--camera-target", type=float, nargs=3, default=None)
    p.add_argument("--camera-fov", type=float, default=None, help="camera field of view (radians)")
    p.add_argument("--n-frame-stack", type=int, default=1)
    p.add_argument("--noise-std", type=float, default=0.0, help="gaussian noise added to the expert's actions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    return p.parse_args()


def vision_state_dim(u):
    """Size of the state obs_mode='rgb' would return for this env (the non-privileged prefix)."""
    saved = u._obs_mode, u.obs_mode_struct
    u._obs_mode, u.obs_mode_struct = "rgb", parse_obs_mode_to_struct("rgb")
    try:
        vis = common.flatten_state_dict(u._get_obs_state_dict(u.get_info()), use_torch=True, device=u.device)
    finally:
        u._obs_mode, u.obs_mode_struct = saved
    return vis


def main():
    args = parse_args()
    device = torch.device("cuda")

    env = gym.make(
        args.env_id,
        num_envs=args.num_envs,
        obs_mode="rgb+state",
        sim_backend="gpu",
        control_mode=args.control_mode,
        sensor_configs=camera_sensor_config(args.image_size, args.camera_eye, args.camera_target, args.camera_fov),
    )
    horizon = gym_utils.find_max_episode_steps_value(env)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if args.grayscale:
        env = GrayscaleObservationWrapper(env)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = FrameStack(env, num_stack=args.n_frame_stack)
    env = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=True, record_metrics=True)
    u = env.unwrapped

    obs, _ = env.reset(seed=args.seed)
    full_dim = obs["state"].shape[-1]
    vis = vision_state_dim(u)
    vis_dim = vis.shape[-1]
    assert torch.equal(vis, obs["state"][:, -1, :vis_dim]), "obs_mode='rgb' state is not a prefix of the rgb+state state"
    print(f"[collect] rgb {tuple(env.single_observation_space['rgb'].shape)}, privileged state {full_dim}, stored state {vis_dim}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    shape = env.single_action_space.shape
    normalize = lambda x: x
    if args.expert_algo == "fastsac":
        state_env = SimpleNamespace(
            single_observation_space=gym.spaces.Box(-np.inf, np.inf, (full_dim,)), single_action_space=env.single_action_space
        )
        actor = FastSACActor(state_env, action_scale=torch.ones(shape), action_bias=torch.zeros(shape), use_tanh=True)
        if "obs_normalizer" in ckpt:
            obs_normalizer = EmpiricalNormalization(shape=(full_dim,), device=device)
            obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
            obs_normalizer.eval()
            normalize = lambda x: obs_normalizer(x, update=False)
    else:
        actor = FlashSACActor(
            env, input_dim=full_dim, action_scale=torch.ones(shape), action_bias=torch.zeros(shape), use_tanh=True
        )
    actor = actor.to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    size = args.num_envs * horizon * args.num_rounds
    rb = ImageReplayBuffer(
        pixels_obs_space=env.single_observation_space["rgb"].shape,
        state_obs_dim=vis_dim,
        act_dim=int(np.prod(shape)),
        size=size,
        horizon=horizon,
        num_envs=args.num_envs,
        n_frame_stack=args.n_frame_stack,
        n_steps=1,
        gamma=0.99,
        device=device,
    )

    def vision_view(o):
        return {"rgb": o["rgb"], "state": o["state"][..., :vis_dim]}

    successes, returns = [], []
    for _ in range(horizon * args.num_rounds):
        with torch.no_grad():
            state = normalize(obs["state"][:, -1])
            actions = (actor.get_action(state) if args.expert_algo == "fastsac" else actor.get_action(state, training=False))[2]
            if args.noise_std > 0:
                actions = (actions + args.noise_std * torch.randn_like(actions)).clamp(-1, 1)
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        real_next_obs = {k: v.clone() for k, v in next_obs.items()}
        if "final_info" in infos:
            for k in real_next_obs:
                real_next_obs[k][truncs] = infos["final_observation"][k][truncs]
            done_mask = infos["_final_info"]
            episode = infos["final_info"]["episode"]
            successes.append(episode["success_once"][done_mask].float().mean().item())
            returns.append(episode["return"][done_mask].float().mean().item())
        rb.add(vision_view(obs), vision_view(real_next_obs), actions, rewards, terms)
        obs = next_obs

    assert rb.current_size == size and rb.step_num == 0 and rb.batch_start_episode == 0, "buffer must hold whole episodes"

    k, H, W, C = env.single_observation_space["rgb"].shape
    metadata = dict(
        env_id=args.env_id,
        control_mode=args.control_mode,
        image_size=args.image_size,
        grayscale=args.grayscale,
        n_frame_stack=args.n_frame_stack,
        num_cameras=C if args.grayscale else C // 3,
        rgb_shape=(k, H, W, C),
        state_dim=vis_dim,
        privileged_state_dim=full_dim,
        horizon=horizon,
        camera=camera_metadata(args.camera_eye, args.camera_target, args.camera_fov),
        checkpoint=args.checkpoint,
        expert_algo=args.expert_algo,
        num_envs=args.num_envs,
        num_rounds=args.num_rounds,
        noise_std=args.noise_std,
        num_transitions=size,
        success_once=float(np.mean(successes)),
        ret=float(np.mean(returns)),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(rb.state_dict(metadata), args.output)
    print(metadata)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
