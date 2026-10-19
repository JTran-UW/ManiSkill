from collections import defaultdict, deque
from dataclasses import dataclass, field
import statistics
import copy
import math
import os
import random
import time
from typing import Optional, Tuple

import tqdm

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import FrameStack
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tyro

from reinforcement_learning.flashsac.networks.flashsac_actor import FlashSACActor
from reinforcement_learning.flashsac.networks.flashsac_critic import FlashSACQNetwork
from reinforcement_learning.flashsac.networks.flashsac_utils import FlashSACEncoder

from reinforcement_learning.utils.args import (
    ENV_TASK_FIELDS,
    EnvArgs,
    RunArgs,
    config_dict,
    gym_make_kwargs,
    parse_env_kwargs,
    restore_from_saved,
)
from reinforcement_learning.utils.augmentations import random_shift
from reinforcement_learning.utils.buffers import ImageReplayBuffer, cat_image_samples, sample_mixed
from reinforcement_learning.utils.init_states import FixedInitStateWrapper, load_init_state
from reinforcement_learning.utils.wrappers import GrayscaleObservationWrapper, camera_metadata, camera_sensor_config

import mani_skill.envs

@dataclass
class AlgoArgs:
    num_envs: int = 1024
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments"""
    buffer_size: int = 1_024_000
    """the replay memory buffer size"""
    buffer_device: str = "cuda"
    """where the replay buffer is stored. Can be 'cpu' or 'cuda' for GPU"""
    gamma: float = 0.97
    """the discount factor gamma"""
    tau: float = 0.01
    """target smoothing coefficient"""
    batch_size: int = 1024
    """the batch size of sample from the replay memory"""
    learning_starts: int = 10_240
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    alpha_lr: float = 3e-4
    """the learning rate of alpha"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.01
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    training_freq: int = 1024
    """training frequency (in steps)"""
    utd: float = 0.0001
    """update to data ratio"""

    # FlashSAC-specific stuff
    n_step_returns: int = 1
    """n-step returns"""
    v_min: float = -20.0
    """minimum of critic distribution"""
    v_max: float = 20.0
    """maximum of critic distribution"""
    use_tanh: bool = True
    """use tanh layer after policy"""
    obs_normalization: bool = True
    """use obs normalization"""
    target_entropy_sigma: float = 0.15
    """ratio of target entropy"""
    noise_repeat_zeta_mu: float = 2.0
    """zeta-distribution shape parameter for noise-repetition exploration"""
    noise_repeat_zeta_max: int = 16
    """max repeat length (in env steps) for noise-repetition exploration"""
    lr_end_ratio: float = 0.5
    """cosine-decay every lr from its initial value to lr_end_ratio * lr over the run's updates (1.0 = constant lr)"""
    logging_interval: int = 100
    """the interval to log the metrics"""

    # RGB-specific stuff
    image_size: int = 64
    """size of image"""
    grayscale: bool = False
    """convert camera images to grayscale in the env (1 channel per camera)"""
    camera_eye: Optional[Tuple[float, float, float]] = None
    """re-pose the env's cameras: world-frame eye position (with --camera-target)"""
    camera_target: Optional[Tuple[float, float, float]] = None
    """re-pose the env's cameras: world-frame look-at target (with --camera-eye)"""
    camera_fov: Optional[float] = None
    """override the env's camera field of view (radians)"""
    image_embed_dim: int = 64
    """image embedding dimension"""
    state_embed_dim: int = 64
    """low-dim state embedding dimension"""
    n_frame_stack: int = 1
    """number of frames to stack"""
    encoder_ema: float = 0.01
    """ema factor image encoder"""
    freeze_encoder: bool = False
    """keep the image encoder (and its target) fixed: no critic gradients, no renorm, no EMA"""
    init_state_train_only: bool = False
    """with --env.init-state-path, pin only the training env; eval envs keep the env's random resets"""
    expert_buffer: Optional[str] = None
    """RLPD: path to a collect_rgb_expert.py buffer mixed into every batch"""
    expert_ratio: float = 0.5
    """RLPD: fraction of each batch drawn from --expert-buffer"""
    aug_pad: int = 0
    bf16: bool = False
    """bf16 autocast for encoder/critic forwards (outputs cast back to fp32), channels_last convs, cudnn.benchmark"""
    compile: bool = False
    """torch.compile (default mode) the network forwards, actor sampling, target projection and the per-update renorm + target EMA"""
    """DrQ random-shift padding in pixels for training batches (0 = no augmentation)"""

    # Logging
    wandb_group: str = "FlashSAC"
    """the group of the run for wandb"""


def _build_truncated_zeta_cdf(mu: float, max_n: int, device) -> torch.Tensor:
    ns = torch.arange(1, max_n + 1, dtype=torch.float32, device=device)
    pmf = ns ** (-mu)
    pmf = pmf / torch.sum(pmf)
    return torch.cumsum(pmf, dim=0)


def _sample_integer_from_cdf(cdf: torch.Tensor) -> torch.Tensor:
    u = torch.rand((), device=cdf.device)
    idx = torch.argmax((u < cdf).to(torch.int32))
    return (idx + 1).to(torch.int32)


ALGO_MODEL_FIELDS = (
    "v_min", "v_max", "use_tanh", "image_size", "grayscale", "image_embed_dim", "state_embed_dim", "n_frame_stack",
    "camera_eye", "camera_target", "camera_fov",
)


def bf16_forward(module):
    """Run module.forward under bf16 autocast and return fp32, without changing the module or its state_dict."""
    fwd = module.forward
    def wrapped(*args, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return fwd(*args, **kwargs).float()
    module.forward = wrapped


def _sample_rollout_action(encoder, actor, x, noise, cur_count, cur_n, zeta_cdf, training: bool):
    x_embed = encoder(x)
    mean, log_std = actor(x_embed, training=training)
    std = log_std.exp()

    reinit = (cur_count == 0) | (cur_count >= cur_n)
    new_noise = torch.randn_like(mean)
    new_n = _sample_integer_from_cdf(zeta_cdf)

    noise = torch.where(reinit, new_noise, noise)
    cur_n = torch.where(reinit, new_n, cur_n)
    cur_count = torch.where(reinit, torch.zeros_like(cur_count), cur_count)

    x_t = mean + std * noise
    if actor.use_tanh:
        action = torch.tanh(x_t) * actor.action_scale + actor.action_bias
    else:
        action = x_t

    return action, noise, cur_count + 1, cur_n



@dataclass
class Args:
    run: tyro.conf.OmitArgPrefixes[RunArgs] = field(default_factory=lambda: RunArgs(eval_freq=200_000, save_freq=1_000_000))
    env: EnvArgs = field(default_factory=EnvArgs)
    algo: tyro.conf.OmitArgPrefixes[AlgoArgs] = field(default_factory=AlgoArgs)


class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
        self.counters = {"per_env_step": 0, "train_episodes": 0}
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value, **self.counters}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)

    ckpt = None
    if args.run.checkpoint is not None:
        ckpt = torch.load(args.run.checkpoint)
        if args.run.config_from_checkpoint:
            saved = ckpt.get("config")
            if saved is None:
                print("checkpoint has no saved config; using CLI env/algo args as given")
            else:
                notes = restore_from_saved(args.env, saved["env"], ENV_TASK_FIELDS)
                notes += restore_from_saved(args.algo, saved["algo"], ALGO_MODEL_FIELDS)
                for note in notes:
                    print(f"[config] {note}")
    if args.run.resume and ckpt is None:
        raise ValueError("--resume needs --checkpoint")
    if args.run.wandb_run_id is not None:
        args.run.track = True
    args.env.obs_mode = "rgb"

    grad_steps_per_iteration = int(args.algo.training_freq * args.algo.utd)
    steps_per_env = args.algo.training_freq // args.algo.num_envs
    embed_dim = args.algo.image_embed_dim + args.algo.state_embed_dim
    if args.run.exp_name is None:
        args.run.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env.id}__{args.run.exp_name}__{args.run.seed}__{int(time.time())}"
    else:
        run_name = args.run.exp_name
    
    # TRY NOT TO MODIFY: seeding
    random.seed(args.run.seed)
    np.random.seed(args.run.seed)
    torch.manual_seed(args.run.seed)
    torch.backends.cudnn.deterministic = args.run.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.run.cuda else "cpu")

    ####### Environment setup #######
    env_kwargs = gym_make_kwargs(args.env)
    envs = gym.make(
        args.env.id,
        sensor_configs=camera_sensor_config(args.algo.image_size, args.algo.camera_eye, args.algo.camera_target, args.algo.camera_fov),
        num_envs=args.algo.num_envs if not args.run.evaluate else 1,
        reconfiguration_freq=args.env.reconfiguration_freq,
        **env_kwargs
    )
    eval_envs = gym.make(
        args.env.id,
        sensor_configs=camera_sensor_config(args.algo.image_size, args.algo.camera_eye, args.algo.camera_target, args.algo.camera_fov),
        num_envs=args.algo.num_eval_envs,
        reconfiguration_freq=args.env.eval_reconfiguration_freq,
        human_render_camera_configs=dict(shader_pack="default"),
        **env_kwargs
    )
    if args.env.init_state_path is not None:
        init_state = load_init_state(args.env.init_state_path, args.env.init_state_index)
        which = "training env" if args.algo.init_state_train_only else "training and eval envs"
        print(f"[init_state] {which} reset to state {init_state['index']} of {init_state['path']}: {init_state['info']}")
        envs = FixedInitStateWrapper(envs, init_state)
        if not args.algo.init_state_train_only:
            eval_envs = FixedInitStateWrapper(eval_envs, init_state)
    max_episode_steps = gym_utils.find_max_episode_steps_value(envs)
    if args.env.num_eval_steps is None:
        args.env.num_eval_steps = max_episode_steps
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.run.capture_video or args.run.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.run.evaluate:
            eval_output_dir = f"{os.path.dirname(args.run.checkpoint)}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if args.run.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // max_episode_steps) % args.run.save_train_video_freq == 0
            envs = RecordEpisode(
                envs,
                output_dir=f"runs/{run_name}/train_videos",
                save_trajectory=args.run.save_train_trajectory,
                trajectory_name="trajectory",
                save_video_trigger=save_video_trigger,
                max_steps_per_video=max_episode_steps,
                video_fps=30,
            )
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=args.run.save_trajectory,
            save_video=args.run.capture_video and args.run.capture_eval_video,
            trajectory_name="trajectory",
            max_steps_per_video=args.env.num_eval_steps,
            video_fps=30,
        )
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=True)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=True)
    if args.algo.grayscale:
        envs = GrayscaleObservationWrapper(envs)
        eval_envs = GrayscaleObservationWrapper(eval_envs)
    envs = FrameStack(envs, num_stack=args.algo.n_frame_stack)
    eval_envs = FrameStack(eval_envs, num_stack=args.algo.n_frame_stack)
    envs = ManiSkillVectorEnv(envs, args.algo.num_envs, ignore_terminations=not args.env.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.algo.num_eval_envs, ignore_terminations=not args.env.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    saved_config = config_dict(run=args.run, env=args.env, algo=args.algo)
    logger = None
    if not args.run.evaluate and not args.run.collect_replay_buffer:
        print("Running training")
        if args.run.track:
            import wandb
            config = {**saved_config, "derived": dict(grad_steps_per_iteration=grad_steps_per_iteration, steps_per_env=steps_per_env, max_episode_steps=max_episode_steps)}
            config["env"]["gym_make_kwargs"] = env_kwargs
            if args.run.wandb_run_id is not None:
                wandb.init(id=args.run.wandb_run_id, resume="must", project=args.run.wandb_project_name, entity=args.run.wandb_entity)
                wandb.config.update(config, allow_val_change=True)
            else:
                wandb.init(
                    project=args.run.wandb_project_name,
                    entity=args.run.wandb_entity,
                    sync_tensorboard=False,
                    config=config,
                    name=run_name,
                    save_code=True,
                    group=args.algo.wandb_group,
                    tags=["sac", "walltime_efficient"]
                )
            wandb.define_metric("per_env_step")
            wandb.define_metric("*", step_metric="per_env_step")
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{section}.{key}|{value}|" for section, values in saved_config.items() for key, value in values.items()])),
        )
        logger = Logger(log_wandb=args.run.track, tensorboard=writer)
    else:
        print("Running evaluation")

    max_action = float(envs.single_action_space.high[0])

    actor = FlashSACActor(
        envs,
        input_dim=embed_dim,
        action_scale=torch.ones(envs.single_action_space.shape),
        action_bias=torch.zeros(envs.single_action_space.shape),
        use_tanh=args.algo.use_tanh
    ).to(device)
    zeta_cdf = _build_truncated_zeta_cdf(args.algo.noise_repeat_zeta_mu, args.algo.noise_repeat_zeta_max, device)
    noise_repeat_n = torch.ones((), dtype=torch.int32, device=device)
    noise_repeat_count = torch.zeros((), dtype=torch.int32, device=device)
    cached_noise = torch.randn((args.algo.num_envs,) + envs.single_action_space.shape, device=device)
    qf1 = FlashSACQNetwork(
        envs,
        input_dim=embed_dim,
        v_min=args.algo.v_min,
        v_max=args.algo.v_max
    ).to(device)
    qf2 = FlashSACQNetwork(
        envs,
        input_dim=embed_dim,
        v_min=args.algo.v_min,
        v_max=args.algo.v_max
    ).to(device)
    qf1_target = FlashSACQNetwork(
        envs,
        input_dim=embed_dim,
        v_min=args.algo.v_min,
        v_max=args.algo.v_max
    ).to(device)
    qf2_target = FlashSACQNetwork(
        envs,
        input_dim=embed_dim,
        v_min=args.algo.v_min,
        v_max=args.algo.v_max
    ).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    encoder = FlashSACEncoder(envs, image_embed_dim=args.algo.image_embed_dim, state_embed_dim=args.algo.state_embed_dim, channels_last=args.algo.bf16).to(device)
    target_encoder = FlashSACEncoder(envs, image_embed_dim=args.algo.image_embed_dim, state_embed_dim=args.algo.state_embed_dim, channels_last=args.algo.bf16).to(device)
    target_encoder.load_state_dict(encoder.state_dict())
    if args.algo.bf16:
        torch.backends.cudnn.benchmark = True
        for m in (encoder, target_encoder):
            m.to(memory_format=torch.channels_last)
        for m in (encoder, target_encoder, qf1, qf2, qf1_target, qf2_target):
            bf16_forward(m)
    if args.algo.freeze_encoder:
        encoder.requires_grad_(False)
        target_encoder.requires_grad_(False)
    encoder_params = [] if args.algo.freeze_encoder else list(encoder.parameters())
    q_optimizer = optim.Adam(encoder_params + list(qf1.parameters()) + list(qf2.parameters()), lr=args.algo.q_lr, fused=True)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.algo.policy_lr, fused=True)
    ema_pairs = [
        (list(qf1_target.parameters()) + list(qf2_target.parameters()), list(qf1.parameters()) + list(qf2.parameters()), args.algo.tau),
    ]
    if not args.algo.freeze_encoder:
        ema_pairs.insert(0, (list(target_encoder.parameters()), list(encoder.parameters()), args.algo.encoder_ema))

    # Normalize after init
    actor.normalize_parameters()
    qf1.normalize_parameters()
    qf2.normalize_parameters()
    qf1_target.normalize_parameters()
    qf2_target.normalize_parameters()
    encoder.normalize_parameters()

    if ckpt is not None:
        actor.load_state_dict(ckpt["actor"])
        encoder.load_state_dict(ckpt["encoder"])
        target_encoder.load_state_dict(ckpt.get("target_encoder", ckpt["encoder"]))
        # checkpoints store the target critics; they seed both online and target
        for online, target, key in ((qf1, qf1_target, "qf1"), (qf2, qf2_target, "qf2")):
            online.load_state_dict(ckpt[key])
            target.load_state_dict(ckpt[key])
        print(f"[checkpoint] loaded actor/encoder/critics from {args.run.checkpoint} (step {ckpt.get('global_step')})")

    def renorm_and_ema():
        if not args.algo.freeze_encoder:
            encoder.normalize_parameters()
        qf1.normalize_parameters()
        qf2.normalize_parameters()
        with torch.no_grad():
            for targets, online, w in ema_pairs:
                torch._foreach_lerp_(targets, online, w)

    if args.algo.compile:
        for m in (encoder, target_encoder, qf1, qf2, qf1_target, qf2_target, actor):
            m.compile()
        actor.get_action = torch.compile(actor.get_action)
        for m in (qf1_target, qf2_target):
            m.projection = torch.compile(m.projection)
        renorm_and_ema = torch.compile(renorm_and_ema)

    # Automatic entropy tuning
    if args.algo.autotune:
        target_entropy = 0.5 * torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item() * torch.log(torch.tensor(2 * torch.pi * torch.e * args.algo.target_entropy_sigma ** 2)).item()
        alpha = torch.tensor(args.algo.alpha)
        log_alpha = torch.tensor([torch.log(alpha)], requires_grad=True, device=device)
        a_optimizer = optim.Adam([log_alpha], lr=args.algo.alpha_lr, fused=True)
    else:
        alpha = args.algo.alpha

    start_step = 0
    if args.run.resume:
        start_step = ckpt.get("global_step", args.run.resume_step)
        if start_step is None:
            raise ValueError("checkpoint stores no global_step; pass --resume-step")
        if args.algo.autotune and "log_alpha" in ckpt:
            log_alpha.data.copy_(ckpt["log_alpha"].to(device))
            alpha = log_alpha.detach().exp()
        restored = []
        if args.algo.freeze_encoder and not args.run.reset_optimizers and "q_optimizer" in ckpt:
            raise ValueError("--freeze-encoder changes the critic optimizer's parameters; resume it with --reset-optimizers")
        if not args.run.reset_optimizers:
            opts = [("q_optimizer", q_optimizer, args.algo.q_lr), ("actor_optimizer", actor_optimizer, args.algo.policy_lr)]
            if args.algo.autotune:
                opts.append(("a_optimizer", a_optimizer, args.algo.alpha_lr))
            for name, opt, lr in opts:
                if name in ckpt:
                    opt.load_state_dict(ckpt[name])
                    # keep Adam moments, but the CLI learning rate wins over the checkpoint's (possibly decayed) one
                    for group in opt.param_groups:
                        group["lr"] = lr
                        group.pop("initial_lr", None)
                    restored.append(name)
        if logger is not None:
            if "per_env_step" in ckpt:
                logger.counters["per_env_step"] = ckpt["per_env_step"]
                logger.counters["train_episodes"] = ckpt.get("train_episodes", 0)
            elif ckpt.get("config") is not None:
                logger.counters["per_env_step"] = start_step // ckpt["config"]["algo"]["num_envs"]
        print(f"[resume] step {start_step}, alpha {float(alpha):.4g}, optimizers restored: {restored or 'none'}, lr q/actor {q_optimizer.param_groups[0]['lr']:.2g}/{actor_optimizer.param_groups[0]['lr']:.2g}, counters: {logger.counters if logger is not None else None}")
    elif args.run.reset_optimizers:
        print("[resume] --reset-optimizers has no effect without --resume")

    total_updates = max(1, (args.run.total_timesteps - start_step - args.algo.learning_starts) // (args.algo.num_envs * steps_per_env) * grad_steps_per_iteration)

    def cosine_lr(step):
        progress = min(step / total_updates, 1.0)
        return args.algo.lr_end_ratio + (1 - args.algo.lr_end_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    q_scheduler = optim.lr_scheduler.LambdaLR(q_optimizer, cosine_lr)
    actor_scheduler = optim.lr_scheduler.LambdaLR(actor_optimizer, cosine_lr)
    if args.algo.autotune:
        a_scheduler = optim.lr_scheduler.LambdaLR(a_optimizer, cosine_lr)

    envs.single_observation_space.dtype = np.float32
    rb = ImageReplayBuffer(
        pixels_obs_space=envs.single_observation_space["rgb"].shape,
        state_obs_dim=envs.single_observation_space["state"].shape[1],
        act_dim=envs.single_action_space.shape[0],
        size=args.algo.buffer_size,
        horizon=gym_utils.find_max_episode_steps_value(envs),
        num_envs=args.algo.num_envs,
        n_frame_stack=args.algo.n_frame_stack,
        n_steps=args.algo.n_step_returns,
        gamma=args.algo.gamma
    )
    expert_rb, n_expert = None, 0
    if args.algo.expert_buffer is not None:
        d = torch.load(args.algo.expert_buffer, weights_only=False)
        meta = d["metadata"]
        expected = dict(
            env_id=args.env.id,
            control_mode=args.env.control_mode,
            image_size=args.algo.image_size,
            n_frame_stack=args.algo.n_frame_stack,
            rgb_shape=tuple(envs.single_observation_space["rgb"].shape),
            state_dim=rb.state_obs_buf.shape[-1],
            horizon=rb.horizon,
            camera=camera_metadata(args.algo.camera_eye, args.algo.camera_target, args.algo.camera_fov),
        )
        mismatch = {k: (meta.get(k), v) for k, v in expected.items() if meta.get(k) != v}
        assert not mismatch, f"expert buffer does not match this run (buffer, run): {mismatch}"
        expert_rb = ImageReplayBuffer.from_state_dict(d, n_steps=args.algo.n_step_returns, gamma=args.algo.gamma, device=device)
        del d
        n_expert = int(round(args.algo.batch_size * args.algo.expert_ratio))
        print(f"[rlpd] expert buffer {args.algo.expert_buffer}: {len(expert_rb):,} transitions, "
              f"expert success {meta.get('success_once', float('nan')):.3f}, {n_expert}/{args.algo.batch_size} per batch")

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset(seed=args.run.seed) # in Gymnasium, seed is given to reset() instead of seed()
    eval_obs, _ = eval_envs.reset(seed=args.run.seed)
    global_step = start_step
    global_update = 0
    actor_loss = policy_entropy = alpha_loss = encoder_grad_norm = torch.zeros((), device=device)
    recent_success = deque(maxlen=5)
    stop_training = False
    learning_has_started = False

    global_steps_per_iteration = args.algo.num_envs * (steps_per_env)
    pbar = tqdm.tqdm(total=args.run.total_timesteps, initial=start_step)
    cumulative_times = defaultdict(float)

    while global_step < args.run.total_timesteps:
        if args.run.eval_freq > 0 and (global_step - args.algo.training_freq) // args.run.eval_freq < global_step // args.run.eval_freq:
            # evaluate
            actor.eval()
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.env.num_eval_steps):
                with torch.no_grad():
                    embed = encoder(eval_obs)
                    eval_actions = actor.get_action(embed, training=False)[2]
                    eval_next_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(eval_actions)

                    real_eval_next_obs = copy.deepcopy(eval_next_obs) if args.run.collect_replay_buffer else eval_next_obs
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)

                        if args.run.collect_replay_buffer:
                            for k in real_eval_next_obs:
                                real_eval_next_obs[k][eval_truncations] = eval_infos["final_observation"][k][eval_truncations]

                    if args.run.collect_replay_buffer:
                        rb.add(eval_obs, real_eval_next_obs, eval_actions, eval_rew, eval_terminations)
                    
                    eval_obs = eval_next_obs

            eval_metrics_mean = {}
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                eval_metrics_mean[k] = mean
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
            pbar.set_description(
                f"success_once: {eval_metrics_mean['success_once']:.2f}, "
                f"return: {eval_metrics_mean['return']:.2f}"
            )
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
                logger.add_scalar("time/wall", time.time() - start_time, global_step)
            if args.run.evaluate or args.run.collect_replay_buffer:
                break
            actor.train()

        save_freq = args.run.eval_freq if args.run.save_freq is None else args.run.save_freq
        if args.run.save_model and save_freq > 0 and (global_step - args.algo.training_freq) // save_freq < global_step // save_freq:
            model_path = f"runs/{run_name}/ckpt_{global_step}.pt"
            torch.save({
                'actor': actor.state_dict(),
                'encoder': encoder.state_dict(),
                'target_encoder': target_encoder.state_dict(),
                'qf1': qf1_target.state_dict(),
                'qf2': qf2_target.state_dict(),
                **({'log_alpha': log_alpha} if args.algo.autotune else {}),
                'config': saved_config,
                'global_step': global_step,
                **({'per_env_step': logger.counters['per_env_step'], 'train_episodes': logger.counters['train_episodes']} if logger is not None else {}),
                'q_optimizer': q_optimizer.state_dict(),
                'actor_optimizer': actor_optimizer.state_dict(),
                **({'a_optimizer': a_optimizer.state_dict()} if args.algo.autotune else {}),
            }, model_path)
            print(f"model saved to {model_path}")


        rollout_time = time.perf_counter()
        for local_step in range(steps_per_env):
            global_step += 1 * args.algo.num_envs
            if logger is not None:
                logger.counters["per_env_step"] += 1


            # ALGO LOGIC: put action logic here
            if not args.run.resume and global_step < args.algo.learning_starts:
                actions = torch.tensor([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                with torch.no_grad():
                    actions, cached_noise, noise_repeat_count, noise_repeat_n = _sample_rollout_action(
                        encoder, actor, obs, cached_noise, noise_repeat_count, noise_repeat_n, zeta_cdf, training=False
                    )

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            dones = terminations | truncations

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = copy.deepcopy(next_obs)
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k in real_next_obs:
                    real_next_obs[k][truncations] = infos["final_observation"][k][truncations]
                logger.counters["train_episodes"] += int(done_mask.sum())
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                if args.run.target_success is not None and "success_once" in final_info["episode"]:
                    recent_success.append(final_info["episode"]["success_once"][done_mask].float().mean().item())
                    stop_training = len(recent_success) == recent_success.maxlen and np.mean(recent_success) >= args.run.target_success
            rb.add(obs, real_next_obs, actions, rewards, terminations)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
        
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        pbar.update(args.algo.num_envs * steps_per_env)

        if stop_training:
            print(f"train/success_once over last {len(recent_success)} batches = {np.mean(recent_success):.3f} >= {args.run.target_success}; stopping at step {global_step}")
            break

        # ALGO LOGIC: training.
        if global_step - start_step < args.algo.learning_starts:
            continue
        
        update_time = time.perf_counter()
        learning_has_started = True
        for local_update in range(grad_steps_per_iteration):
            global_update += 1
            data = rb.sample(args.algo.batch_size - n_expert)
            if n_expert:
                data = cat_image_samples(data, expert_rb.sample(n_expert))

            data_obs = data.observations
            data_next_obs = data.next_observations
            obs_all = dict()
            for k in data_obs:
                obs_all[k] = torch.cat([data_obs[k], data_next_obs[k]], dim=0)
            obs_all["rgb"] = random_shift(obs_all["rgb"], args.algo.aug_pad)
            embed_all = encoder(obs_all)
            data_embed = embed_all[:args.algo.batch_size].detach()
            data_next_embed = embed_all[args.algo.batch_size:].detach()

            bootstrap = (~data.dones.bool()).float()

            with torch.no_grad():
                next_state_actions, next_state_log_probs, _ = actor.get_action(data_next_embed, training=False)
                actions_all = torch.cat([data.actions, next_state_actions], dim=0)
                discount = args.algo.gamma ** data.effective_n_steps
                if args.algo.autotune:
                    entropy_coeff = log_alpha.exp()
                else:
                    entropy_coeff = alpha

                target_embed_all = target_encoder(obs_all)
                qf1_target_outputs_all = qf1_target(target_embed_all, actions_all, training=True)
                qf2_target_outputs_all = qf2_target(target_embed_all, actions_all, training=True)
                qf1_target_next_dist = F.softmax(qf1_target_outputs_all, dim=-1).chunk(2, dim=0)[1]
                qf2_target_next_dist = F.softmax(qf2_target_outputs_all, dim=-1).chunk(2, dim=0)[1]

                qf1_target_dist = qf1_target.projection(
                    qf1_target_next_dist,
                    next_state_actions,
                    data.rewards.squeeze(-1) - discount * bootstrap.squeeze(-1) * entropy_coeff * next_state_log_probs.squeeze(-1),
                    bootstrap.squeeze(-1),
                    discount,
                    training=True
                )
                qf2_target_dist = qf2_target.projection(
                    qf2_target_next_dist,
                    next_state_actions,
                    data.rewards.squeeze(-1) - discount * bootstrap.squeeze(-1) * entropy_coeff * next_state_log_probs.squeeze(-1),
                    bootstrap.squeeze(-1),
                    discount,
                    training=True
                )
                qf1_target_values = qf1_target.get_value(qf1_target_dist)
                qf2_target_values = qf2_target.get_value(qf2_target_dist)
                use_qf1 = (qf1_target_values <= qf2_target_values).unsqueeze(-1)
                qf_target_dist = torch.where(use_qf1, qf1_target_dist, qf2_target_dist)

            qf1_outputs_all = qf1(embed_all, actions_all, training=True)
            qf2_outputs_all = qf2(embed_all, actions_all, training=True)
            qf1_outputs = qf1_outputs_all.chunk(2, dim=0)[0]
            qf2_outputs = qf2_outputs_all.chunk(2, dim=0)[0]
            qf1_log_probs = F.log_softmax(qf1_outputs, dim=-1)
            qf2_log_probs = F.log_softmax(qf2_outputs, dim=-1)
            qf1_loss = -torch.sum(qf_target_dist * qf1_log_probs, dim=-1)
            qf2_loss = -torch.sum(qf_target_dist * qf2_log_probs, dim=-1)
            qf_loss = torch.stack([qf1_loss, qf2_loss]).mean(dim=1).sum(dim=0)

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            if not args.algo.freeze_encoder:
                encoder_grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), float("inf"))
            q_optimizer.step()
            q_scheduler.step()

            # update the policy network
            if global_update % args.algo.policy_frequency == 0:
                pi, log_pi, _ = actor.get_action(data_embed, training=True)
                with torch.no_grad():
                    policy_entropy = -log_pi.mean()
                qf1_pi = qf1(data_embed, pi, training=False)
                qf2_pi = qf2(data_embed, pi, training=False)
                qf1_probs = F.softmax(qf1_pi, dim=-1)
                qf2_probs = F.softmax(qf2_pi, dim=-1)
                qf1_value = qf1.get_value(qf1_probs)
                qf2_value = qf2.get_value(qf2_probs)
                min_qf_pi = torch.min(qf1_value, qf2_value)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                actor_scheduler.step()

                if args.algo.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data_embed, training=False)
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    a_scheduler.step()
                    alpha = log_alpha.detach().exp()

                # Normalize the params
                actor.normalize_parameters()

            renorm_and_ema()

        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # Log training-related data
        if (global_step - args.algo.training_freq) // args.run.log_freq < global_step // args.run.log_freq:
            logger.add_scalar("losses/qf1_values", qf1_target_values.mean().item(), global_step)
            logger.add_scalar("losses/qf2_values", qf2_target_values.mean().item(), global_step)
            logger.add_scalar("losses/qf1_loss", qf1_loss.mean().item(), global_step)
            logger.add_scalar("losses/qf2_loss", qf2_loss.mean().item(), global_step)
            logger.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            logger.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            logger.add_scalar("losses/alpha", float(alpha), global_step)
            logger.add_scalar("charts/q_lr", q_optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("charts/policy_lr", actor_optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("metrics/policy_entropy", policy_entropy.item(), global_step)
            sps = int(global_step / (time.time() - start_time))
            logger.add_scalar("charts/SPS", sps, global_step)
            logger.add_scalar("time/wall", time.time() - start_time, global_step)
            logger.add_scalar("time/rollout_time", cumulative_times["rollout_time"], global_step)
            logger.add_scalar("time/update_time", cumulative_times["update_time"], global_step)
            logger.add_scalar("encoder/grad_norm", encoder_grad_norm.item(), global_step)
            logger.add_scalar("encoder/embed_std", embed_all.detach().std(0).mean().item(), global_step)
            with torch.no_grad():
                target_dist = torch.sqrt(sum((p - tp).pow(2).sum() for p, tp in zip(encoder.parameters(), target_encoder.parameters())))
            logger.add_scalar("encoder/target_param_dist", target_dist.item(), global_step)
            logger.writer.add_scalar("charts/per_env_step", logger.counters["per_env_step"], global_step)
            logger.writer.add_scalar("charts/train_episodes", logger.counters["train_episodes"], global_step)
            logger.add_scalar("charts/expert_frac", n_expert / args.algo.batch_size, global_step)
            samples_per_sec = sps * grad_steps_per_iteration * args.algo.batch_size
            logger.add_scalar("charts/samples_per_sec", samples_per_sec, global_step)
            if args.algo.autotune:
                logger.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    if not args.run.evaluate and not args.run.collect_replay_buffer and args.run.save_model:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save({
            'actor': actor.state_dict(),
            'encoder': encoder.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'qf1': qf1_target.state_dict(),
            'qf2': qf2_target.state_dict(),
            **({'log_alpha': log_alpha} if args.algo.autotune else {}),
            'config': saved_config,
            'global_step': global_step,
            **({'per_env_step': logger.counters['per_env_step'], 'train_episodes': logger.counters['train_episodes']} if logger is not None else {}),
            'q_optimizer': q_optimizer.state_dict(),
            'actor_optimizer': actor_optimizer.state_dict(),
            **({'a_optimizer': a_optimizer.state_dict()} if args.algo.autotune else {}),
        }, model_path)
        print(f"model saved to {model_path}")
        writer.close()
    
    if args.run.collect_replay_buffer:
        buffer_path = f"runs/{run_name}/replay_buffer.pt"
        os.makedirs(os.path.dirname(buffer_path), exist_ok=True)
        torch.save({
            'buffer': rb.export(),
            'metadata': {
                'config': saved_config,
                'checkpoint': args.run.checkpoint,
                'policy': 'deterministic_mean',
                'num_transitions': int((rb.buffer_size if rb.full else rb.pos) * rb.n_envs),
                'num_episodes': int(num_episodes),
                'success_once': float(eval_metrics_mean['success_once']),
                'return': float(eval_metrics_mean['return']),
            },
        }, buffer_path)
        print(f"replay buffer saved to {buffer_path}")

    envs.close()
