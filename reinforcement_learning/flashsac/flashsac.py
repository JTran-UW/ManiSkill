from collections import defaultdict, deque
from dataclasses import dataclass, field
import statistics
import math
import os
import random
import time
from typing import Optional

import tqdm

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
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

from reinforcement_learning.utils.args import (
    ENV_TASK_FIELDS,
    EnvArgs,
    RunArgs,
    config_dict,
    gym_make_kwargs,
    parse_env_kwargs,
    restore_from_saved,
)
from reinforcement_learning.utils.buffers import ReplayBuffer, sample_mixed
from reinforcement_learning.utils.init_states import FixedInitStateWrapper, load_init_state
from reinforcement_learning.utils.utils import EmpiricalNormalization

import mani_skill.envs

@dataclass
class AlgoArgs:
    num_envs: int = 4000
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments"""
    num_eval_default_envs: int = 64
    """with --env.kwargs reset_datasets, extra eval envs on the env's default resets (eval/*; the dataset mixture logs as eval_datasets/*)"""
    buffer_size: int = 4_096_000
    """the replay memory buffer size"""
    buffer_device: str = "cuda"
    """where the replay buffer is stored. Can be 'cpu' or 'cuda' for GPU"""
    gamma: float = 0.97
    """the discount factor gamma"""
    tau: float = 0.01
    """target smoothing coefficient"""
    batch_size: int = 2048
    """the batch size of sample from the replay memory"""
    learning_starts: int = 40_960
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
    training_freq: int = 4096
    """training frequency (in steps)"""
    utd: float = 0.002
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


def _sample_rollout_action(actor, x, noise, cur_count, cur_n, zeta_cdf, training: bool):
    mean, log_std = actor(x, training=training)
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
    run: tyro.conf.OmitArgPrefixes[RunArgs] = field(default_factory=RunArgs)
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

    grad_steps_per_iteration = int(args.algo.training_freq * args.algo.utd)
    steps_per_env = args.algo.training_freq // args.algo.num_envs
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
    reset_datasets = env_kwargs.get("reset_datasets")
    dataset_paths = [p for p in reset_datasets.split(",") if p] if isinstance(reset_datasets, str) else list(reset_datasets or [])
    dataset_names = [os.path.basename(os.path.dirname(p)) for p in dataset_paths]
    envs = gym.make(args.env.id, num_envs=args.algo.num_envs if not args.run.evaluate else 1, reconfiguration_freq=args.env.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env.id, num_envs=args.algo.num_eval_envs, reconfiguration_freq=args.env.eval_reconfiguration_freq, human_render_camera_configs=dict(shader_pack="default"), **env_kwargs)
    if args.env.init_state_path is not None:
        init_state = load_init_state(args.env.init_state_path, args.env.init_state_index)
        print(f"[init_state] training and eval envs reset to state {init_state['index']} of {init_state['path']}: {init_state['info']}")
        envs = FixedInitStateWrapper(envs, init_state)
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
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=args.run.save_train_trajectory, trajectory_name="trajectory", save_video_trigger=save_video_trigger, max_steps_per_video=max_episode_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.run.save_trajectory, save_video=args.run.capture_video and args.run.capture_eval_video, trajectory_name="trajectory", max_steps_per_video=args.env.num_eval_steps, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.algo.num_envs, ignore_terminations=not args.env.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.algo.num_eval_envs, ignore_terminations=not args.env.eval_partial_reset, record_metrics=True)
    eval_default_envs = None
    if dataset_paths and not args.run.collect_replay_buffer:
        default_kwargs = {k: v for k, v in env_kwargs.items() if k not in ("reset_datasets", "reset_dataset_weights")}
        eval_default_envs = ManiSkillVectorEnv(
            gym.make(args.env.id, num_envs=args.algo.num_eval_default_envs, reconfiguration_freq=args.env.eval_reconfiguration_freq, **default_kwargs),
            args.algo.num_eval_default_envs, ignore_terminations=not args.env.eval_partial_reset, record_metrics=True)
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
        action_scale=torch.ones(envs.single_action_space.shape),
        action_bias=torch.zeros(envs.single_action_space.shape),
        use_tanh=args.algo.use_tanh
    ).to(device)
    zeta_cdf = _build_truncated_zeta_cdf(args.algo.noise_repeat_zeta_mu, args.algo.noise_repeat_zeta_max, device)
    noise_repeat_n = torch.ones((), dtype=torch.int32, device=device)
    noise_repeat_count = torch.zeros((), dtype=torch.int32, device=device)
    cached_noise = torch.randn((args.algo.num_envs,) + envs.single_action_space.shape, device=device)
    qf1 = FlashSACQNetwork(envs, v_min=args.algo.v_min, v_max=args.algo.v_max).to(device)
    qf2 = FlashSACQNetwork(envs, v_min=args.algo.v_min, v_max=args.algo.v_max).to(device)
    qf1_target = FlashSACQNetwork(envs, v_min=args.algo.v_min, v_max=args.algo.v_max).to(device)
    qf2_target = FlashSACQNetwork(envs, v_min=args.algo.v_min, v_max=args.algo.v_max).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.algo.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.algo.policy_lr)

    # Normalize after init
    actor.normalize_parameters()
    qf1.normalize_parameters()
    qf2.normalize_parameters()
    qf1_target.normalize_parameters()
    qf2_target.normalize_parameters()

    # Automatic entropy tuning
    if args.algo.autotune:
        target_entropy = 0.5 * torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item() * torch.log(torch.tensor(2 * torch.pi * torch.e * args.algo.target_entropy_sigma ** 2)).item()
        alpha = torch.tensor(args.algo.alpha)
        log_alpha = torch.tensor([torch.log(alpha)], requires_grad=True, device=device)
        a_optimizer = optim.Adam([log_alpha], lr=args.algo.alpha_lr)
    else:
        alpha = args.algo.alpha

    total_updates = max(1, (args.run.total_timesteps - args.algo.learning_starts) // (args.algo.num_envs * steps_per_env) * grad_steps_per_iteration)

    def cosine_lr(step):
        progress = min(step / total_updates, 1.0)
        return args.algo.lr_end_ratio + (1 - args.algo.lr_end_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    q_scheduler = optim.lr_scheduler.LambdaLR(q_optimizer, cosine_lr)
    actor_scheduler = optim.lr_scheduler.LambdaLR(actor_optimizer, cosine_lr)
    if args.algo.autotune:
        a_scheduler = optim.lr_scheduler.LambdaLR(a_optimizer, cosine_lr)

    start_step = 0

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.algo.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.algo.num_envs if not args.run.collect_replay_buffer else args.algo.num_eval_envs,
        n_steps=args.algo.n_step_returns,
        gamma=args.algo.gamma
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset(seed=args.run.seed) # in Gymnasium, seed is given to reset() instead of seed()
    eval_obs, _ = eval_envs.reset(seed=args.run.seed)
    global_step = start_step
    global_update = 0
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
            eval_ids = []
            num_episodes = 0
            for _ in range(args.env.num_eval_steps):
                with torch.no_grad():
                    eval_actions = actor.get_action(eval_obs, training=False)[2]
                    eval_next_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(eval_actions)

                    real_eval_next_obs = eval_next_obs.clone()
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
                        if "reset_dataset" in eval_infos["final_info"]:
                            eval_ids.append(eval_infos["final_info"]["reset_dataset"])

                        if args.run.collect_replay_buffer:
                            real_eval_next_obs[eval_truncations] = eval_infos["final_observation"][eval_truncations]

                    if args.run.collect_replay_buffer:
                        rb.add(eval_obs, real_eval_next_obs, eval_actions, eval_rew, eval_terminations, eval_truncations, eval_infos)
                    
                    eval_obs = eval_next_obs

            eval_prefix = "eval_datasets" if dataset_paths else "eval"
            eval_metrics_mean = {}
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                eval_metrics_mean[k] = mean
                if logger is not None:
                    logger.add_scalar(f"{eval_prefix}/{k}", mean, global_step)
            if dataset_paths and eval_ids and logger is not None:
                ids = torch.stack(eval_ids).flatten()
                for i, name in enumerate(dataset_names):
                    m = ids == i
                    if m.any():
                        for k in ("success_once", "success_at_end", "return"):
                            logger.add_scalar(f"eval_datasets/{name}/{k}", torch.stack(eval_metrics[k]).flatten()[m].float().mean(), global_step)
            if eval_default_envs is not None:
                default_obs, _ = eval_default_envs.reset()
                default_metrics = defaultdict(list)
                for _ in range(args.env.num_eval_steps):
                    with torch.no_grad():
                        default_obs, _, _, _, default_infos = eval_default_envs.step(actor.get_action(default_obs, training=False)[2])
                        if "final_info" in default_infos:
                            for k, v in default_infos["final_info"]["episode"].items():
                                default_metrics[k].append(v[default_infos["_final_info"]])
                for k, v in default_metrics.items():
                    if logger is not None:
                        logger.add_scalar(f"eval/{k}", torch.cat(v).float().mean(), global_step)
            pbar.set_description(
                f"success_once: {eval_metrics_mean['success_once']:.2f}, "
                f"return: {eval_metrics_mean['return']:.2f}"
            )
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.run.evaluate or args.run.collect_replay_buffer:
                break
            actor.train()

        save_freq = args.run.eval_freq if args.run.save_freq is None else args.run.save_freq
        if args.run.save_model and save_freq > 0 and (global_step - args.algo.training_freq) // save_freq < global_step // save_freq:
            model_path = f"runs/{run_name}/ckpt_{global_step}.pt"
            torch.save({
                'actor': actor.state_dict(),
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
                        actor, obs, cached_noise, noise_repeat_count, noise_repeat_n, zeta_cdf, training=False
                    )

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            dones = terminations | truncations


            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.clone()
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                real_next_obs[truncations] = infos["final_observation"][truncations]
                logger.counters["train_episodes"] += int(done_mask.sum())
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                if dataset_paths:
                    ids = final_info["reset_dataset"]
                    for i, name in enumerate(dataset_names):
                        m = done_mask & (ids == i)
                        if m.any():
                            for k in ("success_once", "success_at_end", "return"):
                                logger.add_scalar(f"train_by_reset/{name}/{k}", final_info["episode"][k][m].float().mean(), global_step)
                if args.run.target_success is not None and "success_once" in final_info["episode"]:
                    recent_success.append(final_info["episode"]["success_once"][done_mask].float().mean().item())
                    stop_training = len(recent_success) == recent_success.maxlen and np.mean(recent_success) >= args.run.target_success
            rb.add(obs, real_next_obs, actions, rewards, terminations, truncations, infos)

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
            data = rb.sample(args.algo.batch_size)

            data_obs = data.observations
            data_next_obs = data.next_observations

            bootstrap = (~data.dones.bool()).float()

            with torch.no_grad():
                next_state_actions, next_state_log_probs, _ = actor.get_action(data_next_obs, training=False)
                discount = args.algo.gamma ** data.effective_n_steps
                if args.algo.autotune:
                    entropy_coeff = log_alpha.exp()
                else:
                    entropy_coeff = alpha

                obs_all = torch.cat([data_obs, data_next_obs], dim=0)
                actions_all = torch.cat([data.actions, next_state_actions], dim=0)

                qf1_target_outputs_all = qf1_target(obs_all, actions_all, training=True)
                qf2_target_outputs_all = qf2_target(obs_all, actions_all, training=True)
                qf1_target_next_dist = F.softmax(qf1_target_outputs_all, dim=-1).chunk(2, dim=0)[1]
                qf2_target_next_dist = F.softmax(qf2_target_outputs_all, dim=-1).chunk(2, dim=0)[1]

                qf1_target_dist = qf1_target.projection(
                    qf1_target_next_dist,
                    next_state_actions,
                    data.rewards.squeeze(-1) - discount * bootstrap.squeeze(-1) * entropy_coeff * next_state_log_probs.squeeze(-1),
                    bootstrap.squeeze(-1),
                    discount,
                    training=True,
                )
                qf2_target_dist = qf2_target.projection(
                    qf2_target_next_dist,
                    next_state_actions,
                    data.rewards.squeeze(-1) - discount * bootstrap.squeeze(-1) * entropy_coeff * next_state_log_probs.squeeze(-1),
                    bootstrap.squeeze(-1),
                    discount,
                    training=True,
                )
                qf1_target_values = qf1_target.get_value(qf1_target_dist)
                qf2_target_values = qf2_target.get_value(qf2_target_dist)
                use_qf1 = (qf1_target_values <= qf2_target_values).unsqueeze(-1)
                qf_target_dist = torch.where(use_qf1, qf1_target_dist, qf2_target_dist)

            qf1_outputs_all = qf1(obs_all, actions_all, training=True)
            qf2_outputs_all = qf2(obs_all, actions_all, training=True)
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
            q_optimizer.step()
            q_scheduler.step()

            # update the policy network
            if global_update % args.algo.policy_frequency == 0:
                pi, log_pi, _ = actor.get_action(data_obs, training=True)
                with torch.no_grad():
                    policy_entropy = -log_pi.mean()
                qf1_pi = qf1(data_obs, pi, training=False)
                qf2_pi = qf2(data_obs, pi, training=False)
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
                        _, log_pi, _ = actor.get_action(data_obs, training=False)
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    a_scheduler.step()
                    alpha = log_alpha.exp().item()

                # Normalize the params
                actor.normalize_parameters()

            qf1.normalize_parameters()
            qf2.normalize_parameters()

            # update the target networks
            if global_step % args.algo.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.algo.tau * param.data + (1 - args.algo.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.algo.tau * param.data + (1 - args.algo.tau) * target_param.data)

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
            logger.add_scalar("losses/alpha", alpha, global_step)
            logger.add_scalar("charts/q_lr", q_optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("charts/policy_lr", actor_optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("metrics/policy_entropy", policy_entropy.item(), global_step)
            sps = int(global_step / (time.time() - start_time))
            logger.add_scalar("charts/SPS", sps, global_step)
            logger.writer.add_scalar("charts/per_env_step", logger.counters["per_env_step"], global_step)
            logger.writer.add_scalar("charts/train_episodes", logger.counters["train_episodes"], global_step)
            # logger.add_scalar("charts/expert_ratio", args.algo.expert_ratio if expert_rb is not None else 0.0, global_step)
            samples_per_sec = sps * grad_steps_per_iteration * args.algo.batch_size
            logger.add_scalar("charts/samples_per_sec", samples_per_sec, global_step)
            if args.algo.autotune:
                logger.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    if not args.run.evaluate and not args.run.collect_replay_buffer and args.run.save_model:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save({
            'actor': actor.state_dict(),
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
