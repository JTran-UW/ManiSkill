from collections import defaultdict, deque
from dataclasses import dataclass, field
import statistics
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

from reinforcement_learning.stream_ac.networks.stream_ac_actor import Actor
from reinforcement_learning.stream_ac.networks.stream_ac_critic import Critic
from reinforcement_learning.stream_ac.utils.sparse_init import sparse_init
from reinforcement_learning.stream_ac.utils.optimizer import StreamingOptimizer

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
    num_envs: int = 1
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments"""
    gamma: float = 0.97
    """the discount factor gamma"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    alpha_lr: float = 3e-4
    """the learning rate of alpha"""

    # Stream-AC stuff
    lamda: float = 0.8
    """lambda"""
    beta: float = 0.99995
    """beta"""
    entropy_coeff: float = 0.01
    """entropy coefficient"""
    hidden_size: int = 128
    """hidden size"""
    sparsity: float = 0.95
    """global target sparsity"""

    # Logging
    wandb_group: str = "Stream-AC"
    """the group of the run for wandb"""


# algo args that change the saved networks; restored from --checkpoint like the env task fields
ALGO_MODEL_FIELDS = ("hidden_size",)


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
            notes = restore_from_saved(args.env, ckpt["config"]["env"], ENV_TASK_FIELDS)
            notes += restore_from_saved(args.algo, ckpt["config"]["algo"], ALGO_MODEL_FIELDS)
            for note in notes:
                print(f"[config] {note}")

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
    envs = ManiSkillVectorEnv(envs, args.algo.num_envs, ignore_terminations=not args.env.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.algo.num_eval_envs, ignore_terminations=not args.env.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    saved_config = config_dict(run=args.run, env=args.env, algo=args.algo)
    if args.run.track:
        import wandb
        config = {**saved_config}
        config["env"]["gym_make_kwargs"] = env_kwargs
        wandb.init(
            project=args.run.wandb_project_name,
            entity=args.run.wandb_entity,
            sync_tensorboard=False,
            config=config,
            name=run_name,
            save_code=True,
            group=args.algo.wandb_group,
            tags=["stream_ac"],
        )
        wandb.define_metric("per_env_step")
        wandb.define_metric("*", step_metric="per_env_step")
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{section}.{key}|{value}|" for section, values in saved_config.items() for key, value in values.items()])),
    )
    logger = Logger(log_wandb=args.run.track, tensorboard=writer)

    obs_normalizer = EmpiricalNormalization(shape=envs.single_observation_space.shape, device=device)

    actor = Actor(envs, hidden_size=args.algo.hidden_size).to(device)
    critic = Critic(envs, hidden_size=args.algo.hidden_size).to(device)
    if ckpt is not None:
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
        print(f"[checkpoint] loaded actor, critic and obs_normalizer from {args.run.checkpoint}")
    else:
        sparse_init(list(actor.modules()) + list(critic.modules()), target_sparsity=args.algo.sparsity)
    actor_optimizer = StreamingOptimizer(actor.parameters(), lr=args.algo.policy_lr, gamma=args.algo.gamma, lamda=args.algo.lamda, beta=args.algo.beta)
    critic_optimizer = StreamingOptimizer(critic.parameters(), lr=args.algo.q_lr, gamma=args.algo.gamma, lamda=args.algo.lamda, beta=args.algo.beta)

    normalize_obs = obs_normalizer.forward

    def checkpoint_state():
        return {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "obs_normalizer": obs_normalizer.state_dict(),
            "global_step": global_step,
            "per_env_step": logger.counters["per_env_step"],
            "train_episodes": logger.counters["train_episodes"],
            "config": saved_config,
        }

    def save_checkpoint(name):
        torch.save(checkpoint_state(), f"runs/{run_name}/{name}")
        print(f"model saved to runs/{run_name}/{name}", flush=True)

    def evaluate():
        actor.eval()
        stime = time.perf_counter()
        eval_obs, _ = eval_envs.reset()
        eval_metrics = defaultdict(list)
        for _ in range(args.env.num_eval_steps):
            with torch.no_grad():
                eval_action = actor.get_action(normalize_obs(eval_obs, update=False), deterministic=True)
                eval_obs, _, _, _, eval_infos = eval_envs.step(eval_action)
                if "final_info" in eval_infos:
                    for k, v in eval_infos["final_info"]["episode"].items():
                        eval_metrics[k].append(v[eval_infos["_final_info"]])
        eval_means = {k: torch.cat(v).float().mean().item() for k, v in eval_metrics.items()}
        for k, v in eval_means.items():
            logger.add_scalar(f"eval/{k}", v, global_step)
        eval_time = time.perf_counter() - stime
        logger.add_scalar("time/eval_time", eval_time, global_step)
        pbar.set_description(f"eval success_once: {eval_means.get('success_once', float('nan')):.2f}, return: {eval_means.get('return', float('nan')):.2f}")
        actor.train()
        return eval_time

    save_freq = args.run.eval_freq if args.run.save_freq is None else args.run.save_freq
    window = defaultdict(float)
    window_steps = 0
    eval_time_total = 0.0
    start_time = time.time()
    pbar = tqdm.tqdm(total=args.run.total_timesteps)

    global_step = 0
    obs, _ = envs.reset(seed=args.run.seed)
    eval_envs.reset(seed=args.run.seed)
    obs = normalize_obs(obs)

    rewards = []
    while global_step < args.run.total_timesteps:
        if args.run.eval_freq > 0 and global_step % args.run.eval_freq == 0:
            eval_time_total += evaluate()

        action, log_prob, entropy = actor.get_action_and_log_prob(obs)
        next_obs, reward, terminated, truncated, info = envs.step(action)
        rewards.append(reward)

        reset = bool((terminated | truncated).any())
        # on auto-reset next_obs is the new episode's first obs; bootstrap from the true final obs
        next_obs_n = normalize_obs(next_obs)
        bootstrap_obs = normalize_obs(info["final_observation"]) if reset else next_obs_n
        done_mask = 1.0 - terminated.float()

        v_s = critic.get_value(obs).squeeze(-1)
        with torch.no_grad():
            v_prime = critic.get_value(bootstrap_obs).squeeze(-1)
            delta = reward + args.algo.gamma * v_prime * done_mask - v_s
            delta_value = float(delta.item())
            delta_sign = 1.0 if delta_value > 0.0 else -1.0 if delta_value < 0.0 else 0.0

        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        v_s.sum().backward()
        (log_prob + args.algo.entropy_coeff * delta_sign * entropy).sum().backward()
        actor_optimizer.step(delta_value, reset=reset)
        critic_optimizer.step(delta_value, reset=reset)

        window["td_error"] += delta_value
        window["abs_td_error"] += abs(delta_value)
        window["value"] += v_s.mean().item()
        window["log_prob"] += log_prob.mean().item()
        window["policy_entropy"] += entropy.mean().item()
        window_steps += 1

        global_step += args.algo.num_envs
        logger.counters["per_env_step"] += 1
        pbar.update(args.algo.num_envs)

        if reset:
            done = info["_final_info"]
            logger.counters["train_episodes"] += int(done.sum())
            for k, v in info["final_info"]["episode"].items():
                logger.add_scalar(f"train/{k}", v[done].float().mean(), global_step)
            print(f"Return: {sum(rewards).item():.2f} success_once: {int(info['final_info']['episode']['success_once'][done].float().mean().item())} step: {global_step}", flush=True)
            rewards = []

        obs = next_obs_n

        if global_step % args.run.log_freq == 0:
            logger.add_scalar("losses/td_error", window["td_error"] / window_steps, global_step)
            logger.add_scalar("losses/abs_td_error", window["abs_td_error"] / window_steps, global_step)
            logger.add_scalar("losses/value", window["value"] / window_steps, global_step)
            logger.add_scalar("losses/log_prob", window["log_prob"] / window_steps, global_step)
            logger.add_scalar("metrics/policy_entropy", window["policy_entropy"] / window_steps, global_step)
            logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time - eval_time_total)), global_step)
            logger.writer.add_scalar("charts/per_env_step", logger.counters["per_env_step"], global_step)
            logger.writer.add_scalar("charts/train_episodes", logger.counters["train_episodes"], global_step)
            window.clear()
            window_steps = 0

        if args.run.save_model and save_freq > 0 and global_step % save_freq == 0:
            save_checkpoint(f"ckpt_{global_step}.pt")

    if args.run.eval_freq > 0:
        evaluate()
    if args.run.save_model:
        save_checkpoint("final_ckpt.pt")
    logger.close()

    envs.close()
    eval_envs.close()
