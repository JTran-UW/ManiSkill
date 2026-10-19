from collections import defaultdict
from dataclasses import dataclass, field
import os
import random
import time

import tqdm

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tyro

from reinforcement_learning.stream_ac.networks.stream_ac_actor import Actor
from reinforcement_learning.stream_ac.networks.stream_ac_critic import Critic
from reinforcement_learning.stream_ac.utils.sparse_init import sparse_init

from reinforcement_learning.utils.args import (
    ENV_TASK_FIELDS,
    EnvArgs,
    RunArgs,
    config_dict,
    gym_make_kwargs,
    restore_from_saved,
)
from reinforcement_learning.utils.init_states import FixedInitStateWrapper, load_init_state
from reinforcement_learning.utils.utils import EmpiricalNormalization

import mani_skill.envs


@dataclass
class AlgoArgs:
    num_envs: int = 4096
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments"""
    num_steps: int = 16
    """the number of steps to run in each environment per policy rollout"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.97
    """the discount factor gamma (keep equal to stream_ac's so the critic transfers)"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 8
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.1
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""

    # Stream-AC network
    hidden_size: int = 128
    """hidden size"""
    sparsity: float = 0.95
    """global target sparsity of the initial weights"""

    # Logging
    wandb_group: str = "PPO-StreamAC"
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
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
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

    batch_size = args.algo.num_envs * args.algo.num_steps
    minibatch_size = batch_size // args.algo.num_minibatches
    num_iterations = args.run.total_timesteps // batch_size
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
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.run.capture_video or args.run.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.run.evaluate:
            eval_output_dir = f"{os.path.dirname(args.run.checkpoint)}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.run.save_trajectory, save_video=args.run.capture_video and args.run.capture_eval_video, trajectory_name="trajectory", max_steps_per_video=args.env.num_eval_steps, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.algo.num_envs, ignore_terminations=not args.env.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.algo.num_eval_envs, ignore_terminations=not args.env.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    saved_config = config_dict(run=args.run, env=args.env, algo=args.algo)
    logger = None
    if not args.run.evaluate:
        print("Running training")
        if args.run.track:
            import wandb
            config = {**saved_config, "derived": dict(batch_size=batch_size, minibatch_size=minibatch_size, num_iterations=num_iterations, max_episode_steps=max_episode_steps)}
            config["env"]["gym_make_kwargs"] = env_kwargs
            wandb.init(
                project=args.run.wandb_project_name,
                entity=args.run.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=args.algo.wandb_group,
                tags=["ppo", "stream_ac_arch"],
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{section}.{key}|{value}|" for section, values in saved_config.items() for key, value in values.items()])),
        )
        logger = Logger(log_wandb=args.run.track, tensorboard=writer)
    else:
        print("Running evaluation")

    obs_normalizer = EmpiricalNormalization(shape=envs.single_observation_space.shape, device=device)
    actor = Actor(envs, hidden_size=args.algo.hidden_size).to(device)
    critic = Critic(envs, hidden_size=args.algo.hidden_size).to(device)
    if ckpt is not None:
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
    else:
        sparse_init(list(actor.modules()) + list(critic.modules()), target_sparsity=args.algo.sparsity)
    params = list(actor.parameters()) + list(critic.parameters())
    optimizer = optim.Adam(params, lr=args.algo.learning_rate, eps=1e-5)

    def checkpoint_state():
        return {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "obs_normalizer": obs_normalizer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "config": saved_config,
        }

    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    obs = torch.zeros((args.algo.num_steps, args.algo.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.algo.num_steps, args.algo.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.algo.num_steps, args.algo.num_envs), device=device)
    rewards = torch.zeros((args.algo.num_steps, args.algo.num_envs), device=device)
    dones = torch.zeros((args.algo.num_steps, args.algo.num_envs), device=device)
    values = torch.zeros((args.algo.num_steps, args.algo.num_envs), device=device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.run.seed)
    eval_envs.reset(seed=args.run.seed)
    next_obs = obs_normalizer(next_obs, update=not args.run.evaluate)
    next_done = torch.zeros(args.algo.num_envs, device=device)
    save_freq = args.run.eval_freq if args.run.save_freq is None else args.run.save_freq
    pbar = tqdm.tqdm(total=num_iterations * batch_size)

    for iteration in range(1, num_iterations + 1):
        if args.run.eval_freq > 0 and (global_step - batch_size) // args.run.eval_freq < global_step // args.run.eval_freq:
            actor.eval()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.env.num_eval_steps):
                with torch.no_grad():
                    eval_action = actor.get_action(obs_normalizer(eval_obs, update=False), deterministic=True)
                    eval_obs, _, _, _, eval_infos = eval_envs.step(eval_action.clamp(action_low, action_high))
                    if "final_info" in eval_infos:
                        num_episodes += eval_infos["_final_info"].sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            eval_means = {k: torch.stack(v).float().mean() for k, v in eval_metrics.items()}
            print(f"[eval] step {global_step}: {num_episodes} episodes, " + ", ".join(f"{k}={v:.3f}" for k, v in eval_means.items()))
            if logger is not None:
                for k, v in eval_means.items():
                    logger.add_scalar(f"eval/{k}", v, global_step)
            if args.run.evaluate:
                break
            actor.train()

        if args.run.save_model and save_freq > 0 and global_step > 0 and (global_step - batch_size) // save_freq < global_step // save_freq:
            model_path = f"runs/{run_name}/ckpt_{global_step}.pt"
            torch.save(checkpoint_state(), model_path)
            print(f"model saved to {model_path}")

        if args.algo.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.algo.learning_rate

        rollout_time = time.time()
        final_values = torch.zeros((args.algo.num_steps, args.algo.num_envs), device=device)
        for step in range(args.algo.num_steps):
            global_step += args.algo.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _ = actor.get_action_and_log_prob(next_obs)
                values[step] = critic.get_value(next_obs).flatten()
            actions[step] = action
            logprobs[step] = logprob

            raw_next_obs, reward, terminations, truncations, infos = envs.step(action.clamp(action_low, action_high))
            next_obs = obs_normalizer(raw_next_obs)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.algo.reward_scale

            if "final_info" in infos:
                done_mask = infos["_final_info"]
                for k, v in infos["final_info"]["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                with torch.no_grad():
                    final_obs = obs_normalizer(infos["final_observation"][done_mask], update=False)
                    final_values[step, done_mask] = critic.get_value(final_obs).view(-1)
        rollout_time = time.time() - rollout_time

        # bootstrap value according to termination and truncation
        with torch.no_grad():
            next_value = critic.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.algo.num_steps)):
                if t == args.algo.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]
                delta = rewards[t] + args.algo.gamma * real_next_values - values[t]
                advantages[t] = lastgaelam = delta + args.algo.gamma * args.algo.gae_lambda * next_not_done * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        update_time = time.time()
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(args.algo.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = b_inds[start:start + minibatch_size]

                _, newlogprob, entropy = actor.get_action_and_log_prob(b_obs[mb_inds], b_actions[mb_inds])
                newvalue = critic.get_value(b_obs[mb_inds]).view(-1)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.algo.clip_coef).float().mean().item()]

                if args.algo.target_kl is not None and approx_kl > args.algo.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.algo.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.algo.clip_coef, 1 + args.algo.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if args.algo.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.algo.clip_coef, args.algo.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.algo.ent_coef * entropy_loss + v_loss * args.algo.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, args.algo.max_grad_norm)
                optimizer.step()

            if args.algo.target_kl is not None and approx_kl > args.algo.target_kl:
                break
        update_time = time.time() - update_time
        pbar.update(batch_size)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if (global_step - batch_size) // args.run.log_freq < global_step // args.run.log_freq:
            logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
            logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            logger.add_scalar("losses/explained_variance", explained_var, global_step)
            logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar("time/rollout_fps", batch_size / rollout_time, global_step)

    if not args.run.evaluate:
        if args.run.save_model:
            model_path = f"runs/{run_name}/final_ckpt.pt"
            torch.save(checkpoint_state(), model_path)
            print(f"model saved to {model_path}")
        logger.close()
    envs.close()
    eval_envs.close()
