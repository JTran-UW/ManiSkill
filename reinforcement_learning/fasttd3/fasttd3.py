
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

from reinforcement_learning.fasttd3.networks.fasttd3_actor import Actor
from reinforcement_learning.fasttd3.networks.fasttd3_critic import DistributionalQNetwork

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
    buffer_size: int = 4_096_000
    """the replay memory buffer size"""
    buffer_device: str = "cuda"
    """where the replay buffer is stored. Can be 'cpu' or 'cuda' for GPU"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.1
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 8192
    """the batch size of sample from the reply memory"""
    learning_starts: int = 40_960
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    training_freq: int = 4096
    """training frequency (in steps)"""
    utd: float = 0.002
    """update to data ratio"""

    # Algorithm specific arguments
    n_step_returns: int = 1
    """n-step returns"""
    v_min: float = -20.0
    """minimum of critic distribution"""
    v_max: float = 20.0
    """maximum of critic distribution"""
    policy_noise: float = 0.001
    """the scale of policy noise"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    init_scale: float = 0.01
    """the scale of the initial parameters"""
    std_min: float = 0.001
    """the minimum scale of noise"""
    std_max: float = 0.4
    """the maximum scale of noise"""
    obs_normalization: bool = True
    """use obs normalization"""
    weight_decay: float = 0.001
    """the weight decay of the optimizer"""

    # Logging
    wandb_group: str = "FastTD3"
    """the group of the run for wandb"""


# algo args that change the saved networks; restored from --checkpoint like the env task fields
ALGO_MODEL_FIELDS = ("v_min", "v_max", "obs_normalization")
# actor buffers that depend on num_envs / CLI exploration settings, not on training
ACTOR_RUNTIME_BUFFERS = ("noise_scales", "std_min", "std_max")


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
                    tags=["td3", "walltime_efficient"]
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

    if args.algo.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=envs.single_observation_space.shape, device=device)

    actor = Actor(
        envs,
        args.algo.num_envs,
        args.algo.init_scale,
        std_min=args.algo.std_min,
        std_max=args.algo.std_max
    ).to(device)
    qf1 = DistributionalQNetwork(envs, v_min=args.algo.v_min, v_max=args.algo.v_max).to(device)
    qf2 = DistributionalQNetwork(envs, v_min=args.algo.v_min, v_max=args.algo.v_max).to(device)
    qf1_target = DistributionalQNetwork(envs, v_min=args.algo.v_min, v_max=args.algo.v_max).to(device)
    qf2_target = DistributionalQNetwork(envs, v_min=args.algo.v_min, v_max=args.algo.v_max).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.AdamW(list(qf1.parameters()) + list(qf2.parameters()), lr=args.algo.q_lr, weight_decay=args.algo.weight_decay)
    actor_optimizer = optim.AdamW(list(actor.parameters()), lr=args.algo.policy_lr, weight_decay=args.algo.weight_decay)

    start_step = 0
    if ckpt is not None:
        actor_sd = {k: v for k, v in ckpt["actor"].items() if k not in ACTOR_RUNTIME_BUFFERS}
        missing = set(actor.load_state_dict(actor_sd, strict=False).missing_keys) - set(ACTOR_RUNTIME_BUFFERS)
        if missing:
            raise RuntimeError(f"actor checkpoint is missing {sorted(missing)}")
        qf1.load_state_dict(ckpt["qf1"])
        qf2.load_state_dict(ckpt["qf2"])
        qf1_target.load_state_dict(ckpt.get("qf1_target", ckpt["qf1"]))
        qf2_target.load_state_dict(ckpt.get("qf2_target", ckpt["qf2"]))
        if args.algo.obs_normalization:
            if "obs_normalizer" in ckpt:
                obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
            else:
                print("[checkpoint] WARNING: no obs_normalizer in checkpoint; the actor will see differently-normalized observations")
    if args.run.resume:
        start_step = ckpt.get("global_step", args.run.resume_step)
        if start_step is None:
            raise ValueError("checkpoint stores no global_step; pass --resume-step")
        if not args.run.reset_optimizers:
            for name, opt in [("q_optimizer", q_optimizer), ("actor_optimizer", actor_optimizer)]:
                if name in ckpt:
                    opt.load_state_dict(ckpt[name])
        print(f"[resume] step {start_step}, optimizers restored: "
              f"{[n for n in ('q_optimizer', 'actor_optimizer') if n in ckpt and not args.run.reset_optimizers]}")

    def checkpoint_state():
        return {
            "actor": {k: v for k, v in actor.state_dict().items() if k not in ACTOR_RUNTIME_BUFFERS},
            "qf1": qf1.state_dict(),
            "qf2": qf2.state_dict(),
            "qf1_target": qf1_target.state_dict(),
            "qf2_target": qf2_target.state_dict(),
            **({"obs_normalizer": obs_normalizer.state_dict()} if args.algo.obs_normalization else {}),
            "q_optimizer": q_optimizer.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "global_step": global_step,
            "config": saved_config,
        }

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

    if args.algo.obs_normalization:
        normalize_obs = obs_normalizer.forward

    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset(seed=args.run.seed) # in Gymnasium, seed is given to reset() instead of seed()
    dones = None
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
            num_episodes = 0
            for _ in range(args.env.num_eval_steps):
                with torch.no_grad():
                    if args.algo.obs_normalization:
                        normed_eval_obs = normalize_obs(eval_obs, update=False)
                    else:
                        normed_eval_obs = eval_obs

                    eval_actions = actor.get_action(normed_eval_obs, deterministic=True)
                    eval_next_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(eval_actions)

                    real_eval_next_obs = eval_next_obs.clone()
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)

                        if args.run.collect_replay_buffer:
                            real_eval_next_obs[eval_truncations] = eval_infos["final_observation"][eval_truncations]

                    if args.run.collect_replay_buffer:
                        rb.add(eval_obs, real_eval_next_obs, eval_actions, eval_rew, eval_terminations, eval_truncations, eval_infos)
                    
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
            if args.run.evaluate or args.run.collect_replay_buffer:
                break
            actor.train()

        save_freq = args.run.eval_freq if args.run.save_freq is None else args.run.save_freq
        if args.run.save_model and save_freq > 0 and (global_step - args.algo.training_freq) // save_freq < global_step // save_freq:
            model_path = f"runs/{run_name}/ckpt_{global_step}.pt"
            torch.save(checkpoint_state(), model_path)
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
                if args.algo.obs_normalization:
                    normed_obs = normalize_obs(obs, update=False)
                else:
                    normed_obs = obs

                with torch.no_grad():
                    actions = actor.get_action(normed_obs, dones=dones)
                    actions = actions.clamp(envs.single_action_space.low[0], envs.single_action_space.high[0])

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

            if args.algo.obs_normalization:
                data_obs = normalize_obs(data_obs)
                data_next_obs = normalize_obs(data_next_obs)

            bootstrap = (~data.dones.bool()).float()

            with torch.no_grad():
                clipped_noise = (torch.randn_like(data.actions, device=device) * args.algo.policy_noise).clamp(
                    -args.algo.noise_clip, args.algo.noise_clip
                )

                next_state_actions = (actor(data_next_obs) + clipped_noise).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                discount = args.algo.gamma ** data.effective_n_steps

                qf1_target_dist = qf1_target.projection(
                    data_next_obs,
                    next_state_actions,
                    data.rewards.squeeze(-1),
                    bootstrap.squeeze(-1),
                    discount
                )
                qf2_target_dist = qf2_target.projection(
                    data_next_obs,
                    next_state_actions,
                    data.rewards.squeeze(-1),
                    bootstrap.squeeze(-1),
                    discount,
                )
                qf1_target_values = qf1_target.get_value(qf1_target_dist)
                qf2_target_values = qf2_target.get_value(qf2_target_dist)
                target_value_max = torch.max(qf1_target_values, qf2_target_values)
                target_value_min = torch.min(qf1_target_values, qf2_target_values)

            qf1_outputs = qf1(data_obs, data.actions)
            qf2_outputs = qf2(data_obs, data.actions)
            qf1_log_probs = F.log_softmax(qf1_outputs, dim=-1)
            qf2_log_probs = F.log_softmax(qf2_outputs, dim=-1)
            target_dist = torch.where((qf1_target_values < qf2_target_values).unsqueeze(1), qf1_target_dist, qf2_target_dist)
            qf1_loss = -torch.sum(target_dist * qf1_log_probs, dim=-1)
            qf2_loss = -torch.sum(target_dist * qf2_log_probs, dim=-1)
            qf_loss = torch.stack([qf1_loss, qf2_loss]).mean(dim=1).sum(dim=0)

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_update % args.algo.policy_frequency == 0:
                pi = actor(data_obs)
                qf1_value = qf1.get_value(F.softmax(qf1(data_obs, pi), dim=-1))
                qf2_value = qf2.get_value(F.softmax(qf2(data_obs, pi), dim=-1))
                actor_loss = -torch.min(qf1_value, qf2_value).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

            # update the target network
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
            sps = int(global_step / (time.time() - start_time))
            logger.add_scalar("charts/SPS", sps, global_step)
            logger.writer.add_scalar("charts/per_env_step", logger.counters["per_env_step"], global_step)
            logger.writer.add_scalar("charts/train_episodes", logger.counters["train_episodes"], global_step)
            samples_per_sec = sps * grad_steps_per_iteration * args.algo.batch_size
            logger.add_scalar("charts/samples_per_sec", samples_per_sec, global_step)

    if not args.run.evaluate and not args.run.collect_replay_buffer and args.run.save_model:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(checkpoint_state(), model_path)
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
