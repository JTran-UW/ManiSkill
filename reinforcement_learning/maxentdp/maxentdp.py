"""MaxEntDP (Maximum Entropy RL with Diffusion Policy) on ManiSkill, built on the FastSAC scaffolding (fastsac.py):
same env/eval/replay/expert-mixing/checkpoint plumbing and distributional critics, diffusion actor trained with
Q-weighted Noise Estimation (QNE) and a denoising-error log-prob estimate for the entropy backup."""
from collections import defaultdict, deque
from dataclasses import dataclass, field
import os
import random
import time
from typing import Optional, Tuple

import tqdm

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tyro

from reinforcement_learning.fastsac.networks.fastsac_critic import DistributionalQNetwork
from reinforcement_learning.maxentdp.networks.diffusion_actor import DiffusionActor, qne_target
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
    num_envs: int = 4096
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments"""
    num_eval_default_envs: int = 64
    """with --env.kwargs reset_datasets, extra eval envs on the env's default resets"""
    buffer_size: int = 4_096_000
    """the replay memory buffer size"""
    gamma: float = 0.97
    """the discount factor gamma"""
    tau: float = 0.125
    """target smoothing coefficient"""
    batch_size: int = 8192
    """critic batch size"""
    actor_batch_size: Optional[int] = 2048
    """states per QNE actor update (first rows of the critic batch; None = full batch)"""
    learning_starts: int = 40_960
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """actor learning rate"""
    q_lr: float = 3e-4
    """critic learning rate"""
    alpha_lr: float = 3e-4
    """the learning rate of alpha (with --autotune)"""
    policy_frequency: int = 4
    """actor update every N critic updates"""
    target_network_frequency: int = 1
    """the frequency of updates for the target networks"""
    alpha: float = 0.1
    """MaxEnt temperature: QNE softmax(Q / alpha) and the entropy backup coefficient"""
    autotune: bool = False
    """tune alpha toward target entropy -act_dim * target_entropy_ratio using the next-state log-prob estimates"""
    target_entropy_ratio: float = 0.0
    """ratio of target entropy (with --autotune)"""
    alpha_min: float = 1e-4
    """lower clamp on alpha (with --autotune)"""
    training_freq: int = 4096
    """training frequency (in env steps)"""
    utd: float = 0.002
    """update to data ratio"""
    n_step_returns: int = 1
    """n-step returns"""
    v_min: float = -20.0
    """minimum of critic distribution"""
    v_max: float = 20.0
    """maximum of critic distribution"""
    obs_normalization: bool = True
    """use obs normalization"""
    weight_decay: float = 0.001
    """the weight decay of the optimizers"""
    max_grad_norm: float = 0.0
    """clip actor/critic grad norm (0 = off)"""
    # diffusion actor
    diffusion_steps: int = 20
    """denoising steps of the DDPM sampler"""
    actor_hidden_dims: Tuple[int, ...] = (256, 256)
    """eps-network hidden widths"""
    actor_layer_norm: bool = False
    """LayerNorm after each eps-network hidden layer"""
    qne_samples: int = 64
    """K posterior action candidates per state for QNE (paper 500)"""
    qne_bf16: bool = False
    """score the QNE candidates with the critics under bf16 autocast (no-grad path)"""
    qne_policy_frac: float = 0.0
    """fraction of the QNE actor batch whose actions are fresh policy samples instead of replay actions"""
    qne_beta: float = 0.0
    """> 0: scale-free QNE weights softmax(z-scored Q / beta) per state (alpha then only scales the entropy backup)"""
    actor_q: str = "min_target"
    """critic used for QNE weights: min_target | mean_target | mean_online | target1 (qf1_target only, half the scoring cost)"""
    backup_entropy: bool = True
    """subtract alpha * log pi(a'|s') in the critic target"""
    logp_samples: int = 8
    """noise samples for the log-prob estimate (paper 50)"""
    logp_steps: int = 10
    """diffusion times for the log-prob estimate (paper 20)"""
    eval_mode: str = "sample"
    """sample (stochastic diffusion policy) | select (best of --eval-candidates by Q, paper default) | det (no per-step noise)"""
    eval_candidates: int = 10
    """candidates for --eval-mode select"""
    compile: bool = False
    """torch.compile the actor sampler"""
    tf32: bool = True
    """TF32 matmuls"""
    expert_buffer: Optional[str] = None
    """replay_buffer.pt from fastsac.py --collect-replay-buffer to mix into every batch"""
    expert_ratio: float = 0.0
    """fraction of each batch drawn from --expert-buffer"""
    auto_resume: bool = False
    """resume from the newest loadable runs/<exp_name>/ckpt_*.pt if one exists (preemptible jobs)"""
    wandb_group: str = "MaxEntDP"
    """the group of the run for wandb"""


ALGO_MODEL_FIELDS = ("v_min", "v_max", "obs_normalization", "diffusion_steps", "actor_hidden_dims", "actor_layer_norm")


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


def build_actor(env, algo: AlgoArgs, device):
    return DiffusionActor(
        int(np.prod(env.single_observation_space.shape)), int(np.prod(env.single_action_space.shape)),
        hidden_dims=tuple(algo.actor_hidden_dims), layer_norm=algo.actor_layer_norm, T=algo.diffusion_steps,
    ).to(device)


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.algo.auto_resume and args.run.exp_name and args.run.checkpoint is None:
        import glob
        found = sorted(glob.glob(f"runs/{args.run.exp_name}/ckpt_*.pt"), key=lambda p: int(p.rsplit("_", 1)[1][:-3]))
        for path in reversed(found):
            try:
                torch.load(path, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"[auto_resume] skipping unreadable {path}: {e}")
                continue
            args.run.checkpoint, args.run.resume = path, True
            print(f"[auto_resume] resuming from {path}")
            break

    ckpt = None
    if args.run.checkpoint is not None:
        ckpt = torch.load(args.run.checkpoint, weights_only=False)
        if args.run.config_from_checkpoint and ckpt.get("config") is not None:
            notes = restore_from_saved(args.env, ckpt["config"]["env"], ENV_TASK_FIELDS)
            notes += restore_from_saved(args.algo, ckpt["config"]["algo"], ALGO_MODEL_FIELDS)
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

    random.seed(args.run.seed)
    np.random.seed(args.run.seed)
    torch.manual_seed(args.run.seed)
    torch.backends.cudnn.deterministic = args.run.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.run.cuda else "cpu")
    if args.algo.tf32:
        torch.set_float32_matmul_precision("high")

    ####### Environment setup #######
    env_kwargs = gym_make_kwargs(args.env)
    reset_datasets = env_kwargs.get("reset_datasets")
    dataset_paths = [p for p in reset_datasets.split(",") if p] if isinstance(reset_datasets, str) else list(reset_datasets or [])
    dataset_names = [os.path.basename(os.path.dirname(p)) for p in dataset_paths]
    envs = gym.make(args.env.id, num_envs=args.algo.num_envs if not args.run.evaluate else 1, reconfiguration_freq=args.env.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env.id, num_envs=args.algo.num_eval_envs, reconfiguration_freq=args.env.eval_reconfiguration_freq, human_render_camera_configs=dict(shader_pack="default"), **env_kwargs)
    if args.env.init_state_path is not None:
        init_state = load_init_state(args.env.init_state_path, args.env.init_state_index)
        envs = FixedInitStateWrapper(envs, init_state)
        if args.env.eval_init_state_path is None:
            eval_envs = FixedInitStateWrapper(eval_envs, init_state)
    if args.env.eval_init_state_path is not None:
        eval_envs = FixedInitStateWrapper(eval_envs, load_init_state(args.env.eval_init_state_path, args.env.eval_init_state_index))
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
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.run.save_trajectory, save_video=args.run.capture_video and args.run.capture_eval_video, trajectory_name="trajectory", max_steps_per_video=args.env.num_eval_steps, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.algo.num_envs, ignore_terminations=not args.env.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.algo.num_eval_envs, ignore_terminations=not args.env.eval_partial_reset, record_metrics=True)
    eval_default_envs = None
    if dataset_paths:
        default_kwargs = {k: v for k, v in env_kwargs.items() if k not in ("reset_datasets", "reset_dataset_weights")}
        eval_default_envs = ManiSkillVectorEnv(
            gym.make(args.env.id, num_envs=args.algo.num_eval_default_envs, reconfiguration_freq=args.env.eval_reconfiguration_freq, **default_kwargs),
            args.algo.num_eval_default_envs, ignore_terminations=not args.env.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    assert np.allclose(envs.single_action_space.high, 1) and np.allclose(envs.single_action_space.low, -1), "diffusion actor assumes actions in [-1, 1]"

    saved_config = config_dict(run=args.run, env=args.env, algo=args.algo)
    logger = None
    if not args.run.evaluate:
        if args.run.track:
            import wandb
            config = {**saved_config, "derived": dict(grad_steps_per_iteration=grad_steps_per_iteration, steps_per_env=steps_per_env, max_episode_steps=max_episode_steps)}
            config["env"]["gym_make_kwargs"] = env_kwargs
            if args.run.wandb_run_id is not None:
                wandb.init(id=args.run.wandb_run_id, resume="must", project=args.run.wandb_project_name, entity=args.run.wandb_entity)
                wandb.config.update(config, allow_val_change=True)
            else:
                wandb.init(project=args.run.wandb_project_name, entity=args.run.wandb_entity, config=config, name=run_name,
                           save_code=True, group=args.algo.wandb_group, tags=["maxentdp"])
            wandb.define_metric("per_env_step")
            wandb.define_metric("*", step_metric="per_env_step")
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join(
            [f"|{section}.{key}|{value}|" for section, values in saved_config.items() for key, value in values.items()])))
        logger = Logger(log_wandb=args.run.track, tensorboard=writer)

    if args.algo.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=envs.single_observation_space.shape, device=device)
        normalize_obs = obs_normalizer.forward
    else:
        normalize_obs = lambda x, update=True: x

    actor = build_actor(envs, args.algo, device)
    qf1 = DistributionalQNetwork(envs, v_min=args.algo.v_min, v_max=args.algo.v_max).to(device)
    qf2 = DistributionalQNetwork(envs, v_min=args.algo.v_min, v_max=args.algo.v_max).to(device)
    qf1_target = DistributionalQNetwork(envs, v_min=args.algo.v_min, v_max=args.algo.v_max).to(device)
    qf2_target = DistributionalQNetwork(envs, v_min=args.algo.v_min, v_max=args.algo.v_max).to(device)
    if ckpt is not None:
        actor.load_state_dict(ckpt["actor"])
        qf1.load_state_dict(ckpt["qf1"])
        qf2.load_state_dict(ckpt["qf2"])
        if args.algo.obs_normalization:
            obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.AdamW(list(qf1.parameters()) + list(qf2.parameters()), lr=args.algo.q_lr, weight_decay=args.algo.weight_decay, betas=(0.9, 0.95))
    actor_optimizer = optim.AdamW(list(actor.parameters()), lr=args.algo.policy_lr, weight_decay=args.algo.weight_decay, betas=(0.9, 0.95))

    act_dim = int(np.prod(envs.single_action_space.shape))
    log_alpha = torch.tensor([np.log(args.algo.alpha)], dtype=torch.float32, requires_grad=True, device=device)
    a_optimizer = optim.AdamW([log_alpha], lr=args.algo.alpha_lr, betas=(0.9, 0.95)) if args.algo.autotune else None
    target_entropy = -act_dim * args.algo.target_entropy_ratio

    start_step = 0
    if args.run.resume:
        start_step = ckpt.get("global_step", args.run.resume_step)
        if "log_alpha" in ckpt:
            log_alpha.data.copy_(ckpt["log_alpha"].to(device))
        if not args.run.reset_optimizers:
            for name, opt in [("q_optimizer", q_optimizer), ("actor_optimizer", actor_optimizer), ("a_optimizer", a_optimizer)]:
                if opt is not None and name in ckpt:
                    opt.load_state_dict(ckpt[name])
        if logger is not None and "per_env_step" in ckpt:
            logger.counters["per_env_step"] = ckpt["per_env_step"]
            logger.counters["train_episodes"] = ckpt.get("train_episodes", 0)
        print(f"[resume] step {start_step}, alpha {log_alpha.exp().item():.4g}")

    def alpha_value():
        return log_alpha.detach().exp().clamp(min=args.algo.alpha_min)

    def q_value(net, obs, actions):
        return net.get_value(F.softmax(net(obs, actions), dim=-1))

    def actor_q(obs, actions):
        if args.algo.actor_q == "target1":
            return q_value(qf1_target, obs, actions)
        if args.algo.actor_q == "mean_online":
            return 0.5 * (q_value(qf1, obs, actions) + q_value(qf2, obs, actions))
        q1, q2 = q_value(qf1_target, obs, actions), q_value(qf2_target, obs, actions)
        return torch.min(q1, q2) if args.algo.actor_q == "min_target" else 0.5 * (q1 + q2)

    sample_fn = torch.compile(actor.sample) if args.algo.compile else actor.sample

    @torch.no_grad()
    def act(normed_obs, mode="sample"):
        if mode == "select":
            n = args.algo.eval_candidates
            B = normed_obs.shape[0]
            obs_rep = normed_obs.repeat_interleave(n, 0)
            cands = sample_fn(obs_rep)
            q = actor_q(obs_rep, cands).view(B, n)
            return cands.view(B, n, -1)[torch.arange(B, device=normed_obs.device), q.argmax(1)]
        if mode == "det":
            return actor.sample(normed_obs, stochastic=False)
        return sample_fn(normed_obs)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(args.algo.buffer_size, envs.single_observation_space, envs.single_action_space, device,
                      n_envs=args.algo.num_envs, n_steps=args.algo.n_step_returns, gamma=args.algo.gamma)
    expert_rb = None
    if args.algo.expert_buffer and args.algo.expert_ratio > 0 and not args.run.evaluate:
        expert = torch.load(args.algo.expert_buffer, weights_only=False)
        meta = expert["metadata"]
        print(f"[expert] {args.algo.expert_buffer}: {meta.get('num_transitions')} transitions / {meta.get('num_episodes')} episodes, "
              f"success {meta.get('success_once')}, policy {meta.get('policy')}, from {meta.get('checkpoint')}")
        if expert["buffer"]["observations"].shape[-1] != envs.single_observation_space.shape[0] or \
           expert["buffer"]["actions"].shape[-1] != act_dim:
            raise ValueError("expert buffer obs/action dims differ from this env")
        expert_rb = ReplayBuffer.from_export(expert["buffer"], device, n_steps=args.algo.n_step_returns, gamma=args.algo.gamma)
        del expert

    obs, info = envs.reset(seed=args.run.seed)
    eval_envs.reset(seed=args.run.seed)
    global_step = start_step
    global_update = 0
    recent_success = deque(maxlen=5)
    stop_training = False
    pbar = tqdm.tqdm(total=args.run.total_timesteps, initial=start_step)
    cumulative_times = defaultdict(float)
    start_time = time.time()
    eval_mode = "sample" if args.run.eval_stochastic else args.algo.eval_mode

    def rollout_eval(venvs, prefix, global_step, by_dataset=False):
        """Run one eval batch; logs episode metrics plus per-episode max of any info['success_*'] (e.g. which goal)."""
        o, _ = venvs.reset()
        metrics, ids = defaultdict(list), []
        once = {}
        for _ in range(args.env.num_eval_steps):
            o, _, _, _, infos = venvs.step(act(normalize_obs(o, update=False), eval_mode))
            for k, v in infos.items():
                if k.startswith("success_") and isinstance(v, torch.Tensor) and v.dtype == torch.bool:
                    once[k] = once.get(k, torch.zeros_like(v)) | v
            if "final_info" in infos:
                m = infos["_final_info"]
                for k, v in infos["final_info"]["episode"].items():
                    metrics[k].append(v[m])
                if by_dataset and "reset_dataset" in infos["final_info"]:
                    ids.append(infos["final_info"]["reset_dataset"][m])
        means = {k: torch.cat(v).float().mean() for k, v in metrics.items()}
        means.update({f"{k}_once": v.float().mean() for k, v in once.items()})
        if logger is not None:
            for k, v in means.items():
                logger.add_scalar(f"{prefix}/{k}", v, global_step)
            if ids:
                ids = torch.cat(ids)
                for i, name in enumerate(dataset_names):
                    sel = ids == i
                    if sel.any():
                        for k in ("success_once", "return"):
                            logger.add_scalar(f"{prefix}/{name}/{k}", torch.cat(metrics[k])[sel].float().mean(), global_step)
        return means

    def run_eval(global_step):
        actor.eval()
        stime = time.perf_counter()
        prefix = "eval_init_state" if args.env.eval_init_state_path else ("eval_datasets" if dataset_paths else "eval")
        means = rollout_eval(eval_envs, prefix, global_step, by_dataset=bool(dataset_paths))
        if eval_default_envs is not None:
            means = rollout_eval(eval_default_envs, "eval", global_step)
        pbar.set_description(f"success_once: {means['success_once']:.2f}, return: {means['return']:.2f}")
        if logger is not None:
            logger.add_scalar("time/eval_time", time.perf_counter() - stime, global_step)
        actor.train()
        return means

    def save(path):
        torch.save({
            "actor": actor.state_dict(), "qf1": qf1_target.state_dict(), "qf2": qf2_target.state_dict(),
            **({"obs_normalizer": obs_normalizer.state_dict()} if args.algo.obs_normalization else {}),
            "log_alpha": log_alpha.detach(), "config": saved_config, "global_step": global_step,
            **({"per_env_step": logger.counters["per_env_step"], "train_episodes": logger.counters["train_episodes"]} if logger is not None else {}),
            "q_optimizer": q_optimizer.state_dict(), "actor_optimizer": actor_optimizer.state_dict(),
            **({"a_optimizer": a_optimizer.state_dict()} if a_optimizer is not None else {}),
        }, path)
        print(f"model saved to {path}")

    if args.run.evaluate:
        print(run_eval(global_step))
        envs.close()
        raise SystemExit

    stats = {}
    while global_step < args.run.total_timesteps:
        if args.run.eval_freq > 0 and (global_step - args.algo.training_freq) // args.run.eval_freq < global_step // args.run.eval_freq:
            run_eval(global_step)
        save_freq = args.run.eval_freq if args.run.save_freq is None else args.run.save_freq
        if args.run.save_model and save_freq > 0 and (global_step - args.algo.training_freq) // save_freq < global_step // save_freq:
            save(f"runs/{run_name}/ckpt_{global_step}.pt")

        rollout_time = time.perf_counter()
        for local_step in range(steps_per_env):
            global_step += args.algo.num_envs
            logger.counters["per_env_step"] += 1
            if not args.run.resume and global_step < args.algo.learning_starts:
                actions = torch.rand((envs.num_envs, act_dim), device=device) * 2 - 1
            else:
                actions = act(normalize_obs(obs, update=False))
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            real_next_obs = next_obs.clone()
            if "final_info" in infos:
                final_info, done_mask = infos["final_info"], infos["_final_info"]
                real_next_obs[truncations] = infos["final_observation"][truncations]
                logger.counters["train_episodes"] += int(done_mask.sum())
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                if dataset_paths:
                    ids = final_info["reset_dataset"]
                    for i, name in enumerate(dataset_names):
                        m = done_mask & (ids == i)
                        if m.any():
                            logger.add_scalar(f"train_by_reset/{name}/success_once", final_info["episode"]["success_once"][m].float().mean(), global_step)
                if args.run.target_success is not None:
                    recent_success.append(final_info["episode"]["success_once"][done_mask].float().mean().item())
                    stop_training = len(recent_success) == recent_success.maxlen and np.mean(recent_success) >= args.run.target_success
            rb.add(obs, real_next_obs, actions, rewards, terminations, truncations, infos)
            obs = next_obs
        cumulative_times["rollout_time"] += time.perf_counter() - rollout_time
        pbar.update(args.algo.num_envs * steps_per_env)
        if stop_training:
            print(f"train/success_once over last {len(recent_success)} batches >= {args.run.target_success}; stopping at step {global_step}")
            break
        if global_step - start_step < args.algo.learning_starts:
            continue

        update_time = time.perf_counter()
        for local_update in range(grad_steps_per_iteration):
            global_update += 1
            data = sample_mixed(rb, expert_rb, args.algo.batch_size, args.algo.expert_ratio)
            data_obs = normalize_obs(data.observations)
            data_next_obs = normalize_obs(data.next_observations)
            bootstrap = (~data.dones.bool()).float().squeeze(-1)
            alpha = alpha_value()

            with torch.no_grad():
                next_actions = sample_fn(data_next_obs)
                if args.algo.backup_entropy or args.algo.autotune:
                    next_log_probs = actor.log_prob(data_next_obs, next_actions, args.algo.logp_samples, args.algo.logp_steps)
                else:
                    next_log_probs = torch.zeros(data_next_obs.shape[0], device=device)
                discount = args.algo.gamma ** data.effective_n_steps
                soft = alpha * next_log_probs if args.algo.backup_entropy else torch.zeros_like(next_log_probs)
                target_rewards = data.rewards.squeeze(-1) - discount * bootstrap * soft
                qf1_target_dist = qf1_target.projection(data_next_obs, next_actions, target_rewards, bootstrap, discount)
                qf2_target_dist = qf2_target.projection(data_next_obs, next_actions, target_rewards, bootstrap, discount)
                qf1_target_values = qf1_target.get_value(qf1_target_dist)

            qf1_loss = -torch.sum(qf1_target_dist * F.log_softmax(qf1(data_obs, data.actions), dim=-1), dim=-1)
            qf2_loss = -torch.sum(qf2_target_dist * F.log_softmax(qf2(data_obs, data.actions), dim=-1), dim=-1)
            qf_loss = qf1_loss.mean() + qf2_loss.mean()
            q_optimizer.zero_grad()
            qf_loss.backward()
            if args.algo.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(list(qf1.parameters()) + list(qf2.parameters()), args.algo.max_grad_norm)
            q_optimizer.step()

            if global_update % args.algo.policy_frequency == 0:
                n = args.algo.actor_batch_size or data_obs.shape[0]
                a_obs, a_act = data_obs[:n], data.actions[:n].clamp(-1, 1)
                n_pol = int(round(n * args.algo.qne_policy_frac))
                if n_pol > 0:
                    with torch.no_grad():
                        a_act = torch.cat([sample_fn(a_obs[:n_pol]), a_act[n_pol:]])
                noisy, a1, a2, log_snr = actor.noise_actions(a_act)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.algo.qne_bf16):
                    eps_target, q_cand, w = qne_target(noisy, a1, a2, a_obs, actor_q, args.algo.qne_samples, alpha, args.algo.qne_beta)
                eps_target, q_cand, w = eps_target.float(), q_cand.float(), w.float()
                eps_pred = actor(a_obs, noisy, log_snr)
                actor_loss = ((eps_pred - eps_target) ** 2).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                if args.algo.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), args.algo.max_grad_norm)
                actor_optimizer.step()
                if args.algo.autotune:
                    alpha_loss = -(log_alpha * (next_log_probs.mean() + target_entropy)).mean()
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    stats["alpha_loss"] = alpha_loss.detach()
                stats.update(actor_loss=actor_loss.detach(), qne_w_max=w.max(1).values.mean(),
                             qne_ess=(1.0 / (w ** 2).sum(1)).mean(), qne_q_std=q_cand.std(1).mean(),
                             eps_target_norm=eps_target.norm(dim=-1).mean())

            if global_update % args.algo.target_network_frequency == 0:
                with torch.no_grad():
                    for p, tp in zip(list(qf1.parameters()) + list(qf2.parameters()), list(qf1_target.parameters()) + list(qf2_target.parameters())):
                        tp.lerp_(p, args.algo.tau)
        cumulative_times["update_time"] += time.perf_counter() - update_time

        if (global_step - args.algo.training_freq) // args.run.log_freq < global_step // args.run.log_freq:
            logger.add_scalar("losses/qf1_values", qf1_target_values.mean().item(), global_step)
            logger.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            logger.add_scalar("losses/alpha", alpha.item(), global_step)
            logger.add_scalar("metrics/next_log_prob", next_log_probs.mean().item(), global_step)
            for k, v in stats.items():
                logger.add_scalar(f"losses/{k}" if "loss" in k else f"metrics/{k}", float(v), global_step)
            elapsed = time.time() - start_time
            logger.add_scalar("charts/SPS", int((global_step - start_step) / elapsed), global_step)
            logger.add_scalar("time/rollout_frac", cumulative_times["rollout_time"] / elapsed, global_step)
            logger.add_scalar("time/update_frac", cumulative_times["update_time"] / elapsed, global_step)
            logger.add_scalar("time/elapsed_min", elapsed / 60, global_step)
            logger.add_scalar("charts/expert_ratio", args.algo.expert_ratio if expert_rb is not None else 0.0, global_step)

    if args.run.save_model:
        run_eval(global_step)
        save(f"runs/{run_name}/final_ckpt.pt")
    logger.writer.close()
    envs.close()
