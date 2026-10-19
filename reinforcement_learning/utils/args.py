"""Algorithm-agnostic CLI sections shared by the training scripts in reinforcement_learning/.

Scripts compose these with their own algorithm dataclass, e.g.

    @dataclass
    class Args:
        run: tyro.conf.OmitArgPrefixes[RunArgs] = field(default_factory=RunArgs)
        env: EnvArgs = field(default_factory=EnvArgs)
        algo: tyro.conf.OmitArgPrefixes[AlgoArgs] = field(default_factory=AlgoArgs)

Run and algo flags stay unprefixed (--seed, --gamma); env flags live under --env.* and
task-specific constructor kwargs go through --env.kwargs KEY VALUE [KEY VALUE ...].
"""

import ast
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class RunArgs:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "maniskill_pusht"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    capture_eval_video: bool = True
    """with --capture-video, also record eval videos (--no-capture-eval-video keeps only training videos)"""
    save_trajectory: bool = False
    """whether to save trajectory data into the `videos` folder"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    collect_replay_buffer: bool = False
    """if toggled, only runs data collection with the given model checkpoint and saves them into a replay buffer"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    resume: bool = False
    """continue training from --checkpoint: keep its global step, alpha and (if saved) optimizer states; warm up with the loaded policy instead of random actions"""
    resume_step: Optional[int] = None
    """global step to continue from when the checkpoint does not store one (older checkpoints)"""
    reset_optimizers: bool = False
    """with --resume, don't restore the critic/actor/alpha optimizers from the checkpoint"""
    wandb_run_id: Optional[str] = None
    """resume logging into this existing wandb run (implies --track)"""
    target_success: Optional[float] = None
    """stop training once train/success_once, averaged over the last 5 logged episode batches, reaches this"""
    config_from_checkpoint: bool = True
    """restore the task definition (env id/modes/kwargs) and model-defining algo args from --checkpoint; flags set to non-default values on the CLI still win"""
    total_timesteps: int = 40_000_000
    """total timesteps of the experiments"""
    log_freq: int = 1_000
    """logging frequency in terms of environment steps"""
    eval_freq: int = 1_000_000
    """evaluation frequency in terms of environment steps"""
    eval_every_episodes: int = 0
    """also evaluate before training and after every N training episodes (0 = off)"""
    eval_stochastic: bool = False
    """evaluate with sampled actions instead of the deterministic mean"""
    save_freq: Optional[int] = None
    """checkpoint frequency in terms of environment steps (None = same as --eval-freq, 0 = final checkpoint only)"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of episodes"""
    save_train_trajectory: bool = False
    """with --save-train-video-freq, also save the training episodes' env states/obs/actions to train_videos/trajectory.h5"""


@dataclass
class EnvArgs:
    id: str = "PickCube-v1"
    """the ManiSkill environment id"""
    obs_mode: str = "state"
    """observation mode passed to gym.make"""
    control_mode: Optional[str] = "pd_ee_delta_pose"
    """the control mode to use for the environment (None = env default)"""
    reward_mode: Optional[str] = None
    """reward mode passed to gym.make (None = env default, normalized_dense for most tasks)"""
    sim_backend: str = "gpu"
    """simulation backend passed to gym.make"""
    env_vectorization: str = "gpu"
    """the type of environment vectorization to use"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_eval_steps: Optional[int] = None
    """steps per evaluation rollout (None = the env's max_episode_steps)"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    init_state_path: Optional[str] = None
    """training and eval envs reset to one saved state from this file (scripts/save_init_states.py)"""
    init_state_index: int = 0
    """which saved state in --env.init-state-path to reset to"""
    eval_init_state_path: Optional[str] = None
    """eval envs only: reset to one saved state from this file (overrides --env.init-state-path for eval)"""
    eval_init_state_index: int = 0
    """which saved state in --env.eval-init-state-path eval envs reset to"""
    kwargs: Dict[str, str] = field(default_factory=dict)
    """task-specific kwargs forwarded to gym.make, e.g. --env.kwargs tee_ballast_pos_x 0.1 tee_ballast_pos_y -0.0375 tee_ballast_mass_ratio 0.25 (values parsed as Python literals)"""


# fields that define the task itself (vs. how many envs / how long to evaluate)
ENV_TASK_FIELDS = ("id", "obs_mode", "control_mode", "reward_mode", "kwargs")


def parse_env_kwargs(raw: Dict[str, str]) -> Dict[str, Any]:
    """Parse --env.kwargs values as Python literals; unparseable values stay strings."""
    parsed = {}
    for key, value in raw.items():
        try:
            parsed[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed[key] = value
    return parsed


def gym_make_kwargs(env: EnvArgs) -> Dict[str, Any]:
    """Keyword arguments for gym.make shared by the train and eval envs."""
    kwargs = dict(obs_mode=env.obs_mode, render_mode="rgb_array", sim_backend=env.sim_backend)
    if env.control_mode is not None:
        kwargs["control_mode"] = env.control_mode
    if env.reward_mode is not None:
        kwargs["reward_mode"] = env.reward_mode
    overlap = set(kwargs) & set(env.kwargs)
    if overlap:
        raise ValueError(f"set {sorted(overlap)} via their --env.* flags, not --env.kwargs")
    kwargs.update(parse_env_kwargs(env.kwargs))
    return kwargs


def restore_from_saved(current: Any, saved: Dict[str, Any], fields: Iterable[str]) -> List[str]:
    """Copy `fields` from a saved asdict() into the `current` dataclass.

    A field the user set to a non-default value on the CLI is kept (and reported); `kwargs`
    dicts are merged with CLI keys taking precedence. Returns human-readable notes.
    """
    defaults = type(current)()
    notes = []
    for name in fields:
        if name not in saved:
            continue
        cli_value, default_value, saved_value = getattr(current, name), getattr(defaults, name), saved[name]
        if name == "kwargs":
            merged = {**saved_value, **cli_value}
            if merged != cli_value:
                notes.append(f"{name}: {merged} (checkpoint {saved_value}, CLI {cli_value})")
            setattr(current, name, merged)
        elif cli_value == default_value:
            if saved_value != cli_value:
                notes.append(f"{name}: {saved_value!r} (from checkpoint)")
            setattr(current, name, saved_value)
        elif cli_value != saved_value:
            notes.append(f"{name}: {cli_value!r} (CLI overrides checkpoint {saved_value!r})")
    return notes


def config_dict(**sections: Any) -> Dict[str, Dict[str, Any]]:
    """asdict() each section, e.g. config_dict(run=args.run, env=args.env, algo=args.algo)."""
    return {name: dataclasses.asdict(section) for name, section in sections.items()}
