# Copyright notice
#
# This file contains code adapted from stable-baselines3
# (https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py)
# licensed under the MIT License.
#
# Copyright (c) 2019-2023 Antonin Raffin, Ashley Hill, Anssi Kanervisto,
# Maximilian Ernestus, Rinu Boney, Pavan Goli, and other contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

from __future__ import annotations

from telnetlib import theNULL
import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, NamedTuple

import numpy as np
from sympy import threaded
import torch as th
from tensordict import TensorDict
from gymnasium import spaces

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


__all__ = [
    "BaseBuffer",
    "RolloutBuffer",
    "ReplayBuffer",
    "RolloutBufferSamples",
    "ReplayBufferSamples",
]


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    effective_n_steps: th.Tensor


class ImageReplayBufferSamples(NamedTuple):
    observations: dict
    actions: th.Tensor
    next_observations: dict
    dones: th.Tensor
    rewards: th.Tensor
    effective_n_steps: th.Tensor


def cat_image_samples(a: ImageReplayBufferSamples, b: ImageReplayBufferSamples) -> ImageReplayBufferSamples:
    """Concatenate two sampled batches along the batch dim (dict fields per key)."""
    return ImageReplayBufferSamples(*[
        {k: th.cat([x[k], y[k]]) for k in x} if isinstance(x, dict) else th.cat([x, y])
        for x, y in zip(a, b)
    ])


class AsymmetricReplayBufferSamples(NamedTuple):
    policy_observations: th.Tensor
    critic_observations: th.Tensor
    actions: th.Tensor
    next_policy_observations: th.Tensor
    next_critic_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    effective_n_steps: th.Tensor

def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_obs_shape(
    observation_space: spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_device(device: th.device | str = "auto") -> th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = th.device(device)

    # Cuda not available
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")

    return device


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        """
        :param batch_size: Number of element to sample
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples | RolloutBufferSamples | AsymmetricReplayBufferSamples:
        """
        :param batch_inds:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: th.tensor
    next_observations: th.tensor
    actions: th.tensor
    rewards: th.tensor
    dones: th.tensor
    timeouts: th.tensor

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.observations = th.zeros((self.buffer_size, self.n_envs, *self.obs_shape), device=device)
        self.next_observations = th.zeros((self.buffer_size, self.n_envs, *self.obs_shape), device=device)

        self.actions = th.zeros((self.buffer_size, self.n_envs, self.action_dim), device=device)

        self.rewards = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device=device)
        self.terminations = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device=device)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.truncations = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device=device)
        self.n_steps = n_steps
        self.gamma = gamma

    def add(
        self,
        obs: TensorDict,
        next_obs: TensorDict,
        action: th.tensor,
        reward: th.tensor,
        termination: th.tensor,
        truncation: th.tensor,
        infos: list[dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs

        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.terminations[self.pos] = termination.detach()
        self.truncations[self.pos] = truncation.detach()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    EXPORT_KEYS = ("observations", "next_observations", "actions", "rewards", "terminations", "truncations")

    def export(self) -> dict:
        """Filled rows of the raw storage, time-major (T, n_envs, ...) so n-step windows stay contiguous."""
        n = self.buffer_size if self.full else self.pos
        return {
            **{k: getattr(self, k)[:n].cpu() for k in self.EXPORT_KEYS},
            "n_envs": self.n_envs, "n_steps": self.n_steps, "gamma": self.gamma,
            "pos": self.pos, "full": self.full,
        }

    def load(self, data: dict) -> None:
        """Inverse of export(); the buffer must have the same n_envs and at least as many rows."""
        n = data["observations"].shape[0]
        if data["n_envs"] != self.n_envs or n > self.buffer_size:
            raise ValueError(
                f"buffer has n_envs={self.n_envs}, rows={self.buffer_size}; data has n_envs={data['n_envs']}, rows={n}"
            )
        for k in self.EXPORT_KEYS:
            getattr(self, k)[:n] = data[k].to(self.device)
        if data.get("full") and n == self.buffer_size:
            self.pos, self.full = data["pos"], True
        else:
            self.pos, self.full = n % self.buffer_size, n == self.buffer_size

    @classmethod
    def from_export(cls, data: dict, device, n_steps: int | None = None, gamma: float | None = None) -> "ReplayBuffer":
        """Buffer sized exactly to an export(); n_steps/gamma default to the exported ones."""
        rows, n_envs, obs_dim = data["observations"].shape
        obs_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (data["actions"].shape[-1],), np.float32)
        rb = cls(
            rows * n_envs, obs_space, act_space, device, n_envs=n_envs,
            n_steps=data["n_steps"] if n_steps is None else n_steps,
            gamma=data["gamma"] if gamma is None else gamma,
        )
        rb.load(data)
        return rb

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :return:
        """
        if self.full:
            # FIRST SET self.pos-1 to truncated
            upper_bound = self.buffer_size
            current_pos = self.pos % self.buffer_size
            current_truncations = self.truncations[current_pos - 1, :]
            self.truncations[current_pos - 1, :] = th.logical_not(self.terminations[current_pos - 1, :])
        else:
            # sample only up to self.pos - self.n_steps + 1
            upper_bound = self.pos - self.n_steps + 1
        
        batch_inds = th.randint(0, upper_bound, (batch_size, ), device=self.device) % self.buffer_size
        samples = self._get_samples(batch_inds)
        if self.full:
            self.truncations[self.pos - 1, :] = current_truncations
        return samples

    def _get_samples(self, batch_inds: th.tensor) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = th.randint(0, high=self.n_envs, size=(len(batch_inds),), device=self.device)
        offsets = th.arange(self.n_steps, device=self.device).view(-1, 1)
        seq_inds = (batch_inds + offsets) % self.buffer_size

        n_step_rewards = self.rewards[seq_inds, env_indices]
        n_step_terminations = self.terminations[seq_inds, env_indices].int().bool()
        n_step_truncations = self.truncations[seq_inds, env_indices].int().bool()
        n_step_dones = n_step_terminations | n_step_truncations

        n_step_dones_shifted = th.concat([th.zeros_like(n_step_dones[0].unsqueeze(0)), n_step_dones[:-1, :]])
        done_mask = th.cumprod(1 - n_step_dones_shifted.int(), dim=0)
        effective_n_steps = done_mask.sum(dim=0)
        masked_rewards = done_mask * n_step_rewards
        discounts = th.pow(self.gamma, th.arange(self.n_steps, device=self.device))

        discounted_rewards = masked_rewards * discounts.view(-1, 1)
        n_step_rewards = discounted_rewards.sum(dim=0)

        first_done = th.argmax(n_step_dones.int(), dim=0)
        no_done = n_step_dones.sum(dim=0) == 0
        first_done = th.where(no_done, self.n_steps - 1, first_done)

        data = (
            self.observations[batch_inds, env_indices],
            self.actions[batch_inds, env_indices],
            self.next_observations[(batch_inds + first_done) % self.buffer_size, env_indices],
            self.terminations[(batch_inds + first_done) % self.buffer_size, env_indices],
            n_step_rewards,
            effective_n_steps
        )
        return ReplayBufferSamples(*data)


def sample_mixed(online: ReplayBuffer, expert: ReplayBuffer | None, batch_size: int, expert_ratio: float) -> ReplayBufferSamples:
    """Exactly round(expert_ratio * batch_size) expert transitions, the rest online."""
    n_expert = int(round(batch_size * expert_ratio)) if expert is not None else 0
    if n_expert == 0:
        return online.sample(batch_size)
    if n_expert == batch_size:
        return expert.sample(batch_size)
    a, b = online.sample(batch_size - n_expert), expert.sample(n_expert)
    return ReplayBufferSamples(*(th.cat([x, y]) for x, y in zip(a, b)))


class ImageReplayBuffer:
    def __init__(
        self, pixels_obs_space, state_obs_dim, act_dim, size, horizon, num_envs, n_frame_stack, n_steps, gamma, device="cuda"
    ):
        assert (
            size % horizon == 0
        ), f"{self.__class__.__name__} needs size divisble by horizon"

        num_episodes = size // horizon
        assert (
            num_episodes % num_envs == 0
        ), f"{self.__class__.__name__} needs num_episodes divisible by num_envs"

        ## init buffers as numpy arrays
        self.pixel_obs_buf = th.zeros(
            num_episodes, horizon + n_frame_stack, *pixels_obs_space[1:],
            dtype=th.uint8,
            device=device
        )
        to = list(range(len(pixels_obs_space) + 1))
        to[0] = 1
        to[1] = 0
        self.transpose_order = tuple(to)

        self.state_obs_buf = th.zeros(
            num_episodes, horizon, state_obs_dim, dtype=th.float32, device=device
        )
        self.state_next_obs_buf = th.zeros(
            num_episodes, horizon, state_obs_dim, dtype=th.float32, device=device
        )
        self.acts_buf = th.zeros(num_episodes, horizon, act_dim, dtype=th.float32, device=device)
        self.rews_buf = th.zeros(num_episodes, horizon, dtype=th.float32, device=device)
        self.done_buf = th.zeros(num_episodes, horizon, dtype=th.float32, device=device)

        self.idx_to_coord = th.zeros(size, 2, dtype=th.long, device=device)

        self.max_size = size
        self.num_episodes = num_episodes
        self.horizon = horizon
        self.num_envs = num_envs
        self.frame_stack = n_frame_stack

        self.current_size = 0
        self.batch_start_episode = 0
        self.step_num = 0

        self.n_steps = n_steps
        self.gamma = gamma
        self.device = device

    def add(
        self,
        obs,
        next_obs,
        act,
        rew,
        term,
    ):
        bs, be = self.batch_start_episode, self.batch_start_episode + self.num_envs
        sn = self.step_num
        fs = self.frame_stack

        pixel_obs = obs["rgb"]
        pixel_next_obs = next_obs["rgb"]
        state_obs = obs["state"][:, -1]
        state_next_obs = next_obs["state"][:, -1]

        self.pixel_obs_buf[bs:be, sn : sn + fs] = pixel_obs
        self.pixel_obs_buf[bs:be, sn + 1 : sn + fs + 1] = pixel_next_obs

        self.state_obs_buf[bs:be, sn] = state_obs
        self.state_next_obs_buf[bs:be, sn] = state_next_obs
        self.acts_buf[bs:be, sn] = act
        self.rews_buf[bs:be, sn] = rew
        self.done_buf[bs:be, sn] = term

        if self.current_size < self.max_size:
            self.idx_to_coord[self.current_size : self.current_size + self.num_envs] = (
                th.stack([th.arange(bs, be, device=self.device), th.full((be - bs,), sn, device=self.device)], dim=-1)
            )

        self.step_num += 1
        self.current_size = min(self.current_size + self.num_envs, self.max_size)

        if self.step_num == self.horizon:
            self.step_num = 0
            self.batch_start_episode += self.num_envs
        if self.batch_start_episode == self.num_episodes:
            self.batch_start_episode = 0

    def sample(self, batch_size):
        idxs = th.randint(0, self.current_size, (batch_size,), device=self.device)
        ep_nums, step_nums = self.idx_to_coord[idxs].unbind(-1)
        ep = ep_nums[:, None]

        # steps after t that hold this episode's data: up to the horizon, or only up to step_num for the rows being filled
        limit = th.full_like(step_nums, self.horizon)
        bs, be = self.batch_start_episode, self.batch_start_episode + self.num_envs
        filling = (ep_nums >= bs) & (ep_nums < be) & (step_nums < self.step_num)
        limit = th.where(filling, th.full_like(step_nums, self.step_num), limit)

        i = th.arange(self.n_steps, device=self.device)
        steps = step_nums[:, None] + i
        valid = (steps < limit[:, None]).float()
        steps_c = steps.clamp(max=self.horizon - 1)

        rews = self.rews_buf[ep, steps_c] * valid
        terms = self.done_buf[ep, steps_c] * valid
        # a step counts if it is valid and no termination happened earlier in the window
        alive = th.cumprod(th.cat([th.ones_like(terms[:, :1]), 1 - terms[:, :-1]], dim=1), dim=1) * valid
        effective_n_steps = alive.sum(1).long()

        returns = (rews * alive * self.gamma ** i).sum(1)
        dones = (terms * alive).amax(1)
        last = step_nums + effective_n_steps - 1

        offs = th.arange(self.frame_stack, device=self.device)
        obs = {
            "rgb": self.pixel_obs_buf[ep, step_nums[:, None] + offs],
            "state": self.state_obs_buf[ep_nums, step_nums],
        }
        next_obs = {
            "rgb": self.pixel_obs_buf[ep, (last + 1)[:, None] + offs],
            "state": self.state_next_obs_buf[ep_nums, last],
        }

        return ImageReplayBufferSamples(
            observations=obs,
            actions=self.acts_buf[ep_nums, step_nums],
            next_observations=next_obs,
            dones=dones,
            rewards=returns,
            effective_n_steps=effective_n_steps,
        )

    def __len__(self):
        return self.current_size

    def state_dict(self, metadata=None):
        return {
            "pixel_obs_buf": self.pixel_obs_buf.cpu(),
            "state_obs_buf": self.state_obs_buf.cpu(),
            "state_next_obs_buf": self.state_next_obs_buf.cpu(),
            "acts_buf": self.acts_buf.cpu(),
            "rews_buf": self.rews_buf.cpu(),
            "done_buf": self.done_buf.cpu(),
            "idx_to_coord": self.idx_to_coord.cpu(),
            "max_size": self.max_size,
            "horizon": self.horizon,
            "num_envs": self.num_envs,
            "frame_stack": self.frame_stack,
            "current_size": self.current_size,
            "batch_start_episode": self.batch_start_episode,
            "step_num": self.step_num,
            "metadata": metadata or {},
        }

    @classmethod
    def from_state_dict(cls, d, n_steps, gamma, device="cuda"):
        frame_shape = tuple(d["pixel_obs_buf"].shape[2:])
        buf = cls(
            pixels_obs_space=(d["frame_stack"], *frame_shape),
            state_obs_dim=d["state_obs_buf"].shape[-1],
            act_dim=d["acts_buf"].shape[-1],
            size=d["max_size"],
            horizon=d["horizon"],
            num_envs=d["num_envs"],
            n_frame_stack=d["frame_stack"],
            n_steps=n_steps,
            gamma=gamma,
            device=device,
        )
        for k in ("pixel_obs_buf", "state_obs_buf", "state_next_obs_buf", "acts_buf", "rews_buf", "done_buf", "idx_to_coord"):
            getattr(buf, k).copy_(d[k])
        buf.current_size = d["current_size"]
        buf.batch_start_episode = d["batch_start_episode"]
        buf.step_num = d["step_num"]
        buf.metadata = d.get("metadata", {})
        return buf


class AsymmetricReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    policy_observations: th.tensor
    critic_observations: th.tensor
    next_policy_observations: th.tensor
    next_critic_observations: th.tensor
    actions: th.tensor
    rewards: th.tensor
    terminations: th.tensor
    truncations: th.tensor

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.policy_observations = th.zeros((self.buffer_size, self.n_envs, np.array(self.obs_shape["policy"]).prod()), device=device)
        self.critic_observations = th.zeros((self.buffer_size, self.n_envs, np.array(self.obs_shape["critic"]).prod()), device=device)

        # When optimizing memory, `observations` contains also the next observation
        self.next_policy_observations = th.zeros((self.buffer_size, self.n_envs, np.array(self.obs_shape["policy"]).prod()), device=device)
        self.next_critic_observations = th.zeros((self.buffer_size, self.n_envs, np.array(self.obs_shape["critic"]).prod()), device=device)

        self.actions = th.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=th.float32, device=device
        )

        self.rewards = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device=device)
        self.terminations = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device=device)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.truncations = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device=device)
        self.n_steps = n_steps
        self.gamma = gamma

    def add(
        self,
        obs: TensorDict,
        next_obs: TensorDict,
        action: th.tensor,
        reward: th.tensor,
        termination: th.tensor,
        truncation: th.tensor,
        infos: list[dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))
            print("[WARNING] Discrete obs not handled well in AsymmetricReplayBuffer")

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.policy_observations[self.pos] = obs["policy"].detach()
        self.critic_observations[self.pos] = obs["critic"].detach()
        self.next_policy_observations[self.pos] = next_obs["policy"].detach()
        self.next_critic_observations[self.pos] = next_obs["critic"].detach()

        self.actions[self.pos] = action.detach()
        self.rewards[self.pos] = reward.detach()
        self.terminations[self.pos] = termination.detach()
        self.truncations[self.pos] = truncation.detach()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> AsymmetricReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :return:
        """
        if self.full:
            # FIRST SET self.pos-1 to truncated
            upper_bound = self.buffer_size
            current_pos = self.pos % self.buffer_size
            current_truncations = self.truncations[current_pos - 1, :]
            self.truncations[current_pos - 1, :] = th.logical_not(self.terminations[current_pos - 1, :])
        else:
            # sample only up to self.pos - self.n_steps + 1
            upper_bound = self.pos - self.n_steps + 1
        
        batch_inds = th.randint(0, upper_bound, (batch_size, ), device=self.device) % self.buffer_size
        samples = self._get_samples(batch_inds)
        if self.full:
            self.truncations[self.pos - 1, :] = current_truncations
        return samples

    def _get_samples(self, batch_inds: th.tensor) -> AsymmetricReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = th.randint(0, high=self.n_envs, size=(len(batch_inds),), device=self.device)
        offsets = th.arange(self.n_steps, device=self.device).view(-1, 1)
        seq_inds = (batch_inds + offsets) % self.buffer_size

        n_step_rewards = self.rewards[seq_inds, env_indices]
        n_step_terminations = self.terminations[seq_inds, env_indices].int().bool()
        n_step_truncations = self.truncations[seq_inds, env_indices].int().bool()
        n_step_dones = n_step_terminations | n_step_truncations

        n_step_dones_shifted = th.concat([th.zeros_like(n_step_dones[0].unsqueeze(0)), n_step_dones[:-1, :]])
        done_mask = th.cumprod(1 - n_step_dones_shifted.int(), dim=0)
        effective_n_steps = done_mask.sum(dim=0)
        masked_rewards = done_mask * n_step_rewards
        discounts = th.pow(self.gamma, th.arange(self.n_steps, device=self.device))

        discounted_rewards = masked_rewards * discounts.view(-1, 1)
        n_step_rewards = discounted_rewards.sum(dim=0)

        first_done = th.argmax(n_step_dones.int(), dim=0)
        no_done = n_step_dones.sum(dim=0) == 0
        first_done = th.where(no_done, self.n_steps - 1, first_done)

        data = (
            self.policy_observations[batch_inds, env_indices],
            self.critic_observations[batch_inds, env_indices],
            self.actions[batch_inds, env_indices],
            self.next_policy_observations[(batch_inds + first_done) % self.buffer_size, env_indices],
            self.next_critic_observations[(batch_inds + first_done) % self.buffer_size, env_indices],
            self.terminations[batch_inds + first_done, env_indices],
            n_step_rewards,
            effective_n_steps
        )
        return AsymmetricReplayBufferSamples(*data)


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: int | None = None) -> Generator[RolloutBufferSamples]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
