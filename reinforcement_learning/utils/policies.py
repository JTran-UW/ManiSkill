"""Load a FastSAC checkpoint as a standalone policy (actor + obs normalizer) for scripts."""
from typing import Callable, Optional, Tuple

import torch

from reinforcement_learning.fastsac.networks.fastsac_actor import Actor
from reinforcement_learning.utils.utils import EmpiricalNormalization


def load_fastsac_actor(checkpoint: str, env) -> Tuple[Actor, Optional[EmpiricalNormalization]]:
    """`env` only needs single_observation_space / single_action_space. Returns (actor, obs normalizer or None), in eval mode."""
    ckpt = torch.load(checkpoint, map_location="cuda", weights_only=False)
    use_tanh = (ckpt.get("config") or {}).get("algo", {}).get("use_tanh", True)
    actor = Actor(env, action_scale=torch.ones(env.single_action_space.shape),
                  action_bias=torch.zeros(env.single_action_space.shape), use_tanh=use_tanh).cuda()
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    normalizer = None
    if "obs_normalizer" in ckpt:
        normalizer = EmpiricalNormalization(shape=env.single_observation_space.shape, device="cuda")
        normalizer.load_state_dict(ckpt["obs_normalizer"])
        normalizer.eval()
    return actor, normalizer


def load_fastsac_policy(checkpoint: str, env, deterministic: bool = True) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns obs -> action (no grad): the policy mean, or a sampled action if not deterministic."""
    actor, normalizer = load_fastsac_actor(checkpoint, env)

    @torch.no_grad()
    def policy(obs: torch.Tensor) -> torch.Tensor:
        x = normalizer(obs, update=False) if normalizer is not None else obs
        action, _, mean = actor.get_action(x)
        return mean if deterministic else action

    return policy


def load_policy(checkpoint: str, env, deterministic: bool = True) -> Callable[[torch.Tensor], torch.Tensor]:
    """Policy from a checkpoint (FastSAC layout)."""
    return load_fastsac_policy(checkpoint, env, deterministic)
