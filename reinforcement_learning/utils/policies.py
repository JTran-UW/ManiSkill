"""Load a FastSAC or Stream-AC/PPO-pretrain checkpoint as a standalone policy (actor + obs normalizer) for scripts."""
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


def load_stream_ac_policy(checkpoint: str, env, deterministic: bool = True) -> Callable[[torch.Tensor], torch.Tensor]:
    """Policy from a stream_ac.py / stream_ac ppo_pretrain.py checkpoint: obs -> mean or sampled action (no grad)."""
    from reinforcement_learning.stream_ac.networks.stream_ac_actor import Actor as StreamActor

    ckpt = torch.load(checkpoint, map_location="cuda", weights_only=False)
    actor = StreamActor(env, hidden_size=ckpt["config"]["algo"]["hidden_size"]).cuda()
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    normalizer = EmpiricalNormalization(shape=env.single_observation_space.shape, device="cuda")
    normalizer.load_state_dict(ckpt["obs_normalizer"])
    normalizer.eval()

    @torch.no_grad()
    def policy(obs: torch.Tensor) -> torch.Tensor:
        return actor.get_action(normalizer(obs, update=False), deterministic=deterministic)

    return policy


def load_policy(checkpoint: str, env, deterministic: bool = True) -> Callable[[torch.Tensor], torch.Tensor]:
    """Dispatch on checkpoint layout: stream_ac checkpoints store a state-value `critic`, FastSAC ones `qnet`/`qf1`."""
    keys = torch.load(checkpoint, map_location="cpu", weights_only=False).keys()
    if "critic" in keys:
        return load_stream_ac_policy(checkpoint, env, deterministic)
    return load_fastsac_policy(checkpoint, env, deterministic)
