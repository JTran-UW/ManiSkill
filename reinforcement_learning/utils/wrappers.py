import gymnasium as gym
import torch

from mani_skill.envs.sapien_env import BaseEnv


class GrayscaleObservationWrapper(gym.ObservationWrapper):
    """Converts obs["rgb"] (..., ncam * 3) uint8 from FlattenRGBDObservationWrapper to (..., ncam) uint8 luminance.
    Apply right after FlattenRGBDObservationWrapper (before FrameStack)."""

    def __init__(self, env) -> None:
        super().__init__(env)
        self.base_env.update_obs_space(self.observation(self.base_env._init_raw_obs))

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def observation(self, observation: dict):
        return {**observation, "rgb": rgb_to_gray(observation["rgb"])}


def rgb_to_gray(rgb: torch.Tensor) -> torch.Tensor:
    """(..., ncam * 3) uint8 -> (..., ncam) uint8, ITU-R 601 luma per camera."""
    w = torch.tensor([0.299, 0.587, 0.114], device=rgb.device)
    gray = rgb.unflatten(-1, (-1, 3)).float() @ w
    return gray.round_().clamp_(0, 255).to(torch.uint8)


def camera_sensor_config(image_size, eye=None, target=None, fov=None):
    """sensor_configs for gym.make: square images, optionally re-posing/re-FOVing the env's cameras (look_at eye->target)."""
    cfg = dict(width=image_size, height=image_size)
    if (eye is None) != (target is None):
        raise ValueError("camera eye and target must be given together")
    if eye is not None:
        from mani_skill.utils import sapien_utils

        cfg["pose"] = sapien_utils.look_at(eye=list(eye), target=list(target))
    if fov is not None:
        cfg["fov"] = fov
    return cfg


def camera_metadata(eye=None, target=None, fov=None):
    if eye is None and target is None and fov is None:
        return None
    return dict(eye=None if eye is None else tuple(eye), target=None if target is None else tuple(target), fov=fov)
