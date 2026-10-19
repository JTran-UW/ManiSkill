import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(
        self,
        env,
        num_envs,
        init_scale,
        action_scale=None,
        action_bias=None,
        hidden_dim=512,
        std_min: float = 0.05,
        std_max: float = 0.8,
        use_tanh=False,
        init_trick=False
    ):
        super().__init__()

        n_obs = np.array(env.single_observation_space.shape).prod()
        n_act = np.prod(env.single_action_space.shape)
        self.device = "cuda"
        self.n_envs = num_envs
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, device=self.device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=self.device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=self.device),
            nn.ReLU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim // 4, n_act, device=self.device),
            nn.Tanh(),
        )
        nn.init.normal_(self.fc_mu[0].weight, 0.0, init_scale)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)

        noise_scales = (
            torch.rand(num_envs, 1, device=self.device) * (std_max - std_min) + std_min
        )
        self.register_buffer("noise_scales", noise_scales)

        self.register_buffer("std_min", torch.as_tensor(std_min, device=self.device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=self.device))

    def forward(self, x):
        x_net = self.net(x)
        action = self.fc_mu(x_net)
        return action

    def get_action(self, x, dones=None, deterministic=False):
        # If dones is provided, resample noise for environments that are done
        if dones is not None and dones.sum() > 0:
            # Generate new noise scales for done environments (one per environment)
            new_scales = (
                torch.rand(self.n_envs, 1, device=x.device)
                * (self.std_max - self.std_min)
                + self.std_min
            )

            # Update only the noise scales for environments that are done
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales.copy_(
                torch.where(dones_view, new_scales, self.noise_scales)
            )

        act = self(x)
        if deterministic:
            return act

        noise = torch.randn_like(act) * self.noise_scales
        return act + noise

