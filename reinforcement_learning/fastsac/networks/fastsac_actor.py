import torch
import torch.nn as nn
import numpy as np

import math

LOG_STD_MAX = 0
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, env, action_scale=None, action_bias=None, hidden_dim=512, use_tanh=False, init_trick=False):
        super().__init__()
        input_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = np.prod(env.single_action_space.shape)
        self.device = "cuda"
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, device=self.device),
            nn.LayerNorm(hidden_dim, device=self.device),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=self.device),
            nn.LayerNorm(hidden_dim // 2, device=self.device),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=self.device),
            nn.LayerNorm(hidden_dim // 4, device=self.device),
            nn.SiLU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim // 4, action_dim, device=self.device),
        )
        self.fc_logstd = nn.Linear(hidden_dim // 4, action_dim, device=self.device)
        nn.init.constant_(self.fc_mu[0].weight, 0.0)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)
        nn.init.constant_(self.fc_logstd.weight, 0.0)
        nn.init.constant_(self.fc_logstd.bias, 0.0)
        self.use_tanh = use_tanh

        # Initial exploration tricks
        if init_trick:
            nn.init.orthogonal_(self.fc_mu.weight, gain=0.01)
            nn.init.zeros_(self.fc_mu.bias)

            std_0 = 0.15
            nn.init.zeros_(self.fc_logstd.weight)
            nn.init.constant_(self.fc_logstd.bias, math.log(std_0))

        # action rescaling
        if action_scale != None and action_bias != None:
            action_scale = action_scale
            action_bias = action_bias
        elif env.single_action_space.high[0] == torch.inf:
            action_scale = 1.0
            action_bias = 0.0
        else:
            action_scale = (env.single_action_space.high - env.single_action_space.low) / 2.0
            action_bias = (env.single_action_space.high + env.single_action_space.low) / 2.0

        self.register_buffer(
            "action_scale",
            torch.tensor(
                action_scale,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                action_bias,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = self.net(x)
        mean = self.fc_mu(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))

        if self.use_tanh:
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)
            # Enforcing Action Bound
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
        else:
            action = x_t
            log_prob = normal.log_prob(x_t)
            log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, mean

