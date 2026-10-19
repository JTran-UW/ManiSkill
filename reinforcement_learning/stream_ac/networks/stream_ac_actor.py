import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class Actor(nn.Module):
    def __init__(self, env, hidden_size=128):
        super(Actor, self).__init__()
        n_obs = np.array(env.single_observation_space.shape).prod()
        n_actions = np.array(env.single_action_space.shape).prod()

        self.fc_layer   = nn.Linear(n_obs, hidden_size)
        self.layernorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear_mu = nn.Linear(hidden_size, n_actions)
        self.linear_std = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = self.fc_layer(x)
        x = self.layernorm1(x)
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = self.layernorm2(x)
        x = F.leaky_relu(x)
        mu = self.linear_mu(x)
        pre_std = self.linear_std(x)
        std = F.softplus(pre_std)
        return mu, std
    
    def get_action(self, x, deterministic=False):
        mu, std = self(x)
        if deterministic:
            return mu
        return Normal(mu, std).sample()

    def get_action_and_log_prob(self, x, action=None):
        mu, std = self(x)
        dist = Normal(mu, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1)
