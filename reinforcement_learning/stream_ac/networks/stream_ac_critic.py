import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Critic(nn.Module):
    def __init__(self, env, hidden_size=128):
        super(Critic, self).__init__()
        n_obs = np.array(env.single_observation_space.shape).prod()

        self.fc_layer   = nn.Linear(n_obs, hidden_size)
        self.layernorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.hidden_layer  = nn.Linear(hidden_size, hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear_layer  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc_layer(x)
        x = self.layernorm1(x)
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)      
        x = self.layernorm2(x)
        x = F.leaky_relu(x)
        return self.linear_layer(x)
    
    def get_value(self, x):
        return self(x)
    