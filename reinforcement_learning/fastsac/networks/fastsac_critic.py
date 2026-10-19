import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class DistributionalQNetwork(nn.Module):
    def __init__(
        self,
        env,
        hidden_dim=768,
        num_atoms=101,
        v_min=-20.0,
        v_max=20.0,
        device="cuda"
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, num_atoms)
        )
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        self.device = device
        self.q_support = torch.linspace(v_min, v_max, num_atoms, device=self.device)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.net(x)
        return x

    def projection(
        self,
        obs,
        actions,
        rewards,
        bootstrap,
        discount
    ):
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]
        # import pdb; pdb.set_trace()

        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * self.q_support
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        lower = torch.floor(b).long()
        upper = torch.ceil(b).long()

        is_integer = upper == lower
        lower_mask = torch.logical_and((lower > 0), is_integer)
        upper_mask = torch.logical_and((lower == 0), is_integer)

        lower = torch.where(lower_mask, lower - 1, lower)
        upper = torch.where(upper_mask, upper + 1, upper)

        next_dist = F.softmax(self(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=self.device)
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )

        # Additional safety check for indices
        lower_indices = (lower + offset).view(-1)
        upper_indices = (upper + offset).view(-1)
        max_index = proj_dist.numel() - 1

        lower_indices = torch.clamp(lower_indices, 0, max_index)
        upper_indices = torch.clamp(upper_indices, 0, max_index)

        proj_dist.view(-1).index_add_(0, lower_indices, (next_dist * (upper.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, upper_indices, (next_dist * (b - lower.float())).view(-1))
        return proj_dist
    
    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate value from logits using support"""
        return torch.sum(probs * self.q_support, dim=-1)

