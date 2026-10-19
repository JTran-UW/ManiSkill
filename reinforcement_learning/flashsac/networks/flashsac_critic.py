import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from reinforcement_learning.flashsac.networks.flashsac_utils import (
    UnitLinear,
    UnitBatchNorm,
    UnitRMSNorm,
    FlashSACBlock
)

# ALGO LOGIC: initialize agent here:
class FlashSACQNetwork(nn.Module):
    def __init__(
        self,
        env,
        input_dim=None,
        hidden_dim=256,
        num_atoms=101,
        v_min=-20.0,
        v_max=20.0,
        num_blocks=2,
        device="cuda"
    ):
        super().__init__()
        if not input_dim:
            input_dim = np.array(env.single_observation_space.shape).prod()
        input_dim += np.prod(env.single_action_space.shape)
        
        self.norm1 = UnitBatchNorm(input_dim)
        self.linear1 = UnitLinear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([FlashSACBlock(hidden_dim) for _ in range(num_blocks)])
        self.norm2 = UnitRMSNorm(hidden_dim)
        self.linear2 = UnitLinear(hidden_dim, num_atoms, bias=True)

        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        self.device = device
        self.q_support = torch.linspace(v_min, v_max, num_atoms, device=self.device)

    def forward(self, x, a, training: bool):
        x = torch.cat([x, a], 1)
        x = self.linear1(self.norm1(x, training=training))
        for block in self.blocks:
            x = block(x, training=training) + x
        x = self.linear2(self.norm2(x))
        return x

    def projection(
        self,
        next_dist,
        actions,
        rewards,
        bootstrap,
        discount,
        training: bool,
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

    @torch.no_grad()
    def normalize_parameters(self):
        self.norm1.normalize_parameters()
        self.linear1.normalize_parameters()
        for block in self.blocks:
            block.normalize_parameters()
        self.norm2.normalize_parameters()
        self.linear2.normalize_parameters()
