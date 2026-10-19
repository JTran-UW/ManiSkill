import torch
import torch.nn as nn
import numpy as np

from reinforcement_learning.flashsac.networks.flashsac_utils import (
    UnitLinear,
    UnitConv2d,
    UnitBatchNorm,
    UnitRMSNorm,
    FlashSACBlock
)

LOG_STD_MAX = 2
LOG_STD_MIN = -10


class FlashSACActor(nn.Module):
    def __init__(
        self,
        env,
        input_dim=None,
        action_scale=None,
        action_bias=None,
        hidden_dim=128,
        num_blocks=2,
        use_tanh=False,
    ):
        super().__init__()
        if not input_dim:
            input_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = np.prod(env.single_action_space.shape)

        self.norm1 = UnitBatchNorm(input_dim)
        self.linear1 = UnitLinear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([FlashSACBlock(hidden_dim) for _ in range(num_blocks)])
        self.norm2 = UnitRMSNorm(hidden_dim)

        self.fc_mu = UnitLinear(hidden_dim, action_dim, bias=True)
        self.fc_logstd = UnitLinear(hidden_dim, action_dim, bias=True)
        self.use_tanh = use_tanh

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

    def forward(self, x, training: bool):
        x = self.linear1(self.norm1(x, training=training))
        for block in self.blocks:
            x = block(x, training=training) + x
        x = self.norm2(x)
        mean = self.fc_mu(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, training: bool):
        mean, log_std = self(x, training=training)
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
    
    @torch.no_grad()
    def normalize_parameters(self):
        self.norm1.normalize_parameters()
        self.linear1.normalize_parameters()
        for block in self.blocks:
            block.normalize_parameters()
        self.norm2.normalize_parameters()
        self.fc_mu.normalize_parameters()
        self.fc_logstd.normalize_parameters()

