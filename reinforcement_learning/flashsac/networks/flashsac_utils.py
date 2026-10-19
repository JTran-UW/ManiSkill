import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class UnitLinear(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        nn.init.orthogonal_(self.linear.weight, gain=1)
    
    def forward(self, x):
        return self.linear(x)

    def normalize_parameters(self):
        self.linear.weight.copy_(F.normalize(self.linear.weight, dim=-1, eps=1e-8))


class UnitConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        nn.init.orthogonal_(self.conv.weight, gain=1)
    
    def forward(self, x):
        return self.conv(x)

    def normalize_parameters(self):
        self.conv.weight.copy_(F.normalize(self.conv.weight.flatten(1), dim=-1).view_as(self.conv.weight))


class UnitBatchNorm(nn.Module):
    def __init__(self, input_dim, momentum=0.01, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x, training: bool):
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=training,
            momentum=self.momentum,
            eps=self.eps,
        )

    def normalize_parameters(self):
        scale, bias = self.weight.data, self.bias.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)
        self.bias.data.copy_(bias * norm_factor)


class UnitRMSNorm(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.norm = nn.RMSNorm(input_dim)
    
    def forward(self, x):
        return self.norm(x)
    
    def normalize_parameters(self):
        scale = self.norm.weight
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.norm.weight.copy_(scale * norm_factor)


class FlashSACBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden_dim = 4 * input_dim
        self.linear1 = UnitLinear(input_dim, hidden_dim)
        self.norm1 = UnitBatchNorm(hidden_dim)
        self.linear2 = UnitLinear(hidden_dim, input_dim)
        self.norm2 = UnitBatchNorm(input_dim)
    
    def forward(self, x, training: bool):
        x = F.relu(self.norm1(self.linear1(x), training=training))
        x = F.relu(self.norm2(self.linear2(x), training=training))
        return x
    
    def normalize_parameters(self):
        self.linear1.normalize_parameters()
        self.linear2.normalize_parameters()
        self.norm1.normalize_parameters()
        self.norm2.normalize_parameters()


class FlashSACEncoder(nn.Module):
    def __init__(
        self,
        env,
        image_embed_dim=64,
        state_embed_dim=64,
        channels_last=False,
    ):
        super().__init__()
        self.channels_last = channels_last
        k, image_size, _, C = env.single_observation_space["rgb"].shape
        in_channels = k * C
        assert image_size % 32 == 0, image_size
        self.first_pool = image_size // 32
        self.state_input_dim = env.single_observation_space["state"].shape[1]
        self.conv_layers = nn.ModuleList([
            UnitConv2d(in_channels, 16, 3, padding=1, bias=True),
            UnitConv2d(16, 32, 3, padding=1, bias=True),
            UnitConv2d(32, 64, 3, padding=1, bias=True),
            UnitConv2d(64, 64, 3, padding=1, bias=True),
            UnitConv2d(64, 64, 1, padding=0, bias=True)
        ])
        self.fc = UnitLinear(64 * 4 * 4, image_embed_dim)
        self.state_proj = UnitLinear(self.state_input_dim, state_embed_dim)

    def forward(self, x):
        x_rgb = x["rgb"]
        B, k, H, W, C = x_rgb.shape
        if self.channels_last:
            x_rgb = x_rgb.permute(0, 2, 3, 1, 4).reshape(B, H, W, k * C).permute(0, 3, 1, 2)
        else:
            x_rgb = x_rgb.permute(0, 1, 4, 2, 3)
            x_rgb = x_rgb.reshape(B, k * C, H, W)
        x_rgb = x_rgb.float().div_(255.0)
        x_state = x["state"]
        if x_state.dim() == 3:
            x_state = x_state[:, -1]
        x_state = x_state.reshape(B, k * self.state_input_dim)

        for i, layer in enumerate(self.conv_layers[:-1]):
            pool = self.first_pool if i == 0 else 2
            x_rgb = F.max_pool2d(F.relu(layer(x_rgb)), pool, pool)
        x_rgb = F.relu(self.conv_layers[-1](x_rgb))

        image_embed = self.fc(x_rgb.flatten(1))
        state_embed = self.state_proj(x_state.flatten(1))

        return torch.concat([image_embed, state_embed], dim=-1)

    @torch.no_grad
    def normalize_parameters(self):
        for layer in self.conv_layers:
            layer.normalize_parameters()
        self.fc.normalize_parameters()
        self.state_proj.normalize_parameters()

