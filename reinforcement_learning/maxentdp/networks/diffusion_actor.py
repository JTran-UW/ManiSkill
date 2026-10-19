"""PyTorch port of the MaxEntDP diffusion actor (github.com/diffusionyes/MaxEntDP, jaxrl5 max_entropy_learner.py).

Actions live in [-1, 1]. The network predicts the DDPM noise eps(s, a_t, log_snr_t) on the continuous cosine VP
schedule of DPM-Solver (t in [1e-3, 0.9946]).
"""
import math

import torch
import torch.nn as nn

COSINE_S = 0.008
T_MAX = 0.9946
_LOG_ALPHA_0 = math.log(math.cos(COSINE_S / (1.0 + COSINE_S) * math.pi / 2.0))


def alpha_hat(t: torch.Tensor) -> torch.Tensor:
    """alpha_bar(t) = marginal_alpha(t) ** 2 of the cosine VP schedule."""
    log_alpha = torch.log(torch.cos((t + COSINE_S) / (1.0 + COSINE_S) * math.pi / 2.0)) - _LOG_ALPHA_0
    return torch.exp(2.0 * log_alpha)


def truncated_normal(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Standard normal truncated to [lo, hi] elementwise (inverse CDF in float64, left-tail flip for stability)."""
    lo64, hi64 = lo.double(), hi.double()
    flip = lo64 > 0
    a = torch.where(flip, -hi64, lo64)
    b = torch.where(flip, -lo64, hi64)
    ca, cb = torch.special.ndtr(a), torch.special.ndtr(b)
    u = ca + (cb - ca) * torch.rand_like(ca)
    x = torch.special.ndtri(u.clamp(min=1e-300))
    x = torch.where(flip, -x, x)
    x = torch.where(torch.isfinite(x), x, torch.randn_like(x))
    return torch.minimum(torch.maximum(x, lo64), hi64).to(lo.dtype)


class FourierFeatures(nn.Module):
    def __init__(self, output_size: int, input_dim: int = 1):
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(output_size // 2, input_dim) * 0.2)

    def forward(self, x):
        f = 2 * math.pi * x @ self.kernel.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


def _mlp(dims, layer_norm: bool):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i + 2 < len(dims):
            if layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.Mish())
    return nn.Sequential(*layers)


class DiffusionActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256, 256), time_dim: int = 64, cond_dims=(128, 128),
                 layer_norm: bool = False, T: int = 20, clip_sampler: bool = True):
        super().__init__()
        self.act_dim = act_dim
        self.T = T
        self.clip_sampler = clip_sampler
        self.time_ff = FourierFeatures(time_dim)
        self.cond = _mlp([time_dim, *cond_dims], layer_norm=False)
        self.net = _mlp([act_dim + obs_dim + cond_dims[-1], *hidden_dims, act_dim], layer_norm=layer_norm)
        ahp = alpha_hat(torch.linspace(1e-3, T_MAX, T + 1))
        ah = ahp[1:]
        alphas = ahp[1:] / ahp[:-1]
        betas = 1 - alphas
        self.register_buffer("ah", ah)
        self.register_buffer("log_snr", ah.log() - (1 - ah).log())
        self.register_buffer("x0_coef", 1 / ah.sqrt())
        self.register_buffer("eps_coef", (1 - ah).sqrt())
        self.register_buffer("mean_coef1", ahp[:-1].sqrt() * betas / (1 - ah))
        self.register_buffer("mean_coef2", alphas.sqrt() * (1 - ahp[:-1]) / (1 - ah))
        self.register_buffer("sigma", betas.sqrt())

    def forward(self, obs, noisy_actions, log_snr):
        cond = self.cond(self.time_ff(log_snr))
        return self.net(torch.cat([noisy_actions, obs, cond], dim=-1))

    def _split_first(self):
        """First layer of self.net split by input block: W [a | s | cond], so the s and cond terms can be broadcast."""
        lin = self.net[0]
        A, C = self.act_dim, self.cond[-1].out_features
        return lin.weight[:, :A], lin.weight[:, A:-C], lin.weight[:, -C:], lin.bias

    def _from_parts(self, h):
        return self.net[1:](h)

    def sample(self, obs: torch.Tensor, stochastic: bool = True) -> torch.Tensor:
        """Ancestral DDPM sampling (jaxrl5 ddpm_sampler); obs/time terms of the first layer computed once.
        stochastic=False: same start noise schedule but no per-step noise (posterior-mean denoising path)."""
        Wa, Ws, Wc, b = self._split_first()
        hs = obs @ Ws.T + b
        hc = self.cond(self.time_ff(self.log_snr[:, None])) @ Wc.T  # (T, H)
        x = torch.randn(obs.shape[0], self.act_dim, device=obs.device, dtype=obs.dtype)
        for i in range(self.T - 1, -1, -1):
            eps = self._from_parts(x @ Wa.T + hs + hc[i])
            x0 = (x - self.eps_coef[i] * eps) * self.x0_coef[i]
            if self.clip_sampler:
                x0 = x0.clamp(-1, 1)
            x = self.mean_coef1[i] * x0 + self.mean_coef2[i] * x
            if i > 0 and stochastic:
                x = x + self.sigma[i] * torch.randn_like(x)
        return x.clamp(-1, 1)

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor, num_samples: int = 50, T: int = 20) -> torch.Tensor:
        """Denoising-error estimate of log pi(a|s) (jaxrl5 calc_log_prob)."""
        B, A = actions.shape
        ahp = alpha_hat(torch.linspace(1e-3, T_MAX, T + 1, device=actions.device))
        ah, ah_prev = ahp[1:], ahp[:-1]
        noise = torch.randn(B, num_samples, T, A, device=actions.device)
        a1 = ah.sqrt().view(1, 1, T, 1)
        a2 = (1 - ah).sqrt().view(1, 1, T, 1)
        noisy = a1 * actions[:, None, None] + a2 * noise
        Wa, Ws, Wc, b = self._split_first()
        hc = self.cond(self.time_ff((ah.log() - (1 - ah).log())[:, None])) @ Wc.T  # (T, H)
        h = noisy @ Wa.T + (obs @ Ws.T + b)[:, None, None] + hc[None, None]
        eps = self._from_parts(h)
        if self.clip_sampler:
            x0 = ((noisy - a2 * eps) / a1).clamp(-1, 1)
            eps = (noisy - a1 * x0) / a2
        weight = (ah_prev - ah) / (2 * ah * (1 - ah))
        err = ((eps - noise) ** 2).sum(-1).mean(1)
        return ((ah * A - err) * weight).sum(-1) - 0.5 * A * math.log(2 * math.pi * math.e)

    def noise_actions(self, actions: torch.Tensor):
        """Forward process at t ~ U(1e-3, T_MAX): (noisy actions, alpha_1, alpha_2, log_snr)."""
        B = actions.shape[0]
        t = torch.rand(B, device=actions.device) * (T_MAX - 1e-3) + 1e-3
        ah = alpha_hat(t)[:, None]
        a1, a2 = ah.sqrt(), (1 - ah).sqrt()
        noisy = a1 * actions + a2 * torch.randn_like(actions)
        return noisy, a1, a2, ah.log() - (1 - ah).log()


def qne_target(noisy: torch.Tensor, a1: torch.Tensor, a2: torch.Tensor, obs: torch.Tensor, q_fn, K: int, temp, beta: float = 0.0):
    """Q-weighted Noise Estimation: eps target = -E_{w ~ softmax(Q/temp)}[n] over K posterior clean actions
    a0 = noisy / a1 + (a2 / a1) n with n truncated so a0 stays in [-1, 1]. beta > 0: per-state scale-free weights
    softmax((Q - mean_K Q) / (beta * std_K Q)) instead of a fixed temperature."""
    B, A = noisy.shape
    std = (a2 / a1)[:, :, None]
    x = noisy[:, None].expand(B, K, A)
    center = x / a1[:, :, None]
    noise = truncated_normal((-1 - center) / std, (1 - center) / std)
    clean = center + std * noise
    q = q_fn(obs[:, None].expand(B, K, obs.shape[-1]).reshape(B * K, -1), clean.reshape(B * K, A)).view(B, K)
    if beta > 0:
        w = torch.softmax((q - q.mean(1, keepdim=True)) / (beta * q.std(1, keepdim=True) + 1e-8), dim=1)
    else:
        w = torch.softmax(q / temp, dim=1)
    return -(w[:, :, None] * noise).sum(1), q, w
