"""CPU sanity check: QNE on a fixed bimodal Q(a) in 1-D/2-D -> diffusion samples ~ exp(Q/alpha); log_prob tracks the true density."""
import math, torch
from reinforcement_learning.maxentdp.networks.diffusion_actor import DiffusionActor, qne_target, truncated_normal

torch.manual_seed(0)
# truncated normal check
lo = torch.tensor([-3.0, 2.0, -9.0, 5.0]).repeat(20000, 1); hi = torch.tensor([3.0, 4.0, -8.0, 9.0]).repeat(20000, 1)
x = truncated_normal(lo, hi)
assert ((x >= lo) & (x <= hi)).all() and torch.isfinite(x).all()
print("truncnorm means", x.mean(0).tolist())

A, alpha = 2, 0.1
centers = torch.tensor([[0.6, 0.0], [-0.6, 0.0]])
def Q(obs, a):  # two modes, std 0.1 each at alpha=0.1
    d = torch.cdist(a, centers) ** 2
    return -(d.min(1).values) * 5.0
actor = DiffusionActor(obs_dim=1, act_dim=A, hidden_dims=(256, 256), T=20)
opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
obs = torch.zeros(1024, 1)
for it in range(3000):
    a_rb = torch.rand(1024, A) * 2 - 1  # uniform proposal like random warmup data
    noisy, a1, a2, log_snr = actor.noise_actions(a_rb)
    with torch.no_grad():
        tgt, q, w = qne_target(noisy, a1, a2, obs, Q, 128, alpha)
    loss = ((actor(obs, noisy, log_snr) - tgt) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    s = actor.sample(torch.zeros(4000, 1))
right = (s[:, 0] > 0).float().mean().item()
print(f"loss {loss.item():.3f}  frac right mode {right:.2f}  mean|x| {s[:,0].abs().mean():.3f} (target 0.6)  std y {s[:,1].std():.3f} (target {math.sqrt(alpha/10):.3f})")
# true log density of the mixture (each mode isotropic N(c, alpha/(2*5)) truncated negligibly)
var = alpha / 10
pts = torch.tensor([[0.6, 0.0], [0.6, 0.15], [0.0, 0.0], [-0.6, 0.05]])
true = torch.logsumexp(torch.stack([-((pts - c) ** 2).sum(1) / (2 * var) for c in centers]), 0) - math.log(2) - A / 2 * math.log(2 * math.pi * var)
with torch.no_grad():
    est = actor.log_prob(torch.zeros(len(pts), 1), pts, num_samples=256, T=20)
print("log_prob est ", [round(v, 2) for v in est.tolist()])
print("log_prob true", [round(v, 2) for v in true.tolist()])
assert 0.3 < right < 0.7, "mode collapse"
