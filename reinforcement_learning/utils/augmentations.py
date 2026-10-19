import torch


def random_shift(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """DrQ random shift on (N, k, H, W, C) images: replicate-pad by `pad` px and randomly crop back to (H, W).
    One integer shift per sample, shared by all k frames; works directly on uint8."""
    if pad <= 0:
        return x
    N, k, H, W, C = x.shape
    shift = torch.randint(-pad, pad + 1, (N, 2), device=x.device)
    rows = (torch.arange(H, device=x.device) + shift[:, :1]).clamp_(0, H - 1)
    cols = (torch.arange(W, device=x.device) + shift[:, 1:]).clamp_(0, W - 1)
    n = torch.arange(N, device=x.device)[:, None, None]
    return x[n, :, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2, 4)
