import math

import torch
import torch.nn.functional as F


def mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, y)


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    value = mse(x, y).item()
    if value <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / value)


def max_abs_error_pixels(x: torch.Tensor, y: torch.Tensor) -> float:
    return torch.max(torch.abs(torch.round(x * 255.0) - torch.round(y * 255.0))).item()


def bits_per_pixel(bits: torch.Tensor, x: torch.Tensor) -> float:
    pixels = x.shape[0] * x.shape[2] * x.shape[3]
    return bits.sum().item() / float(pixels)


def _gaussian_window(size: int, sigma: float, channels: int, device) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.expand(channels, 1, size, size).contiguous()


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    channels = x.shape[1]
    window = _gaussian_window(window_size, sigma, channels, x.device)
    padding = window_size // 2
    mu_x = F.conv2d(x, window, padding=padding, groups=channels)
    mu_y = F.conv2d(y, window, padding=padding, groups=channels)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, window, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy
    c1 = 0.01**2
    c2 = 0.03**2
    value = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return value.mean()


def ms_ssim(x: torch.Tensor, y: torch.Tensor, levels: int = 4) -> float:
    return ms_ssim_tensor(x, y, levels=levels).item()


def ms_ssim_tensor(x: torch.Tensor, y: torch.Tensor, levels: int = 4) -> torch.Tensor:
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363], device=x.device)
    values = []
    cur_x = x
    cur_y = y
    usable_levels = min(levels, int(math.log2(min(x.shape[-2:]))))
    for _ in range(max(usable_levels, 1)):
        values.append(ssim(cur_x, cur_y).clamp(1e-6, 1.0))
        if min(cur_x.shape[-2:]) <= 32:
            break
        cur_x = F.avg_pool2d(cur_x, kernel_size=2)
        cur_y = F.avg_pool2d(cur_y, kernel_size=2)
    weights = weights[: len(values)]
    weights = weights / weights.sum()
    stacked = torch.stack(values)
    return torch.prod(stacked ** weights)


def ms_ssim_loss(x: torch.Tensor, y: torch.Tensor, levels: int = 4) -> torch.Tensor:
    return 1.0 - ms_ssim_tensor(x, y, levels=levels)
