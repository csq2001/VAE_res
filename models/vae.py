from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F

from .decoder import ImageDecoder
from .encoder import ImageEncoder
from .residual_entropy import ResidualEntropyModel


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


def quantize_image_ste(x: torch.Tensor) -> torch.Tensor:
    pixels = ste_round(x.clamp(0.0, 1.0) * 255.0)
    return pixels.clamp(0.0, 255.0)


@dataclass
class CodecOutput:
    x_tilde: torch.Tensor
    x_hat: torch.Tensor
    y: torch.Tensor
    y_hat: torch.Tensor
    q: torch.Tensor
    latent_bits: torch.Tensor
    residual_bits: torch.Tensor
    residual_logits: torch.Tensor


class FactorizedGaussianPrior(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(channels))
        self.log_scale = nn.Parameter(torch.zeros(channels))

    def rate_bits(self, y_hat: torch.Tensor, quant_step: float = 1.0) -> torch.Tensor:
        loc = self.loc.view(1, -1, 1, 1)
        scale = F.softplus(self.log_scale).view(1, -1, 1, 1) + 1e-5
        dist = Normal(loc.expand_as(y_hat), scale.expand_as(y_hat))
        half = quant_step / 2.0
        upper = dist.cdf(y_hat + half)
        lower = dist.cdf(y_hat - half)
        prob = (upper - lower).clamp_min(1e-9)
        return -torch.log2(prob).sum(dim=(1, 2, 3))


class VaeResidualCodec(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 64,
        base_channels: int = 64,
        residual_hidden: int = 64,
        residual_condition_channels: int = 16,
        residual_extra_blocks: int = 1,
        max_q: int = 64,
        latent_quant_step: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_quant_step = latent_quant_step
        self.encoder = ImageEncoder(in_channels, latent_channels, base_channels)
        self.decoder = ImageDecoder(in_channels, latent_channels, base_channels)
        self.prior = FactorizedGaussianPrior(latent_channels)
        self.residual_condition_channels = residual_condition_channels
        self.residual_condition = (
            nn.Sequential(
                nn.Conv2d(latent_channels, residual_condition_channels, 1),
                nn.LeakyReLU(0.1, inplace=True),
            )
            if residual_condition_channels > 0
            else None
        )
        self.residual_entropy = ResidualEntropyModel(
            in_channels,
            condition_channels=residual_condition_channels,
            hidden=residual_hidden,
            max_q=max_q,
            extra_blocks=residual_extra_blocks,
        )

    def quantize_latent(self, y: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        step = self.latent_quant_step
        if self.training and not deterministic:
            noise = torch.empty_like(y).uniform_(-0.5 * step, 0.5 * step)
            return y + noise
        return torch.round(y / step) * step

    def forward(self, x: torch.Tensor, tau: int = 2, deterministic: bool = False) -> CodecOutput:
        y = self.encoder(x)
        y_hat = self.quantize_latent(y, deterministic=deterministic)
        x_tilde = self.decoder(y_hat)
        if x_tilde.shape[-2:] != x.shape[-2:]:
            x_tilde = F.interpolate(x_tilde, size=x.shape[-2:], mode="bilinear", align_corners=False)

        step = 2 * tau + 1
        x_pixels = torch.round(x.clamp(0.0, 1.0) * 255.0)
        tilde_pixels = quantize_image_ste(x_tilde)
        residual = x_pixels - tilde_pixels
        q = ste_round(residual / float(step)).clamp(
            -self.residual_entropy.max_q, self.residual_entropy.max_q
        )
        x_hat_pixels = (tilde_pixels + q * step).clamp(0.0, 255.0)
        x_hat = x_hat_pixels / 255.0

        condition = None
        if self.residual_condition is not None:
            condition = self.residual_condition(y_hat)
            condition = F.interpolate(condition, size=x_tilde.shape[-2:], mode="bilinear", align_corners=False)
        residual_logits = self.residual_entropy(x_tilde, condition)
        latent_bits = self.prior.rate_bits(y_hat, self.latent_quant_step)
        residual_bits = self.residual_entropy.rate_bits(residual_logits, q)
        return CodecOutput(x_tilde, x_hat, y, y_hat, q, latent_bits, residual_bits, residual_logits)
