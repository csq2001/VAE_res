from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature, gate = self.conv(x).chunk(2, dim=1)
        return F.silu(feature) * torch.sigmoid(gate)


class GatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = GatedConv2d(channels, channels, dilation=dilation)
        self.conv2 = GatedConv2d(channels, channels, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


@dataclass
class LatentInpaintOutput:
    repaired: torch.Tensor
    predicted: torch.Tensor
    uncertainty: torch.Tensor
    support: torch.Tensor


class LatentInpainter(nn.Module):
    """Mask-aware latent completion network for the frozen VAE codec.

    ``valid_mask`` uses 1 for CRC/RS-valid coefficients and 0 for missing
    coefficients. The final fusion guarantees that valid latent values are
    never changed by this network.
    """

    def __init__(
        self,
        latent_channels: int = 64,
        channel_group: int = 32,
        hidden_channels: int = 128,
        context_channels: int = 192,
        quant_step: float = 1.0,
    ) -> None:
        super().__init__()
        if latent_channels < 1 or channel_group < 1:
            raise ValueError("latent_channels and channel_group must be positive")
        self.latent_channels = latent_channels
        self.channel_group = channel_group
        self.groups = (latent_channels + channel_group - 1) // channel_group
        self.quant_step = float(quant_step)
        input_channels = latent_channels * 2 + self.groups

        self.stem = GatedConv2d(input_channels, hidden_channels)
        self.local = nn.Sequential(
            GatedResidualBlock(hidden_channels, dilation=1),
            GatedResidualBlock(hidden_channels, dilation=2),
        )
        self.down = GatedConv2d(
            hidden_channels,
            context_channels,
            stride=2,
        )
        self.context = nn.Sequential(
            GatedResidualBlock(context_channels, dilation=1),
            GatedResidualBlock(context_channels, dilation=2),
            GatedResidualBlock(context_channels, dilation=4),
            GatedResidualBlock(context_channels, dilation=8),
        )
        self.up = GatedConv2d(context_channels, hidden_channels)
        self.fuse = GatedConv2d(hidden_channels * 2, hidden_channels)
        self.refine = GatedResidualBlock(hidden_channels)
        self.latent_head = nn.Conv2d(hidden_channels, latent_channels, 3, padding=1)
        self.uncertainty_head = nn.Conv2d(
            hidden_channels,
            self.groups,
            3,
            padding=1,
        )

    def group_support(self, valid_mask: torch.Tensor) -> torch.Tensor:
        support: list[torch.Tensor] = []
        for start in range(0, self.latent_channels, self.channel_group):
            group_mask = valid_mask[
                :, start : min(start + self.channel_group, self.latent_channels)
            ].mean(dim=1, keepdim=True)
            support.append(
                F.avg_pool2d(group_mask, kernel_size=7, stride=1, padding=3)
            )
        return torch.cat(support, dim=1)

    def _quantize_ste(self, value: torch.Tensor) -> torch.Tensor:
        if self.quant_step <= 0:
            return value
        quantized = torch.round(value / self.quant_step) * self.quant_step
        if self.training:
            return value + (quantized - value).detach()
        return quantized

    def forward(
        self,
        latent: torch.Tensor,
        valid_mask: torch.Tensor,
        prior_loc: torch.Tensor,
        prior_scale: torch.Tensor,
    ) -> LatentInpaintOutput:
        if latent.shape != valid_mask.shape:
            raise ValueError("latent and valid_mask must have the same shape")
        if latent.shape[1] != self.latent_channels:
            raise ValueError(
                f"expected {self.latent_channels} latent channels, "
                f"received {latent.shape[1]}"
            )
        loc = prior_loc.reshape(1, -1, 1, 1).to(latent)
        scale = prior_scale.reshape(1, -1, 1, 1).to(latent).clamp_min(1e-4)
        normalized = (latent - loc) / scale
        support = self.group_support(valid_mask)
        features = self.local(
            self.stem(torch.cat((normalized, valid_mask, support), dim=1))
        )
        context = self.context(self.down(features))
        context = F.interpolate(
            context,
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        context = self.up(context)
        fused = self.refine(self.fuse(torch.cat((features, context), dim=1)))
        predicted = loc + scale * self.latent_head(fused)
        predicted = self._quantize_ste(predicted)
        repaired = valid_mask * latent + (1.0 - valid_mask) * predicted
        uncertainty = self.uncertainty_head(fused).clamp(-6.0, 6.0)
        return LatentInpaintOutput(
            repaired=repaired,
            predicted=predicted,
            uncertainty=uncertainty,
            support=support,
        )
