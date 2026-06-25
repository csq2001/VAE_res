import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualEntropyModel(nn.Module):
    """Predicts p(q | x_tilde, latent condition) for quantized residual symbols."""

    def __init__(
        self,
        channels: int = 1,
        condition_channels: int = 16,
        hidden: int = 96,
        max_q: int = 64,
        extra_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.max_q = max_q
        self.num_symbols = 2 * max_q + 1
        layers = [
            nn.Conv2d(channels + condition_channels, hidden, 5, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        for _ in range(extra_blocks):
            layers.extend([nn.Conv2d(hidden, hidden, 3, padding=1), nn.LeakyReLU(0.1, inplace=True)])
        layers.append(nn.Conv2d(hidden, channels * self.num_symbols, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_tilde: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        if condition is None:
            return self.net(x_tilde)
        return self.net(torch.cat([x_tilde, condition], dim=1))

    def symbols_to_targets(self, q: torch.Tensor) -> torch.Tensor:
        return (q.clamp(-self.max_q, self.max_q) + self.max_q).long()

    def targets_to_symbols(self, targets: torch.Tensor) -> torch.Tensor:
        return targets.long() - self.max_q

    def rate_bits(self, logits: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        targets = self.symbols_to_targets(q)
        batch, _, height, width = logits.shape
        logits = logits.view(batch, self.channels, self.num_symbols, height, width)
        logits = logits.permute(0, 1, 3, 4, 2).reshape(-1, self.num_symbols)
        targets = targets.permute(0, 1, 2, 3).reshape(-1)
        nll = F.cross_entropy(logits, targets, reduction="none")
        nll = nll.view(batch, self.channels, height, width)
        return nll.sum(dim=(1, 2, 3)) / torch.log(torch.tensor(2.0, device=nll.device))
