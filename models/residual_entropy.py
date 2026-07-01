import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualEntropyModel(nn.Module):
    """Predicts residual symbols with optional two-pass checkerboard context."""

    def __init__(
        self,
        channels: int = 1,
        condition_channels: int = 16,
        hidden: int = 96,
        max_q: int = 64,
        extra_blocks: int = 1,
        checkerboard_context: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.max_q = max_q
        self.num_symbols = 2 * max_q + 1
        self.checkerboard_context = checkerboard_context
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
        self.context_net = (
            nn.Sequential(
                nn.Conv2d(channels + 1, hidden, 5, padding=2),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(hidden, hidden, 3, padding=1),
            )
            if checkerboard_context
            else None
        )

    @staticmethod
    def anchor_mask(reference: torch.Tensor) -> torch.Tensor:
        height, width = reference.shape[-2:]
        rows = torch.arange(height, device=reference.device).view(height, 1)
        columns = torch.arange(width, device=reference.device).view(1, width)
        return ((rows + columns) % 2 == 0).to(reference.dtype).view(1, 1, height, width)

    def forward(
        self,
        x_tilde: torch.Tensor,
        condition: torch.Tensor | None = None,
        q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs = x_tilde if condition is None else torch.cat([x_tilde, condition], dim=1)
        if not self.checkerboard_context:
            return self.net(inputs)
        if q is None:
            raise ValueError("q is required when checkerboard context is enabled")

        features = inputs
        for layer in list(self.net.children())[:-1]:
            features = layer(features)

        anchor_mask = self.anchor_mask(q)
        nonanchor_mask = 1.0 - anchor_mask
        q_anchors = q * anchor_mask / float(max(self.max_q, 1))
        expanded_mask = anchor_mask.expand(q.shape[0], -1, -1, -1)
        context = self.context_net(torch.cat([q_anchors, expanded_mask], dim=1))
        features = features + context * nonanchor_mask
        return self.net[-1](features)

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
