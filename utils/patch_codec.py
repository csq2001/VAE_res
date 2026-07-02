from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models import LatentInpainter, VaeResidualCodec
from models.vae import quantize_image_ste

from .block_bitstream import StreamRecoveryReport
from .latent_tile_codec import (
    LatentTileAddress,
    LatentTileConfig,
    LatentTilePacket,
    add_latent_rs_parity,
    encode_latent_tile,
    recover_latent_rs_tiles,
)
from .ecc_rs import RSConfig
from .marker_code import MarkerConfig
from .packet_format import DNAPacket, PacketConfig, normalize_image_id
from .residual_tile_codec import (
    ResidualTileAddress,
    ResidualTileConfig,
    ResidualTileError,
    ResidualTilePacket,
    decode_residual_tile,
    encode_residual_tile,
)


class BlockDecodeError(RuntimeError):
    pass


@dataclass
class BlockCodecOutput:
    block_id: int
    row_id: int
    col_id: int
    y_hat: torch.Tensor
    q: torch.Tensor
    residual_logits: torch.Tensor
    tau: int
    block_size: int
    original_image_size: tuple[int, int]
    encoded_image_size: tuple[int, int]
    latent_stream: bytes
    residual_stream: bytes
    latent_shape: tuple[int, int, int, int]
    latent_packets: list[DNAPacket]
    latent_tiles: list[LatentTilePacket]
    residual_packets: list[DNAPacket]
    residual_tiles: list[ResidualTilePacket]
    reference_reconstruction: torch.Tensor

    def packets(
        self,
    ) -> list[DNAPacket | LatentTilePacket | ResidualTilePacket]:
        return (
            self.latent_packets
            + self.latent_tiles
            + self.residual_packets
            + self.residual_tiles
        )


@dataclass
class EncodedBlockImage:
    image_id: int
    original_height: int
    original_width: int
    channels: int
    block_size: int
    tau: int
    rows: int
    columns: int
    blocks: list[BlockCodecOutput]

    def packets(
        self,
    ) -> list[DNAPacket | LatentTilePacket | ResidualTilePacket]:
        return [packet for block in self.blocks for packet in block.packets()]

    def reference_image(self) -> torch.Tensor:
        if len(self.blocks) == 1:
            return self.blocks[0].reference_reconstruction[
                ..., : self.original_height, : self.original_width
            ]
        canvas = torch.zeros(
            1,
            self.channels,
            self.rows * self.block_size,
            self.columns * self.block_size,
        )
        for block in self.blocks:
            top = block.row_id * self.block_size
            left = block.col_id * self.block_size
            block_height, block_width = block.encoded_image_size
            canvas[
                :,
                :,
                top : top + block_height,
                left : left + block_width,
            ] = block.reference_reconstruction
        return canvas[..., : self.original_height, : self.original_width]


@dataclass(frozen=True)
class BlockRecoveryReport:
    block_id: int
    latent: StreamRecoveryReport
    residual: StreamRecoveryReport
    latent_corrected_codewords: int = 0
    latent_rs_recovered_tiles: int = 0
    latent_predicted_tiles: int = 0
    residual_corrected_codewords: int = 0


@dataclass
class DecodedBlockImage:
    image: torch.Tensor
    blocks: list[BlockRecoveryReport]


class BlockDNACodec:
    def __init__(
        self,
        model: VaeResidualCodec,
        *,
        block_size: int = 64,
        tau: int = 5,
        residual_codec: str = "zlib",
        packet_config: PacketConfig | None = None,
        rs_config: RSConfig | None = None,
        marker_config: MarkerConfig | None = None,
        latent_tile_config: LatentTileConfig | None = None,
        residual_tile_config: ResidualTileConfig | None = None,
        latent_inpainter: LatentInpainter | None = None,
        rans_precision: int = 16,
        max_homopolymer: int = 3,
    ) -> None:
        if block_size <= 0 or block_size % 8:
            raise ValueError("block_size must be a positive multiple of 8")
        if tau < 0:
            raise ValueError("tau cannot be negative")
        if residual_codec not in {"zlib", "rans"}:
            raise ValueError("residual_codec must be zlib or rans")
        self.model = model
        self.block_size = block_size
        self.tau = tau
        self.residual_codec = residual_codec
        self.packet_config = packet_config or PacketConfig()
        self.rs_config = rs_config or RSConfig()
        self.marker_config = marker_config or MarkerConfig()
        self.latent_tile_config = latent_tile_config or LatentTileConfig()
        self.residual_tile_config = residual_tile_config or ResidualTileConfig()
        self.latent_inpainter = latent_inpainter
        self.rans_precision = rans_precision
        self.max_homopolymer = max_homopolymer

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _split_image(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int, int, int]:
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("encode_image expects shape [1, C, H, W]")
        _, channels, height, width = image.shape
        if channels != self.model.in_channels:
            raise ValueError(
                f"model expects {self.model.in_channels} channels, received {channels}"
            )
        rows = (height + self.block_size - 1) // self.block_size
        columns = (width + self.block_size - 1) // self.block_size
        padded = F.pad(
            image,
            (
                0,
                columns * self.block_size - width,
                0,
                rows * self.block_size - height,
            ),
        )
        blocks = (
            padded.unfold(2, self.block_size, self.block_size)
            .unfold(3, self.block_size, self.block_size)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(-1, channels, self.block_size, self.block_size)
        )
        return blocks, height, width, rows, columns

    @staticmethod
    def _pad_full_image(
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("encode_image expects shape [1, C, H, W]")
        height, width = image.shape[-2:]
        padded_height = (height + 7) // 8 * 8
        padded_width = (width + 7) // 8 * 8
        padded = F.pad(
            image,
            (0, padded_width - width, 0, padded_height - height),
        )
        return padded, height, width

    @torch.no_grad()
    def encode_image(
        self,
        image: torch.Tensor,
        image_id: int | str = 0,
    ) -> EncodedBlockImage:
        image = image.to(self.device)
        full_image, height, width = self._pad_full_image(image)
        if full_image.shape[1] != self.model.in_channels:
            raise ValueError(
                f"model expects {self.model.in_channels} channels, "
                f"received {full_image.shape[1]}"
            )
        encoded_height, encoded_width = full_image.shape[-2:]
        rows = columns = 1
        self.model.eval()
        output = self.model(full_image, tau=self.tau, deterministic=True)
        normalized_image_id = normalize_image_id(image_id)
        encoded_blocks: list[BlockCodecOutput] = []
        for block_id in range(1):
            row_id = col_id = 0
            y_hat = output.y_hat[block_id : block_id + 1].cpu()
            q = output.q[block_id : block_id + 1].cpu()
            residual_logits = output.residual_logits[block_id : block_id + 1].cpu()
            latent_data_tiles: list[LatentTilePacket] = []
            latent_size = self.latent_tile_config.spatial_size
            channel_group = self.latent_tile_config.channel_group
            for channel_start in range(0, y_hat.shape[1], channel_group):
                channel_count = min(
                    channel_group,
                    y_hat.shape[1] - channel_start,
                )
                for latent_row in range(0, y_hat.shape[2], latent_size):
                    tile_height = min(
                        latent_size,
                        y_hat.shape[2] - latent_row,
                    )
                    for latent_col in range(0, y_hat.shape[3], latent_size):
                        tile_width = min(
                            latent_size,
                            y_hat.shape[3] - latent_col,
                        )
                        latent_data_tiles.append(
                            encode_latent_tile(
                                y_hat[
                                    :,
                                    channel_start : channel_start + channel_count,
                                    latent_row : latent_row + tile_height,
                                    latent_col : latent_col + tile_width,
                                ],
                                LatentTileAddress(
                                    image_id=normalized_image_id,
                                    block_id=block_id,
                                    tile_row=latent_row // latent_size,
                                    tile_col=latent_col // latent_size,
                                    tile_height=tile_height,
                                    tile_width=tile_width,
                                    channel_start=channel_start,
                                    channel_count=channel_count,
                                    payload_length=0,
                                ),
                                self.latent_tile_config,
                            )
                        )
            latent_stream = b"".join(
                tile.payload for tile in latent_data_tiles
            )
            latent_tiles = add_latent_rs_parity(
                latent_data_tiles,
                self.latent_tile_config,
            )
            residual_tiles: list[ResidualTilePacket] = []
            tile_size = self.residual_tile_config.tile_size
            tile_rows = (encoded_height + tile_size - 1) // tile_size
            tile_columns = (encoded_width + tile_size - 1) // tile_size
            for tile_row in range(tile_rows):
                top = tile_row * tile_size
                tile_height = min(tile_size, encoded_height - top)
                for tile_col in range(tile_columns):
                    left = tile_col * tile_size
                    tile_width = min(tile_size, encoded_width - left)
                    q_tile = q[
                        :,
                        :,
                        top : top + tile_height,
                        left : left + tile_width,
                    ]
                    residual_tiles.append(
                        encode_residual_tile(
                            q_tile,
                            ResidualTileAddress(
                                image_id=normalized_image_id,
                                block_id=block_id,
                                tile_row=tile_row,
                                tile_col=tile_col,
                                tile_height=tile_height,
                                tile_width=tile_width,
                                channels=q.shape[1],
                                tau=self.tau,
                                payload_length=0,
                            ),
                            self.residual_tile_config,
                        )
                    )
            residual_stream = b"".join(tile.payload for tile in residual_tiles)
            encoded_blocks.append(
                BlockCodecOutput(
                    block_id=block_id,
                    row_id=row_id,
                    col_id=col_id,
                    y_hat=y_hat,
                    q=q,
                    residual_logits=residual_logits,
                    tau=self.tau,
                    block_size=max(encoded_height, encoded_width),
                    original_image_size=(height, width),
                    encoded_image_size=(encoded_height, encoded_width),
                    latent_stream=latent_stream,
                    residual_stream=residual_stream,
                    latent_shape=tuple(y_hat.shape),
                    latent_packets=[],
                    latent_tiles=latent_tiles,
                    residual_packets=[],
                    residual_tiles=residual_tiles,
                    reference_reconstruction=output.x_hat[
                        block_id : block_id + 1
                    ].cpu(),
                )
            )
        return EncodedBlockImage(
            image_id=normalized_image_id,
            original_height=height,
            original_width=width,
            channels=image.shape[1],
            block_size=self.block_size,
            tau=self.tau,
            rows=rows,
            columns=columns,
            blocks=encoded_blocks,
        )

    @torch.no_grad()
    def decode_image(self, encoded: EncodedBlockImage) -> DecodedBlockImage:
        canvas = torch.zeros(
            1,
            encoded.channels,
            max(block.encoded_image_size[0] for block in encoded.blocks),
            max(block.encoded_image_size[1] for block in encoded.blocks),
            device=self.device,
        )
        reports: list[BlockRecoveryReport] = []
        for block in encoded.blocks:
            prior = torch.round(self.model.prior.loc.detach()).reshape(
                1, -1, 1, 1
            )
            y_hat = prior.expand(block.latent_shape).clone().to(self.device)
            valid_latent_mask = torch.zeros_like(y_hat)
            latent_recovery = recover_latent_rs_tiles(
                block.latent_tiles,
                self.latent_tile_config,
            )
            latent_size = self.latent_tile_config.spatial_size
            for tile in block.latent_tiles:
                if tile.header.is_parity:
                    continue
                tile_result = latent_recovery.tiles.get(
                    tile.header.packet_index
                )
                if tile_result is None:
                    continue
                top = tile.header.tile_row * latent_size
                left = tile.header.tile_col * latent_size
                channel_end = (
                    tile.header.channel_start + tile.header.channel_count
                )
                y_hat[
                    :,
                    tile.header.channel_start : channel_end,
                    top : top + tile.header.tile_height,
                    left : left + tile.header.tile_width,
                ] = tile_result.y_hat.to(self.device)
                valid_latent_mask[
                    :,
                    tile.header.channel_start : channel_end,
                    top : top + tile.header.tile_height,
                    left : left + tile.header.tile_width,
                ] = 1.0
            latent_data_count = sum(
                not tile.header.is_parity for tile in block.latent_tiles
            )
            latent_missing_tiles = (
                latent_data_count - len(latent_recovery.tiles)
            )
            latent_predicted_tiles = 0
            if self.latent_inpainter is not None and latent_missing_tiles:
                self.latent_inpainter.eval()
                prior_scale = F.softplus(
                    self.model.prior.log_scale.detach()
                ) + 1e-5
                inpainted = self.latent_inpainter(
                    y_hat,
                    valid_latent_mask,
                    self.model.prior.loc.detach(),
                    prior_scale,
                )
                y_hat = inpainted.repaired
                latent_predicted_tiles = latent_missing_tiles
            latent_report = StreamRecoveryReport(
                total_packets=len(block.latent_tiles),
                valid_packets=latent_recovery.valid_packets,
                erasures=latent_missing_tiles,
                recovered_data_packets=latent_recovery.corrected_tiles,
                errors=latent_recovery.errors,
            )

            q = torch.zeros(
                1,
                encoded.channels,
                block.encoded_image_size[0],
                block.encoded_image_size[1],
                device=self.device,
            )
            residual_errors: list[str] = []
            residual_valid = 0
            residual_corrected_tiles = 0
            residual_corrected_codewords = 0
            tile_size = self.residual_tile_config.tile_size
            for tile in block.residual_tiles:
                try:
                    tile_result = decode_residual_tile(
                        tile,
                        self.residual_tile_config,
                    )
                    top = tile.header.tile_row * tile_size
                    left = tile.header.tile_col * tile_size
                    q[
                        :,
                        :,
                        top : top + tile.header.tile_height,
                        left : left + tile.header.tile_width,
                    ] = tile_result.q.to(self.device)
                    tile.is_erasure = False
                    tile.error = None
                    residual_valid += 1
                    residual_corrected_codewords += tile_result.corrected_codewords
                    residual_corrected_tiles += int(
                        tile_result.corrected_codewords > 0
                    )
                except ResidualTileError as exc:
                    tile.is_erasure = True
                    tile.error = str(exc)
                    residual_errors.append(
                        f"tile ({tile.header.tile_row},"
                        f"{tile.header.tile_col}): {exc}"
                    )
            residual_report = StreamRecoveryReport(
                total_packets=len(block.residual_tiles),
                valid_packets=residual_valid,
                erasures=len(block.residual_tiles) - residual_valid,
                recovered_data_packets=residual_corrected_tiles,
                errors=tuple(residual_errors),
            )

            x_tilde = self.model.decoder(y_hat)
            if x_tilde.shape[-2:] != block.encoded_image_size:
                x_tilde = F.interpolate(
                    x_tilde,
                    size=block.encoded_image_size,
                    mode="bilinear",
                    align_corners=False,
                )
            tilde_pixels = quantize_image_ste(x_tilde)
            reconstructed = (
                tilde_pixels + q * float(2 * encoded.tau + 1)
            ).clamp(0.0, 255.0) / 255.0
            top = 0 if len(encoded.blocks) == 1 else block.row_id * encoded.block_size
            left = 0 if len(encoded.blocks) == 1 else block.col_id * encoded.block_size
            block_height, block_width = block.encoded_image_size
            canvas[
                :,
                :,
                top : top + block_height,
                left : left + block_width,
            ] = reconstructed
            reports.append(
                BlockRecoveryReport(
                    block_id=block.block_id,
                    latent=latent_report,
                    residual=residual_report,
                    latent_corrected_codewords=(
                        latent_recovery.corrected_codewords
                    ),
                    latent_rs_recovered_tiles=(
                        latent_recovery.rs_recovered_tiles
                    ),
                    latent_predicted_tiles=latent_predicted_tiles,
                    residual_corrected_codewords=residual_corrected_codewords,
                )
            )
        image = canvas[..., : encoded.original_height, : encoded.original_width].cpu()
        return DecodedBlockImage(image=image, blocks=reports)
