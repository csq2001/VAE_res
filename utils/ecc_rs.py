from __future__ import annotations

import importlib.util
from dataclasses import dataclass


class ReedSolomonError(ValueError):
    pass


@dataclass(frozen=True)
class RSConfig:
    data_shards: int = 6
    parity_shards: int = 2

    def __post_init__(self) -> None:
        if self.data_shards < 1:
            raise ValueError("data_shards must be positive")
        if self.parity_shards < 1:
            raise ValueError("parity_shards must be positive")
        if self.data_shards + self.parity_shards > 255:
            raise ValueError("total RS shards cannot exceed 255")


REEDSOLO_AVAILABLE = importlib.util.find_spec("reedsolo") is not None


class ReedSolomonCodec:
    """Packet-erasure RS interface backed by the bundled GF(256) codec.

    ``reedsolo`` is detected through ``REEDSOLO_AVAILABLE`` and can still be
    used by callers for within-packet symbol correction. This class operates
    across equal-sized packet shards, which is the layer needed after CRC has
    converted damaged packets into erasures.
    """

    def __init__(self, config: RSConfig | None = None) -> None:
        self.config = config or RSConfig()

    def encode(self, data_shards: list[bytes]) -> list[bytes]:
        if len(data_shards) != self.config.data_shards:
            raise ValueError(
                f"expected {self.config.data_shards} data shards, "
                f"received {len(data_shards)}"
            )
        return data_shards + encode_parity_shards(
            data_shards,
            self.config.parity_shards,
        )

    def decode(
        self,
        shards: list[bytes | None],
        erasure_positions: list[int] | tuple[int, ...] | None = None,
    ) -> list[bytes]:
        total = self.config.data_shards + self.config.parity_shards
        if len(shards) != total:
            raise ValueError(f"expected {total} total shards, received {len(shards)}")
        erased = set(erasure_positions or ())
        available = {
            index: shard
            for index, shard in enumerate(shards)
            if shard is not None and index not in erased
        }
        return recover_data_shards(
            available,
            self.config.data_shards,
            self.config.parity_shards,
        )


def _make_gf_tables() -> tuple[list[int], list[int]]:
    exponent = [0] * 512
    logarithm = [0] * 256
    value = 1
    for index in range(255):
        exponent[index] = value
        logarithm[value] = index
        value <<= 1
        if value & 0x100:
            value ^= 0x11D
    for index in range(255, 512):
        exponent[index] = exponent[index - 255]
    return exponent, logarithm


GF_EXP, GF_LOG = _make_gf_tables()


def _gf_mul(left: int, right: int) -> int:
    if left == 0 or right == 0:
        return 0
    return GF_EXP[GF_LOG[left] + GF_LOG[right]]


def _gf_inv(value: int) -> int:
    if value == 0:
        raise ReedSolomonError("zero has no multiplicative inverse")
    return GF_EXP[255 - GF_LOG[value]]


def _gf_pow(value: int, power: int) -> int:
    if power == 0:
        return 1
    if value == 0:
        return 0
    return GF_EXP[(GF_LOG[value] * power) % 255]


def _matrix_multiply(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    rows = len(left)
    columns = len(right[0])
    shared = len(right)
    result = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        for column in range(columns):
            value = 0
            for index in range(shared):
                value ^= _gf_mul(left[row][index], right[index][column])
            result[row][column] = value
    return result


def _matrix_inverse(matrix: list[list[int]]) -> list[list[int]]:
    size = len(matrix)
    augmented = [
        row[:] + [1 if row_index == column else 0 for column in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if augmented[row][column] != 0),
            None,
        )
        if pivot is None:
            raise ReedSolomonError("RS shard matrix is singular")
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = _gf_inv(augmented[column][column])
        augmented[column] = [_gf_mul(value, scale) for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    value ^ _gf_mul(factor, pivot_value)
                    for value, pivot_value in zip(augmented[row], augmented[column])
                ]
    return [row[size:] for row in augmented]


def generator_matrix(data_shards: int, parity_shards: int) -> list[list[int]]:
    total = data_shards + parity_shards
    vandermonde = [
        [_gf_pow(row + 1, column) for column in range(data_shards)]
        for row in range(total)
    ]
    transform = _matrix_inverse(vandermonde[:data_shards])
    return _matrix_multiply(vandermonde, transform)


def _validate_shards(shards: list[bytes]) -> int:
    if not shards:
        raise ValueError("at least one shard is required")
    shard_size = len(shards[0])
    if shard_size == 0 or any(len(shard) != shard_size for shard in shards):
        raise ValueError("all shards must have the same non-zero length")
    return shard_size


def encode_parity_shards(data: list[bytes], parity_shards: int) -> list[bytes]:
    shard_size = _validate_shards(data)
    matrix = generator_matrix(len(data), parity_shards)
    parity: list[bytes] = []
    for row in matrix[len(data) :]:
        output = bytearray(shard_size)
        for coefficient, shard in zip(row, data):
            if coefficient == 0:
                continue
            for index, value in enumerate(shard):
                output[index] ^= _gf_mul(coefficient, value)
        parity.append(bytes(output))
    return parity


def recover_data_shards(
    available_shards: dict[int, bytes],
    data_shards: int,
    parity_shards: int,
) -> list[bytes]:
    if len(available_shards) < data_shards:
        raise ReedSolomonError(
            f"need {data_shards} shards, only {len(available_shards)} are available"
        )
    total = data_shards + parity_shards
    if any(index < 0 or index >= total for index in available_shards):
        raise ValueError("shard index is outside the RS group")
    selected_indices = sorted(available_shards)[:data_shards]
    selected = [available_shards[index] for index in selected_indices]
    shard_size = _validate_shards(selected)
    matrix = generator_matrix(data_shards, parity_shards)
    decode_matrix = _matrix_inverse([matrix[index] for index in selected_indices])

    recovered = [bytearray(shard_size) for _ in range(data_shards)]
    for output_index, row in enumerate(decode_matrix):
        for coefficient, shard in zip(row, selected):
            if coefficient == 0:
                continue
            for byte_index, value in enumerate(shard):
                recovered[output_index][byte_index] ^= _gf_mul(coefficient, value)
    return [bytes(shard) for shard in recovered]
