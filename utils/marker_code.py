from __future__ import annotations

from dataclasses import dataclass

from .dna_constraints import dna_stats


class MarkerDecodeError(ValueError):
    pass


BLOCK_START_MARKER = "ACGTACGTCAGTGCATACGA"
LATENT_MARKER = "AGTCGATGCTACGTAGCATA"
RESIDUAL_MARKER = "TCAGCTACGATGTCGACGTA"
SYNC_MARKER = "GATCGTACAGCTAGTCGATC"
BLOCK_END_MARKER = "CGTATCAGTGACGCTACGAT"


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_base in enumerate(left, start=1):
        current = [row]
        for column, right_base in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_base != right_base),
                )
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class MarkerConfig:
    block_start_marker: str = BLOCK_START_MARKER
    latent_marker: str = LATENT_MARKER
    residual_marker: str = RESIDUAL_MARKER
    sync_marker: str = SYNC_MARKER
    block_end_marker: str = BLOCK_END_MARKER
    sync_interval: int = 100
    max_marker_edits: int = 2
    max_sync_drift: int = 12

    def __post_init__(self) -> None:
        markers = self.markers()
        for name, marker in markers.items():
            if not marker or any(base not in "ACGT" for base in marker):
                raise ValueError(f"{name} must contain only A/C/G/T")
            stats = dna_stats(marker)
            if not 0.4 <= stats.gc <= 0.6:
                raise ValueError(f"{name} GC content must be between 40% and 60%")
            if stats.max_homopolymer > 2:
                raise ValueError(f"{name} contains a homopolymer longer than 2")
        if self.sync_interval < 16:
            raise ValueError("sync_interval must be at least 16 nt")
        minimum_distance = levenshtein_distance(self.latent_marker, self.residual_marker)
        if minimum_distance < max(6, len(self.latent_marker) // 3):
            raise ValueError("latent and residual markers are not sufficiently separated")

    def markers(self) -> dict[str, str]:
        return {
            "block_start_marker": self.block_start_marker,
            "latent_marker": self.latent_marker,
            "residual_marker": self.residual_marker,
            "sync_marker": self.sync_marker,
            "block_end_marker": self.block_end_marker,
        }


@dataclass(frozen=True)
class MarkerMatch:
    start: int
    end: int
    distance: int


def _best_marker_match(
    sequence: str,
    marker: str,
    search_start: int,
    search_end: int,
    max_edits: int,
) -> MarkerMatch | None:
    best: MarkerMatch | None = None
    minimum_length = max(1, len(marker) - max_edits)
    maximum_length = len(marker) + max_edits
    search_start = max(search_start, 0)
    search_end = min(search_end, len(sequence))
    for start in range(search_start, search_end):
        for length in range(minimum_length, maximum_length + 1):
            end = start + length
            if end > len(sequence):
                continue
            distance = levenshtein_distance(sequence[start:end], marker)
            if distance <= max_edits and (
                best is None
                or distance < best.distance
                or (distance == best.distance and start < best.start)
            ):
                best = MarkerMatch(start, end, distance)
    return best


def classify_stream_marker(
    marker_sequence: str,
    config: MarkerConfig | None = None,
) -> str:
    config = config or MarkerConfig()
    latent_distance = levenshtein_distance(marker_sequence, config.latent_marker)
    residual_distance = levenshtein_distance(marker_sequence, config.residual_marker)
    best = min(latent_distance, residual_distance)
    if best > config.max_marker_edits or latent_distance == residual_distance:
        raise MarkerDecodeError("stream marker cannot be classified")
    return "latent" if latent_distance < residual_distance else "residual"


def frame_payload_dna(
    payload_dna: str,
    stream_type: str,
    config: MarkerConfig | None = None,
) -> str:
    config = config or MarkerConfig()
    if stream_type == "latent":
        stream_marker = config.latent_marker
    elif stream_type == "residual":
        stream_marker = config.residual_marker
    else:
        raise ValueError(f"unsupported stream type: {stream_type}")
    chunks = [
        payload_dna[index : index + config.sync_interval]
        for index in range(0, len(payload_dna), config.sync_interval)
    ]
    body = config.sync_marker.join(chunks)
    return config.block_start_marker + stream_marker + body + config.block_end_marker


def deframe_payload_dna(
    sequence: str,
    config: MarkerConfig | None = None,
) -> tuple[str, str]:
    config = config or MarkerConfig()
    for stream_type, stream_marker in (
        ("latent", config.latent_marker),
        ("residual", config.residual_marker),
    ):
        prefix = config.block_start_marker + stream_marker
        if sequence.startswith(prefix) and sequence.endswith(config.block_end_marker):
            body = sequence[len(prefix) : -len(config.block_end_marker)]
            return stream_type, "".join(body.split(config.sync_marker))

    edge_slack = config.max_marker_edits + config.max_sync_drift
    start_match = _best_marker_match(
        sequence,
        config.block_start_marker,
        0,
        min(len(sequence), len(config.block_start_marker) + edge_slack),
        config.max_marker_edits,
    )
    if start_match is None:
        raise MarkerDecodeError("block start marker not found")

    marker_search_end = min(
        len(sequence),
        start_match.end + max(len(config.latent_marker), len(config.residual_marker)) + edge_slack,
    )
    latent_match = _best_marker_match(
        sequence,
        config.latent_marker,
        start_match.end,
        marker_search_end,
        config.max_marker_edits,
    )
    residual_match = _best_marker_match(
        sequence,
        config.residual_marker,
        start_match.end,
        marker_search_end,
        config.max_marker_edits,
    )
    candidates = [
        ("latent", latent_match),
        ("residual", residual_match),
    ]
    candidates = [(name, match) for name, match in candidates if match is not None]
    if not candidates:
        raise MarkerDecodeError("stream marker not found")
    candidates.sort(key=lambda item: (item[1].distance, item[1].start))
    if len(candidates) > 1 and candidates[0][1].distance == candidates[1][1].distance:
        raise MarkerDecodeError("stream marker is ambiguous")
    stream_type, stream_match = candidates[0]

    end_search_start = max(
        stream_match.end,
        len(sequence) - len(config.block_end_marker) - edge_slack,
    )
    end_match = _best_marker_match(
        sequence,
        config.block_end_marker,
        end_search_start,
        len(sequence),
        config.max_marker_edits,
    )
    if end_match is None:
        raise MarkerDecodeError("block end marker not found")
    body = sequence[stream_match.end : end_match.start]

    payload_parts: list[str] = []
    cursor = 0
    while len(body) - cursor > config.sync_interval:
        expected = cursor + config.sync_interval
        search_start = max(cursor, expected - config.max_sync_drift)
        search_end = min(
            len(body),
            expected + config.max_sync_drift + len(config.sync_marker) + config.max_marker_edits,
        )
        sync_match = _best_marker_match(
            body,
            config.sync_marker,
            search_start,
            search_end,
            config.max_marker_edits,
        )
        if sync_match is None:
            # There is no marker after the final (possibly insertion-expanded) chunk.
            if len(body) - cursor <= config.sync_interval + config.max_sync_drift:
                break
            raise MarkerDecodeError("periodic sync marker not found")
        payload_parts.append(body[cursor : sync_match.start])
        cursor = sync_match.end
    payload_parts.append(body[cursor:])
    return stream_type, "".join(payload_parts)
