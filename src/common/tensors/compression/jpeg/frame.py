"""Minimal baseline grayscale JFIF serialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...abstraction import AbstractTensor
from ...autograd import autograd
from ..block_transform import orthonormal_dct_basis
from ..bitstream import tensor_octets_to_bytes, unpack_octets
from ..coefficient_events import (
    collect_block_coefficient_events,
    collect_component_block_coefficient_events,
    slice_block_coefficient_events,
)
from .huffman import (
    JPEG_AC_CHROMINANCE_COUNTS,
    JPEG_AC_CHROMINANCE_SYMBOLS,
    JPEG_AC_LUMINANCE_COUNTS,
    JPEG_AC_LUMINANCE_SYMBOLS,
    JPEG_DC_CHROMINANCE_COUNTS,
    JPEG_DC_CHROMINANCE_SYMBOLS,
    JPEG_DC_LUMINANCE_COUNTS,
    JPEG_DC_LUMINANCE_SYMBOLS,
    jpeg_standard_ac_chrominance,
    jpeg_standard_ac_luminance,
    jpeg_standard_dc_chrominance,
    jpeg_standard_dc_luminance,
)
from .scan import (
    encode_baseline_color_component_scan,
    encode_baseline_color_scan,
    encode_baseline_luma_scan,
)
from .transform import (
    JPEG_CHROMA_QUANTIZATION,
    JPEG_LUMA_QUANTIZATION,
    JPEG_ZIGZAG,
    jpeg_luma_coefficients,
    jpeg_ycbcr_coefficients,
    rgb_to_ycbcr,
)


@dataclass(frozen=True)
class JPEGEncodingResources:
    """Backend-resident constants shared by every frame of an encoder."""

    dct_basis: AbstractTensor
    luma_quantization: AbstractTensor
    chroma_quantization: AbstractTensor
    ycbcr_quantization: AbstractTensor
    zigzag: AbstractTensor
    luma_dc_table: object
    luma_ac_table: object
    chroma_dc_table: object
    chroma_ac_table: object

    def release(self) -> None:
        """Release owned accelerator storage while its context remains live."""

        tensors = [
            self.dct_basis,
            self.luma_quantization,
            self.chroma_quantization,
            self.ycbcr_quantization,
            self.zigzag,
        ]
        for table in (
            self.luma_dc_table,
            self.luma_ac_table,
            self.chroma_dc_table,
            self.chroma_ac_table,
        ):
            tensors.extend((table.codes, table.lengths))
            if table.symbols is not None:
                tensors.append(table.symbols)
        released: set[int] = set()
        for tensor in tensors:
            data = getattr(tensor, "data", None)
            release = getattr(data, "release", None)
            if callable(release) and id(data) not in released:
                release()
                released.add(id(data))


def prepare_jpeg_encoding_resources(
    like: AbstractTensor,
) -> JPEGEncodingResources:
    """Create invariant JPEG tensors once on ``like``'s active backend."""

    if not isinstance(like, AbstractTensor):
        raise TypeError("JPEG resources require an AbstractTensor exemplar")
    with autograd.no_grad():
        luma_quantization = like.ensure_tensor(JPEG_LUMA_QUANTIZATION)
        chroma_quantization = like.ensure_tensor(JPEG_CHROMA_QUANTIZATION)
        return JPEGEncodingResources(
            dct_basis=orthonormal_dct_basis(8, like=like),
            luma_quantization=luma_quantization,
            chroma_quantization=chroma_quantization,
            ycbcr_quantization=AbstractTensor.stack(
                (
                    luma_quantization,
                    chroma_quantization,
                    chroma_quantization,
                ),
                dim=0,
            ).reshape(3, 1, 1, 8, 8),
            zigzag=like.ensure_tensor(JPEG_ZIGZAG).to_dtype("int64"),
            luma_dc_table=jpeg_standard_dc_luminance(like),
            luma_ac_table=jpeg_standard_ac_luminance(like),
            chroma_dc_table=jpeg_standard_dc_chrominance(like),
            chroma_ac_table=jpeg_standard_ac_chrominance(like),
        )


def _u16(value: int) -> bytes:
    if value < 0 or value > 0xFFFF:
        raise ValueError("JPEG 16-bit field is out of range")
    return value.to_bytes(2, "big")


def _segment(marker: int, payload: bytes) -> bytes:
    return b"\xFF" + bytes((marker,)) + _u16(len(payload) + 2) + payload


def _dht_table(
    table_class: int,
    table_id: int,
    counts: tuple[int, ...],
    symbols: tuple[int, ...],
) -> bytes:
    if len(counts) != 16 or sum(counts) != len(symbols):
        raise ValueError("invalid JPEG Huffman table serialization")
    return (
        bytes(((table_class << 4) | table_id,))
        + bytes(counts)
        + bytes(symbols)
    )


def _stuff_entropy_octets(octets: AbstractTensor) -> AbstractTensor:
    """Insert a zero octet after every JPEG entropy ``0xFF`` tensor value."""
    if not isinstance(octets, AbstractTensor) or octets.ndims() != 1:
        raise TypeError("JPEG entropy octets must be a one-dimensional tensor")
    count = octets.shape[0]
    if count == 0:
        return octets
    is_marker = (octets == 0xFF).to_dtype("int64")
    marker_count = int(is_marker.sum().item())
    if marker_count == 0:
        return octets
    preceding_markers = is_marker.cumsum(dim=0) - is_marker
    destinations = (
        AbstractTensor.arange(count, cls=type(octets))
        + preceding_markers
    ).to_dtype("int64")
    stuffed = AbstractTensor.zeros(
        (count + marker_count,),
        dtype=octets.dtype,
        cls=type(octets),
    )
    with autograd.no_grad():
        stuffed = AbstractTensor.scatter(
            stuffed,
            destinations,
            octets,
            dim=0,
        )
    return stuffed


class _EntropyTensorAccumulator:
    """Join independently packed tensor scans without byte-aligning batches."""

    def __init__(self) -> None:
        self._pending: AbstractTensor | None = None

    def append(self, scan, *, final: bool = False) -> bytes:
        bit_count = int(scan.valid_bits.item())
        if final and self._pending is None:
            # ``compact_codewords`` has already formed the exact leading
            # octets.  A one-batch scan only needs JPEG's all-ones fill in the
            # unused tail of its last byte, then marker stuffing.  Expanding
            # every octet back to eight bits and reducing it to the same octet
            # was pure work introduced by the multi-batch carry path.
            byte_count = (bit_count + 7) // 8
            octets = scan.octets[:byte_count]
            remainder = bit_count % 8
            if remainder:
                fill = (1 << (8 - remainder)) - 1
                octets = AbstractTensor.cat(
                    (octets[:-1], octets[-1:] + fill),
                    dim=0,
                )
            return tensor_octets_to_bytes(
                _stuff_entropy_octets(octets)
            )

        source_bits = unpack_octets(scan).bits[:bit_count]
        combined = source_bits
        if self._pending is not None and self._pending.shape[0]:
            combined = AbstractTensor.cat(
                (self._pending, source_bits),
                dim=0,
            )

        complete_bit_count = (combined.shape[0] // 8) * 8
        self._pending = combined[complete_bit_count:]
        if complete_bit_count == 0:
            return b""

        byte_weights = combined.ensure_tensor(
            (128, 64, 32, 16, 8, 4, 2, 1)
        )
        octets = (
            combined[:complete_bit_count].reshape(-1, 8)
            * byte_weights.unsqueeze(0)
        ).sum(dim=1).to_dtype("int64")
        return tensor_octets_to_bytes(_stuff_entropy_octets(octets))

    def finish(self) -> bytes:
        if self._pending is None or self._pending.shape[0] == 0:
            return b""
        padding = 8 - self._pending.shape[0]
        padded = AbstractTensor.cat(
            (
                self._pending,
                AbstractTensor.ones(
                    (padding,), cls=type(self._pending)
                ),
            ),
            dim=0,
        )
        byte_weights = padded.ensure_tensor(
            (128, 64, 32, 16, 8, 4, 2, 1)
        )
        octets = (
            padded.reshape(1, 8) * byte_weights.unsqueeze(0)
        ).sum(dim=1).to_dtype("int64")
        self._pending = None
        return tensor_octets_to_bytes(
            _stuff_entropy_octets(octets)
        )


def _jfif_app0() -> bytes:
    return (
        b"JFIF\x00"
        + b"\x01\x02"
        + b"\x00"
        + _u16(1)
        + _u16(1)
        + b"\x00\x00"
    )


def _serialized_quantization(table) -> bytes:
    flattened = tuple(value for row in table for value in row)
    return bytes(flattened[index] for index in JPEG_ZIGZAG)


def _validate_samples(samples: AbstractTensor) -> None:
    """Validate identity without forcing numerical device materialization."""
    if not isinstance(samples, AbstractTensor):
        raise TypeError("JPEG samples must be an AbstractTensor")


def _validate_frame(
    samples: AbstractTensor,
    *,
    color: bool,
) -> tuple[int, int]:
    if not isinstance(samples, AbstractTensor):
        raise TypeError("samples must be an AbstractTensor")
    if color:
        if samples.ndims() != 3 or samples.shape[-1] != 3:
            raise ValueError(
                "color JFIF input must have shape (height, width, 3)"
            )
        height, width, _ = samples.shape
    else:
        if samples.ndims() != 2:
            raise ValueError("grayscale JFIF input must be two-dimensional")
        height, width = samples.shape
    if height < 1 or width < 1 or height > 0xFFFF or width > 0xFFFF:
        raise ValueError("JFIF dimensions must fit unsigned 16-bit fields")
    if height % 8 or width % 8:
        raise ValueError("initial JFIF encoder requires dimensions divisible by 8")
    _validate_samples(samples)
    return int(height), int(width)


def _grayscale_header(height: int, width: int) -> bytes:
    dqt = bytes((0x00,)) + _serialized_quantization(
        JPEG_LUMA_QUANTIZATION
    )
    sof0 = (
        b"\x08"
        + _u16(height)
        + _u16(width)
        + b"\x01"
        + b"\x01\x11\x00"
    )
    dht = (
        _dht_table(
            0,
            0,
            JPEG_DC_LUMINANCE_COUNTS,
            JPEG_DC_LUMINANCE_SYMBOLS,
        )
        + _dht_table(
            1,
            0,
            JPEG_AC_LUMINANCE_COUNTS,
            JPEG_AC_LUMINANCE_SYMBOLS,
        )
    )
    sos = b"\x01\x01\x00\x00\x3F\x00"
    return (
        b"\xFF\xD8"
        + _segment(0xE0, _jfif_app0())
        + _segment(0xDB, dqt)
        + _segment(0xC0, sof0)
        + _segment(0xC4, dht)
        + _segment(0xDA, sos)
    )


def _color_header(height: int, width: int) -> bytes:
    dqt = (
        b"\x00"
        + _serialized_quantization(JPEG_LUMA_QUANTIZATION)
        + b"\x01"
        + _serialized_quantization(JPEG_CHROMA_QUANTIZATION)
    )
    sof0 = (
        b"\x08"
        + _u16(height)
        + _u16(width)
        + b"\x03"
        + b"\x01\x11\x00"
        + b"\x02\x11\x01"
        + b"\x03\x11\x01"
    )
    dht = (
        _dht_table(
            0,
            0,
            JPEG_DC_LUMINANCE_COUNTS,
            JPEG_DC_LUMINANCE_SYMBOLS,
        )
        + _dht_table(
            1,
            0,
            JPEG_AC_LUMINANCE_COUNTS,
            JPEG_AC_LUMINANCE_SYMBOLS,
        )
        + _dht_table(
            0,
            1,
            JPEG_DC_CHROMINANCE_COUNTS,
            JPEG_DC_CHROMINANCE_SYMBOLS,
        )
        + _dht_table(
            1,
            1,
            JPEG_AC_CHROMINANCE_COUNTS,
            JPEG_AC_CHROMINANCE_SYMBOLS,
        )
    )
    sos = (
        b"\x03"
        + b"\x01\x00"
        + b"\x02\x11"
        + b"\x03\x11"
        + b"\x00\x3F\x00"
    )
    return (
        b"\xFF\xD8"
        + _segment(0xE0, _jfif_app0())
        + _segment(0xDB, dqt)
        + _segment(0xC0, sof0)
        + _segment(0xC4, dht)
        + _segment(0xDA, sos)
    )


def iter_jfif_chunks(
    samples: AbstractTensor,
    *,
    mcu_rows_per_batch: int = 8,
    resources: JPEGEncodingResources | None = None,
):
    """Yield a complete baseline JFIF while bounding tensor work by MCU rows."""
    if not isinstance(samples, AbstractTensor):
        raise TypeError("samples must be an AbstractTensor")
    if mcu_rows_per_batch < 1:
        raise ValueError("mcu_rows_per_batch must be positive")
    color = samples.ndims() == 3 and samples.shape[-1] == 3
    if not color and samples.ndims() != 2:
        raise ValueError(
            "JFIF input must have shape (height, width) or (height, width, 3)"
        )
    height, width = _validate_frame(samples, color=color)
    yield (
        _color_header(height, width)
        if color
        else _grayscale_header(height, width)
    )

    rows_per_batch = mcu_rows_per_batch * 8
    entropy = _EntropyTensorAccumulator()
    previous_dc = [0, 0, 0]
    resources = resources or prepare_jpeg_encoding_resources(samples)
    luma_dc_table = resources.luma_dc_table
    luma_ac_table = resources.luma_ac_table
    chroma_dc_table = resources.chroma_dc_table
    chroma_ac_table = resources.chroma_ac_table
    for row_start in range(0, height, rows_per_batch):
        row_stop = min(height, row_start + rows_per_batch)
        # JPEG serialization is a terminal, quantized byte boundary. Recording
        # thousands of entropy-coding primitives on the training tape retains
        # intermediates that can never participate in a useful backward pass.
        with autograd.no_grad():
            batch = samples[row_start:row_stop]
            if color:
                planes = rgb_to_ycbcr(batch)
                component_coefficients = jpeg_ycbcr_coefficients(
                    planes,
                    basis=resources.dct_basis,
                    quantization=resources.ycbcr_quantization,
                    zigzag=resources.zigzag,
                )
                combined_events = collect_component_block_coefficient_events(
                    component_coefficients,
                    max_magnitude_bits=11,
                    previous_dc=previous_dc,
                )
                block_count = component_coefficients[0].reshape(-1, 64).shape[0]
                y_events = slice_block_coefficient_events(
                    combined_events, 0, block_count
                )
                chroma_events = slice_block_coefficient_events(
                    combined_events, block_count, block_count * 3
                )
                previous_dc = [
                    component_coefficients[component].reshape(-1, 64)[-1, 0]
                    for component in range(3)
                ]
                scan = encode_baseline_color_component_scan(
                    y_events,
                    chroma_events,
                    luma_dc_table=luma_dc_table,
                    luma_ac_table=luma_ac_table,
                    chroma_dc_table=chroma_dc_table,
                    chroma_ac_table=chroma_ac_table,
                )
            else:
                coefficients = jpeg_luma_coefficients(
                    batch,
                    basis=resources.dct_basis,
                    quantization=resources.luma_quantization,
                    zigzag=resources.zigzag,
                )
                events = collect_block_coefficient_events(
                    coefficients,
                    max_magnitude_bits=11,
                    previous_dc=previous_dc[0],
                )
                previous_dc[0] = coefficients.reshape(-1, 64)[-1, 0]
                scan = encode_baseline_luma_scan(
                    events,
                    dc_table=luma_dc_table,
                    ac_table=luma_ac_table,
                )
            encoded = entropy.append(
                scan,
                final=row_start == 0 and row_stop == height,
            )
        if encoded:
            yield encoded
    tail = entropy.finish()
    if tail:
        yield tail
    yield b"\xFF\xD9"


def iter_ycbcr_jfif_chunks(
    planes,
    *,
    mcu_rows_per_batch: int = 8,
    resources: JPEGEncodingResources | None = None,
):
    """Yield baseline 4:4:4 JFIF from resident Y, Cb, and Cr planes.

    This entry point lets a graph optimizer fuse an image producer, palette,
    and color transform without forcing an RGB stack or a second color
    conversion. The remaining block transform and entropy stages stay ordinary
    AbstractTensor composition.
    """

    if mcu_rows_per_batch < 1:
        raise ValueError("mcu_rows_per_batch must be positive")
    if not isinstance(planes, (tuple, list)) or len(planes) != 3:
        raise ValueError("YCbCr input must contain exactly three planes")
    if any(not isinstance(plane, AbstractTensor) for plane in planes):
        raise TypeError("every YCbCr plane must be an AbstractTensor")
    if any(plane.ndims() != 2 for plane in planes):
        raise ValueError("every YCbCr plane must be two-dimensional")
    if any(plane.shape != planes[0].shape for plane in planes[1:]):
        raise ValueError("YCbCr planes must share one shape")
    height, width = _validate_frame(planes[0], color=False)
    yield _color_header(height, width)

    rows_per_batch = mcu_rows_per_batch * 8
    entropy = _EntropyTensorAccumulator()
    previous_dc = [0, 0, 0]
    resources = resources or prepare_jpeg_encoding_resources(planes[0])
    luma_dc_table = resources.luma_dc_table
    luma_ac_table = resources.luma_ac_table
    chroma_dc_table = resources.chroma_dc_table
    chroma_ac_table = resources.chroma_ac_table
    for row_start in range(0, height, rows_per_batch):
        row_stop = min(height, row_start + rows_per_batch)
        with autograd.no_grad():
            batches = tuple(plane[row_start:row_stop] for plane in planes)
            component_coefficients = jpeg_ycbcr_coefficients(
                batches,
                basis=resources.dct_basis,
                quantization=resources.ycbcr_quantization,
                zigzag=resources.zigzag,
            )
            combined_events = collect_component_block_coefficient_events(
                component_coefficients,
                max_magnitude_bits=11,
                previous_dc=previous_dc,
            )
            block_count = component_coefficients[0].reshape(-1, 64).shape[0]
            y_events = slice_block_coefficient_events(
                combined_events, 0, block_count
            )
            chroma_events = slice_block_coefficient_events(
                combined_events, block_count, block_count * 3
            )
            previous_dc = [
                component_coefficients[component].reshape(-1, 64)[-1, 0]
                for component in range(3)
            ]
            scan = encode_baseline_color_component_scan(
                y_events,
                chroma_events,
                luma_dc_table=luma_dc_table,
                luma_ac_table=luma_ac_table,
                chroma_dc_table=chroma_dc_table,
                chroma_ac_table=chroma_ac_table,
            )
            encoded = entropy.append(
                scan,
                final=row_start == 0 and row_stop == height,
            )
        if encoded:
            yield encoded
    tail = entropy.finish()
    if tail:
        yield tail
    yield b"\xFF\xD9"


def encode_ycbcr_jfif(
    planes,
    *,
    mcu_rows_per_batch: int = 8,
    resources: JPEGEncodingResources | None = None,
) -> bytes:
    """Encode three full-resolution AbstractTensor planes as 4:4:4 JFIF."""

    return b"".join(
        iter_ycbcr_jfif_chunks(
            planes,
            mcu_rows_per_batch=mcu_rows_per_batch,
            resources=resources,
        )
    )


def encode_grayscale_jfif(
    samples: AbstractTensor,
    *,
    mcu_rows_per_batch: int = 8,
    resources: JPEGEncodingResources | None = None,
) -> bytes:
    """Encode a two-dimensional sample tensor as a streaming baseline JFIF."""
    if not isinstance(samples, AbstractTensor) or samples.ndims() != 2:
        raise ValueError("grayscale JFIF input must be two-dimensional")
    return b"".join(
        iter_jfif_chunks(
            samples,
            mcu_rows_per_batch=mcu_rows_per_batch,
            resources=resources,
        )
    )


def encode_color_jfif(
    samples: AbstractTensor,
    *,
    mcu_rows_per_batch: int = 8,
    resources: JPEGEncodingResources | None = None,
) -> bytes:
    """Encode an RGB tensor as a streaming 4:4:4 baseline JFIF."""
    if (
        not isinstance(samples, AbstractTensor)
        or samples.ndims() != 3
        or samples.shape[-1] != 3
    ):
        raise ValueError("color JFIF input must have shape (height, width, 3)")
    return b"".join(
        iter_jfif_chunks(
            samples,
            mcu_rows_per_batch=mcu_rows_per_batch,
            resources=resources,
        )
    )


def encode_jfif(
    samples: AbstractTensor,
    *,
    mcu_rows_per_batch: int = 8,
    resources: JPEGEncodingResources | None = None,
) -> bytes:
    """Dispatch a grayscale or RGB tensor to the baseline JFIF encoder."""
    return b"".join(
        iter_jfif_chunks(
            samples,
            mcu_rows_per_batch=mcu_rows_per_batch,
            resources=resources,
        )
    )


def write_grayscale_jfif(
    path: str | Path,
    samples: AbstractTensor,
    **encoder_options,
) -> Path:
    """Encode and write one grayscale JFIF frame at the explicit I/O boundary."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(
        encode_grayscale_jfif(samples, **encoder_options)
    )
    return destination


def write_jfif(
    path: str | Path,
    samples: AbstractTensor,
    **encoder_options,
) -> Path:
    """Encode grayscale or RGB samples and write one complete JFIF image."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        for chunk in iter_jfif_chunks(samples, **encoder_options):
            output.write(chunk)
    return destination


__all__ = [
    "JPEGEncodingResources",
    "encode_color_jfif",
    "encode_grayscale_jfif",
    "encode_jfif",
    "encode_ycbcr_jfif",
    "iter_jfif_chunks",
    "iter_ycbcr_jfif_chunks",
    "prepare_jpeg_encoding_resources",
    "write_grayscale_jfif",
    "write_jfif",
]
