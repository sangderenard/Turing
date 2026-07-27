"""JPEG coefficient preparation through AbstractTensor composition."""

from __future__ import annotations

from ...abstraction import AbstractTensor
from ..block_transform import block_view_2d, dct_2d_blocks


JPEG_ZIGZAG = (
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
)

JPEG_LUMA_QUANTIZATION = (
    (16, 11, 10, 16, 24, 40, 51, 61),
    (12, 12, 14, 19, 26, 58, 60, 55),
    (14, 13, 16, 24, 40, 57, 69, 56),
    (14, 17, 22, 29, 51, 87, 80, 62),
    (18, 22, 37, 56, 68, 109, 103, 77),
    (24, 35, 55, 64, 81, 104, 113, 92),
    (49, 64, 78, 87, 103, 121, 120, 101),
    (72, 92, 95, 98, 112, 100, 103, 99),
)

JPEG_CHROMA_QUANTIZATION = (
    (17, 18, 24, 47, 99, 99, 99, 99),
    (18, 21, 26, 66, 99, 99, 99, 99),
    (24, 26, 56, 99, 99, 99, 99, 99),
    (47, 66, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
)


def jpeg_component_coefficients(
    samples: AbstractTensor,
    quantization,
) -> AbstractTensor:
    """Return zigzag-ordered coefficients for one JPEG sample component."""
    if not isinstance(samples, AbstractTensor):
        raise TypeError("samples must be an AbstractTensor")
    blocks = block_view_2d(
        samples - 128.0, block_height=8, block_width=8
    )
    transformed = dct_2d_blocks(blocks)
    quantization_tensor = samples.ensure_tensor(quantization)
    scaled = transformed / quantization_tensor
    # The orthonormal cosine basis is evaluated in backend precision. A tiny
    # format-level tie tolerance keeps mathematical half-integers stable across
    # float32, float64, and the C backend's double storage.
    rounded = scaled.sign() * ((scaled.abs() + 0.500001) // 1)
    flattened = rounded.reshape(*(rounded.shape[:-2] + (64,)))
    zigzag = samples.ensure_tensor(JPEG_ZIGZAG).to_dtype("int64")
    return flattened[..., zigzag]


def jpeg_luma_coefficients(samples: AbstractTensor) -> AbstractTensor:
    """Return zigzag-ordered quantized coefficients for a luma plane."""
    return jpeg_component_coefficients(samples, JPEG_LUMA_QUANTIZATION)


def jpeg_chroma_coefficients(samples: AbstractTensor) -> AbstractTensor:
    """Return zigzag-ordered quantized coefficients for a chroma plane."""
    return jpeg_component_coefficients(samples, JPEG_CHROMA_QUANTIZATION)


def rgb_to_ycbcr(samples: AbstractTensor) -> tuple[
    AbstractTensor,
    AbstractTensor,
    AbstractTensor,
]:
    """Convert an RGB tensor to full-resolution JPEG Y, Cb, and Cr planes."""
    if not isinstance(samples, AbstractTensor):
        raise TypeError("samples must be an AbstractTensor")
    if samples.ndims() != 3 or samples.shape[-1] != 3:
        raise ValueError("RGB input must have shape (height, width, 3)")
    red = samples[..., 0]
    green = samples[..., 1]
    blue = samples[..., 2]
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    blue_difference = (
        -0.168736 * red - 0.331264 * green + 0.5 * blue + 128.0
    )
    red_difference = (
        0.5 * red - 0.418688 * green - 0.081312 * blue + 128.0
    )
    return (
        luminance.clamp(0, 255),
        blue_difference.clamp(0, 255),
        red_difference.clamp(0, 255),
    )


__all__ = [
    "JPEG_CHROMA_QUANTIZATION",
    "JPEG_LUMA_QUANTIZATION",
    "JPEG_ZIGZAG",
    "jpeg_chroma_coefficients",
    "jpeg_component_coefficients",
    "jpeg_luma_coefficients",
    "rgb_to_ycbcr",
]
