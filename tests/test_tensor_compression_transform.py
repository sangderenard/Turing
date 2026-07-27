import pytest

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.compression.block_transform import (
    block_view_2d,
    dct_2d_blocks,
    orthonormal_dct_basis,
)
from src.common.tensors.compression.jpeg.transform import (
    JPEG_CHROMA_QUANTIZATION,
    JPEG_LUMA_QUANTIZATION,
    JPEG_ZIGZAG,
    jpeg_chroma_coefficients,
    jpeg_luma_coefficients,
    jpeg_ycbcr_coefficients,
)


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_orthonormal_dct_maps_constant_block_to_dc(backend):
    with AT.use_backend(backend):
        block = AT.ones((1, 1, 8, 8))
        basis = orthonormal_dct_basis(8, like=block)
        coefficients = dct_2d_blocks(block, basis=basis)

    values = coefficients.tolist()[0][0]
    assert values[0][0] == pytest.approx(8.0, abs=2e-6)
    assert sum(abs(value) for row in values for value in row[1:]) < 2e-5
    assert sum(abs(row[0]) for row in values[1:]) < 2e-5


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_block_view_preserves_spatial_block_order(backend):
    source = [[row * 16 + col for col in range(16)] for row in range(8)]
    with AT.use_backend(backend):
        blocks = block_view_2d(
            AT.tensor(source), block_height=8, block_width=8
        )

    assert blocks.shape == (1, 2, 8, 8)
    assert blocks.tolist()[0][0][0] == list(range(8))
    assert blocks.tolist()[0][1][0] == list(range(8, 16))


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_jpeg_flat_field_has_only_a_dc_coefficient(backend):
    with AT.use_backend(backend):
        samples = AT.full((8, 8), 129)
        coefficients = jpeg_luma_coefficients(samples)

    assert coefficients.shape == (1, 1, 64)
    assert coefficients.tolist()[0][0][0] == 1
    assert coefficients.tolist()[0][0][1:] == [0] * 63


def test_zigzag_is_a_complete_permutation():
    assert sorted(JPEG_ZIGZAG) == list(range(64))


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_batched_ycbcr_transform_matches_three_component_transforms(backend):
    with AT.use_backend(backend):
        planes = (
            AT.full((8, 16), 137),
            AT.full((8, 16), 121),
            AT.full((8, 16), 149),
        )
        basis = orthonormal_dct_basis(8, like=planes[0])
        luma_quantization = planes[0].ensure_tensor(
            JPEG_LUMA_QUANTIZATION
        )
        chroma_quantization = planes[0].ensure_tensor(
            JPEG_CHROMA_QUANTIZATION
        )
        quantization = AT.stack(
            (
                luma_quantization,
                chroma_quantization,
                chroma_quantization,
            ),
            dim=0,
        ).reshape(3, 1, 1, 8, 8)
        zigzag = planes[0].ensure_tensor(JPEG_ZIGZAG).to_dtype("int64")
        batched = jpeg_ycbcr_coefficients(
            planes,
            basis=basis,
            quantization=quantization,
            zigzag=zigzag,
        )
        separate = (
            jpeg_luma_coefficients(
                planes[0],
                basis=basis,
                quantization=luma_quantization,
                zigzag=zigzag,
            ),
            jpeg_chroma_coefficients(
                planes[1],
                basis=basis,
                quantization=chroma_quantization,
                zigzag=zigzag,
            ),
            jpeg_chroma_coefficients(
                planes[2],
                basis=basis,
                quantization=chroma_quantization,
                zigzag=zigzag,
            ),
        )

    assert batched.shape == (3, 1, 2, 64)
    assert batched.tolist() == [
        component.tolist() for component in separate
    ]
