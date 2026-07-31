from io import BytesIO

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.compression.jpeg.frame import (
    encode_color_jfif,
    encode_grayscale_jfif,
    encode_jfif,
    encode_jfif_resident,
)


def test_resident_jfif_packet_matches_host_serialization():
    samples = np.arange(8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)
    with AT.use_backend("numpy"):
        tensor = AT.tensor(samples)
        resident = encode_jfif_resident(tensor)
        expected = encode_jfif(tensor)

    assert resident.to_bytes() == expected


def test_glsl_resident_jfif_packet_matches_numpy():
    from src.common.tensors.accelerator_backends.glsl_backend import (
        GLContextUnavailable,
        require_gl_context,
    )

    try:
        require_gl_context()
    except GLContextUnavailable as exc:
        pytest.skip(f"no OpenGL 4.3+ compute context: {exc}")

    samples = np.arange(8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)
    with AT.use_backend("numpy"):
        expected = encode_jfif(AT.tensor(samples))
    with AT.use_backend("glsl"):
        resident = encode_jfif_resident(AT.tensor(samples))

    assert resident.to_bytes() == expected


from src.common.tensors.compression.jpeg.huffman import (
    JPEG_AC_LUMINANCE_SYMBOLS,
    jpeg_standard_ac_luminance,
)


def _pattern(height, width):
    return [
        [
            int((row * 3 + column * 5 + 40 * ((row // 8 + column // 8) % 2)) % 256)
            for column in range(width)
        ]
        for row in range(height)
    ]


def _color_pattern(height, width):
    colors = (
        (240, 24, 32),
        (24, 232, 48),
        (32, 48, 240),
        (236, 220, 32),
    )
    return [
        [
            colors[
                2 * (row >= height // 2)
                + (column >= width // 2)
            ]
            for column in range(width)
        ]
        for row in range(height)
    ]


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_grayscale_jfif_is_identical_and_independently_readable(backend):
    image_module = pytest.importorskip("PIL.Image")
    with AT.use_backend(backend):
        encoded = encode_grayscale_jfif(AT.tensor(_pattern(24, 32)))

    assert encoded.startswith(b"\xFF\xD8\xFF\xE0")
    assert encoded.endswith(b"\xFF\xD9")
    image = image_module.open(BytesIO(encoded))
    image.load()
    assert image.format == "JPEG"
    assert image.mode == "L"
    assert image.size == (32, 24)
    assert min(image.getdata()) < max(image.getdata())


def test_standard_ac_table_preserves_transmitted_symbol_order():
    with AT.use_backend("numpy"):
        table = jpeg_standard_ac_luminance(AT.tensor([0]))
        first_symbols = AT.tensor(list(JPEG_AC_LUMINANCE_SYMBOLS[:6]))
        codes, lengths = table.lookup(first_symbols)

    assert lengths.tolist() == [2, 2, 3, 4, 4, 4]
    assert codes.tolist() == [0b00, 0b01, 0b100, 0b1010, 0b1011, 0b1100]


def test_initial_jfif_encoder_states_alignment_requirement():
    with AT.use_backend("numpy"):
        with pytest.raises(ValueError, match="divisible by 8"):
            encode_grayscale_jfif(AT.zeros((9, 8)))


def test_abstract_tensor_jpg_redirect_returns_bytes_and_writes(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    with AT.use_backend("numpy"):
        image = AT.tensor(_pattern(16, 24))
        encoded = image.jpg()
        destination = image.jpg(path=tmp_path / "tensor.jpg")

    assert destination.read_bytes() == encoded
    decoded = image_module.open(BytesIO(encoded))
    decoded.load()
    assert decoded.mode == "L"
    assert decoded.size == (24, 16)


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_color_jfif_is_real_444_ycbcr_and_independently_readable(backend):
    image_module = pytest.importorskip("PIL.Image")
    with AT.use_backend(backend):
        encoded = encode_color_jfif(AT.tensor(_color_pattern(16, 16)))

    sof = encoded.index(b"\xFF\xC0")
    sof_length = int.from_bytes(encoded[sof + 2:sof + 4], "big")
    sof_payload = encoded[sof + 4:sof + 2 + sof_length]
    assert sof_payload[5] == 3
    assert sof_payload[6:] == (
        b"\x01\x11\x00"
        b"\x02\x11\x01"
        b"\x03\x11\x01"
    )

    image = image_module.open(BytesIO(encoded))
    image.load()
    assert image.format == "JPEG"
    assert image.mode == "RGB"
    assert image.size == (16, 16)
    expected = _color_pattern(16, 16)
    for row, column in ((3, 3), (3, 12), (12, 3), (12, 12)):
        actual = image.getpixel((column, row))
        target = expected[row][column]
        assert max(abs(a - b) for a, b in zip(actual, target)) < 24


def test_abstract_tensor_jpg_redirect_dispatches_rgb():
    image_module = pytest.importorskip("PIL.Image")
    with AT.use_backend("numpy"):
        encoded = AT.tensor(_color_pattern(16, 16)).jpg()
    image = image_module.open(BytesIO(encoded))
    image.load()
    assert image.mode == "RGB"


@pytest.mark.parametrize("color", [False, True])
def test_streaming_mcu_batches_are_byte_identical(color):
    samples = _color_pattern(32, 24) if color else _pattern(32, 24)
    with AT.use_backend("numpy"):
        tensor = AT.tensor(samples)
        one_row = encode_jfif(tensor, mcu_rows_per_batch=1)
        whole_frame = encode_jfif(tensor, mcu_rows_per_batch=32)

    assert one_row == whole_frame
