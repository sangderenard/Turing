"""src/common/tensors/compression/tensor_payload.py: a tensor as a compressed payload, and back, losslessly."""
from __future__ import annotations

import math

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.compression import tensor_payload


def _same(left, right):
    assert tensor_payload.dtype_name(left) == tensor_payload.dtype_name(right)
    assert tuple(left.get_shape()) == tuple(right.get_shape())
    a, b = left.tolist(), right.tolist()

    def flat(x):
        return [z for y in x for z in flat(y)] if isinstance(x, list) else [x]

    for x, y in zip(flat(a), flat(b), strict=True):
        assert (math.isnan(x) and math.isnan(y)) if isinstance(x, float) and math.isnan(x) else x == y


@pytest.mark.parametrize("dtype", ["float32", "float64", "int64", "int32", "uint8", "bool"])
def test_a_tensor_comes_back_with_its_dtype_shape_and_values(dtype):
    values = [[[0.1 * (i + j + k) - 0.7 for k in range(4)] for j in range(3)] for i in range(2)]
    if dtype == "bool":
        values = [[[(i + j + k) % 2 == 0 for k in range(4)] for j in range(3)] for i in range(2)]
    elif dtype.startswith(("int", "uint")):
        values = [[[(i * 12 + j * 4 + k) % 200 for k in range(4)] for j in range(3)] for i in range(2)]
    tensor = AbstractTensor.tensor(values, dtype=dtype)
    _same(tensor_payload.loads(tensor_payload.dumps(tensor)), tensor)


def test_float_bits_are_kept_exactly_including_nan_and_signed_zero():
    tensor = AbstractTensor.tensor([1.0 / 3.0, -0.0, float("nan"), float("inf"), 5e-324], dtype="float64")
    back = tensor_payload.loads(tensor_payload.dumps(tensor)).tolist()
    assert back[0] == 1.0 / 3.0 and math.copysign(1.0, back[1]) == -1.0 and math.isnan(back[2])
    assert back[3] == float("inf") and back[4] == 5e-324


def test_a_repetitive_map_compresses():
    tensor = AbstractTensor.tensor([[float(i % 7) for i in range(64)] for _ in range(64)], dtype="float32")
    payload = tensor_payload.dumps(tensor)
    assert len(payload) < 64 * 64 * 4 / 4
    _same(tensor_payload.loads(payload), tensor)


def test_the_payload_is_the_handle_and_the_raw_region():
    import numpy as np
    from src.compiler import tile_row_codec

    region = (np.arange(4096 * 12, dtype=np.float64).reshape(4096, 4, 3) % 7) * 0.25
    payload = tensor_payload.dumps(region)
    handle, packed = tile_row_codec.decode_rows(payload[len(tensor_payload.MAGIC):])
    assert handle == ("float64", (4096, 4, 3))
    assert tile_row_codec.decompress_octets(packed) == region.astype("<f8").tobytes()
    assert len(payload) < region.nbytes / 8
    back = tensor_payload.loads(payload)
    assert tuple(back.get_shape()) == (4096, 4, 3) and (np.asarray(back.tolist()) == region).all()


def test_a_numpy_region_and_a_strided_view_come_back_in_c_order():
    import numpy as np

    base = np.arange(24, dtype=np.int32).reshape(4, 6)
    view = base[:, ::2].T                       # not contiguous
    back = tensor_payload.loads(tensor_payload.dumps(view))
    assert tensor_payload.dtype_name(back) == "int32" and back.tolist() == view.tolist()


def test_save_and_load_through_a_file(tmp_path):
    tensor = AbstractTensor.tensor([[1, 2, 3], [4, 5, 6]], dtype="int64")
    target = tmp_path / "triangle_map.tp"
    written = tensor_payload.save(target, tensor)
    assert written == target.stat().st_size and not (tmp_path / "triangle_map.tp.partial").exists()
    _same(tensor_payload.load(target), tensor)


def test_what_is_not_a_tensor_is_refused_before_writing(tmp_path):
    with pytest.raises(TypeError):
        tensor_payload.save(tmp_path / "x", {"not": "a tensor"})
    assert not (tmp_path / "x").exists() and not (tmp_path / "x.partial").exists()
    with pytest.raises(ValueError):
        tensor_payload.loads(b"pickle bytes, not a payload")
