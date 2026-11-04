"""Native scalar-first arithmetic after a validated runtime reshape."""

import numpy as np
import pytest

from src.compiler.native_faults import NativeIntOverflow, NativeValueError
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from tests.test_dynamic_reshape_geometry import _lower


def _feeds(module, root, values, width):
    parameters = dict(module.functions[root].metadata["parameter_names"])
    return {
        parameters["bits"]: np.asarray(values, dtype=np.int64).reshape(2, 4),
        parameters["width"]: np.array([width], dtype=np.int64),
    }


def _sole_output(run, feeds):
    outputs = [value for key, value in run.buffers.items() if key not in feeds]
    assert len(outputs) == 1
    return outputs[0]


def test_scalar_first_runtime_reshape_native_signed_lanes_and_faults(tmp_path):
    source = """def root(bits: AbstractTensor, width: int) -> AbstractTensor:
    shaped = bits.reshape(-1, width)
    return 1 - shaped
"""
    module, _, exports = _lower(source)
    root = exports[0]
    artifact = emit_ssa_module_to_c(module, root)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "scalar_first_reshape")

    # Public output geometry for width 2 is still reported as the input's
    # 2x4 shape. This test checks the exact signed lanes, independently of
    # that separate runtime shape-output contract.
    for start in (0, 2**53 + 1):
        values = np.arange(start, start + 8, dtype=np.int64).reshape(2, 4)
        expected = np.array([1 - int(value) for value in values.flat],
                            dtype=np.int64)
        for width in (4, 2):
            feeds = _feeds(module, root, values, width)
            output = _sole_output(artifact.prepare_execution(feeds).run(), feeds)
            assert output.dtype == np.dtype("int64")
            np.testing.assert_array_equal(output.ravel(), expected)

    values = np.arange(8, dtype=np.int64).reshape(2, 4)
    for width in (0, -2, 3):
        with pytest.raises(NativeValueError, match="code 98"):
            artifact.prepare_execution(_feeds(module, root, values, width)).run()
    with pytest.raises(NativeIntOverflow, match="code 86"):
        artifact.prepare_execution(_feeds(module, root, values, 2**31)).run()


def test_runtime_reshape_private_scalar_reduction_native(tmp_path):
    source = """def root(bits: AbstractTensor, width: int):
    shaped = bits.reshape(-1, width)
    return (1 - shaped).sum().item()
"""
    module, _, exports = _lower(source)
    root = exports[0]
    artifact = emit_ssa_module_to_c(module, root)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "reshape_private_reduction")
    values = np.arange(8, dtype=np.int64).reshape(2, 4)
    for width in (4, 2):
        feeds = _feeds(module, root, values, width)
        output = _sole_output(artifact.prepare_execution(feeds).run(), feeds)
        assert output.item() == -20
