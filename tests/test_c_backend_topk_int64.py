"""Authored C and repository LLVM top-k keep exact signed int64 order."""

import ctypes
from pathlib import Path

from cffi import FFI
import pytest
from llvmlite import binding as llvm

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    C_SSA_OUTPUT_ARGUMENTS, LLVM_SSA_MODULE,
    c_backend_repository_ssa_reference, validate_translation_table,
)


@pytest.fixture(scope="module")
def native_kernels():
    root = Path(__file__).resolve().parents[1]
    directory = root / "src/common/tensors/accelerator_backends/c_backend"
    ffi = FFI()
    ffi.cdef(
        "void topk_int64_dim(const long long*, const int*, int, int, int, "
        "long long*, long long*);"
        "void topk_double_dim(const double*, const int*, int, int, int, "
        "double*, double*);"
    )
    c = ffi.verify(
        '#include "ctensor_ops.h"',
        sources=[str(directory / "ctensor_ops.c")],
        include_dirs=[str(directory)],
    )
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    module = llvm.parse_assembly(LLVM_SSA_MODULE)
    module.verify()
    target = llvm.Target.from_default_triple()
    engine = llvm.create_mcjit_compiler(module, target.create_target_machine())
    engine.finalize_object()
    address = engine.get_function_address("topk_int64_dim")
    assert address
    pointer64 = ctypes.POINTER(ctypes.c_int64)
    pointer32 = ctypes.POINTER(ctypes.c_int32)
    llvm_topk = ctypes.CFUNCTYPE(
        None, pointer64, pointer32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, pointer64, pointer64,
    )(address)
    return ffi, c, engine, llvm_topk


@pytest.mark.parametrize(
    ("data", "shape", "axis", "k", "expected_values", "expected_indices"),
    [
        ([1, 3, 2, 4, 6, 5], (2, 3), 0, 2,
         [4, 6, 5, 1, 3, 2], [1, 1, 1, 0, 0, 0]),
        ([1, 3, 2, 4, 6, 5], (2, 3), 1, 2,
         [3, 2, 6, 5], [1, 2, 1, 2]),
        ([2**53, 2**53 + 1, 2**53 + 2], (3,), 0, 2,
         [2**53 + 2, 2**53 + 1], [2, 1]),
        ([2**63 - 1, 2**63 - 1, -(2**63), 2**63 - 2], (4,), 0, 3,
         [2**63 - 1, 2**63 - 1, 2**63 - 2], [0, 1, 3]),
        ([-(2**63), -7, -3], (3,), 0, 3,
         [-3, -7, -(2**63)], [2, 1, 0]),
        ([-9, -2, -7, -2], (4,), 0, 4,
         [-2, -2, -7, -9], [1, 3, 2, 0]),
        ([-5, -5, -5, -5], (4,), 0, 4,
         [-5, -5, -5, -5], [0, 1, 2, 3]),
    ],
)
def test_topk_int64_c_and_llvm_match_exact_values_and_row_major_indices(
    native_kernels, data, shape, axis, k, expected_values, expected_indices,
):
    ffi, c, _engine, llvm_topk = native_kernels
    n = len(expected_values)
    c_input = ffi.new("long long[]", data)
    c_shape = ffi.new("int[]", shape)
    c_indices = ffi.new("long long[]", n)
    c_values = ffi.new("long long[]", n)
    c.topk_int64_dim(
        c_input, c_shape, len(shape), axis, k, c_indices, c_values,
    )
    assert list(c_values) == expected_values
    assert list(c_indices) == expected_indices

    llvm_input = (ctypes.c_int64 * len(data))(*data)
    llvm_shape = (ctypes.c_int32 * len(shape))(*shape)
    llvm_indices = (ctypes.c_int64 * n)()
    llvm_values = (ctypes.c_int64 * n)()
    llvm_topk(
        llvm_input, llvm_shape, len(shape), axis, k,
        llvm_indices, llvm_values,
    )
    assert list(llvm_values) == expected_values
    assert list(llvm_indices) == expected_indices


def test_topk_int64_zero_k_writes_no_output(native_kernels):
    ffi, c, _engine, llvm_topk = native_kernels
    c_input = ffi.new("long long[]", [-3, 2, -1])
    c_shape = ffi.new("int[]", [3])
    c_indices = ffi.new("long long[]", [101, 102])
    c_values = ffi.new("long long[]", [103, 104])
    c.topk_int64_dim(c_input, c_shape, 1, 0, 0, c_indices, c_values)
    assert list(c_indices) == [101, 102]
    assert list(c_values) == [103, 104]

    llvm_input = (ctypes.c_int64 * 3)(-3, 2, -1)
    llvm_shape = (ctypes.c_int32 * 1)(3)
    llvm_indices = (ctypes.c_int64 * 2)(101, 102)
    llvm_values = (ctypes.c_int64 * 2)(103, 104)
    llvm_topk(llvm_input, llvm_shape, 1, 0, 0, llvm_indices, llvm_values)
    assert list(llvm_indices) == [101, 102]
    assert list(llvm_values) == [103, 104]


def test_repository_reference_exposes_two_typed_outputs():
    validate_translation_table()
    assert C_SSA_OUTPUT_ARGUMENTS["topk_int64_dim"] == (
        ("values", 6), ("indices", 5),
    )
    reference = c_backend_repository_ssa_reference()
    assert reference.operation("topk").entrypoints == ("topk_int64_dim",)
    function = reference.module.functions["topk_int64_dim"]
    assert function.metadata["named_outputs"] == (
        ("values", function.args[6].id),
        ("indices", function.args[5].id),
    )
    assert function.args[0].dtype == "int64"
    assert function.args[1].dtype == "int32"
    assert function.args[5].dtype == function.args[6].dtype == "int64"


@pytest.mark.parametrize(
    ("axis", "count", "expected_values", "expected_indices"),
    [
        (0, 6, [4, 6, 5, 1, 3, 2], [1, 1, 1, 0, 0, 0]),
        (1, 4, [3, 2, 6, 5], [1, 2, 1, 2]),
    ],
)
def test_existing_double_topk_writes_row_major_axis_outputs(
    native_kernels, axis, count, expected_values, expected_indices,
):
    ffi, c, _engine, _llvm_topk = native_kernels
    data = ffi.new("double[]", [1, 3, 2, 4, 6, 5])
    shape = ffi.new("int[]", [2, 3])
    indices = ffi.new("double[]", count)
    values = ffi.new("double[]", count)
    c.topk_double_dim(data, shape, 2, axis, 2, indices, values)
    assert list(values) == expected_values
    assert list(indices) == expected_indices
