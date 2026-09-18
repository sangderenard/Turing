from __future__ import annotations

import ctypes
import re

import numpy as np
import pytest
from llvmlite import binding as llvm

from src.common.tensors.accelerator_backends.c_backend import C, ffi
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    LLVM_SSA_MODULE,
    PRECOMPILE_INTERNAL_OPERATORS,
    TRANSLATIONS,
    covered_operations,
    extract_c_function,
    extract_llvm_declaration,
    extract_llvm_function,
    lower_abstract_tensor_tape_to_llvm_ssa,
    translations_for_operation,
    validate_translation_table,
)
from src.common.tensors.abstraction import tensor_identity
from src.common.tensors.autograd import autograd
from src.common.tensors.numpy_backend import NumPyTensorOperations
from src.common.tensors.operator_catalog import (
    CANONICAL_ABSTRACT_TENSOR_OPERATORS,
)


_REAL_TRANSLATED_SYMBOLS = {
    "fill_double",
    "binary_value",
    "binary_double",
    "binary_scalar_double",
    "unary_double",
    "matmul_double",
    "batched_matmul_indexed_double",
    "where_double",
    "broadcast_double",
    "reduce_dim_double",
    "transpose_double",
    "cumsum_dim_double",
    "stack_double",
    "cat_double",
    "pad_double_nd",
    "slice_copy_double",
    "index_select_double",
    "gather_values_double",
    "index_assign_double",
    "index_set_double",
    "unfold2d_double",
    "fold2d_double",
    "sign_double",
    "count_true_double",
    "mask_select_double",
    "increment_mask_double",
    # One kernel per value-precision cast, not one shared narrowing kernel:
    # ``double`` is a copying identity under the double working type and
    # ``bool`` is a zero test, neither of which is the float32 rounding
    # ``float`` performs.  All four are real definitions in both languages --
    # this list is the independent third party that says so, so it is written
    # out rather than derived from TRANSLATIONS.
    "cast_double_to_int_values",
    "cast_double_to_float_values",
    "cast_double_to_bool_values",
    "cast_double_to_double_values",
    "sum_double",
    "create_arange",
}


@pytest.fixture(scope="module")
def llvm_engine():
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    module = llvm.parse_assembly(LLVM_SSA_MODULE)
    module.verify()
    target = llvm.Target.from_default_triple()
    engine = llvm.create_mcjit_compiler(module, target.create_target_machine())
    engine.finalize_object()
    return engine


def _llvm_function(engine, name, *argument_types, result_type=None):
    address = engine.get_function_address(name)
    assert address
    return ctypes.CFUNCTYPE(result_type, *argument_types)(address)


def _c_array(values):
    return ffi.new("double[]", [float(value) for value in values])


def test_translation_table_uses_only_real_c_function_definitions():
    validate_translation_table()

    defined = set(
        re.findall(
            r"(?m)^define(?:\s+internal)?\s+(?:void|double|i32)\s+@([A-Za-z0-9_]+)",
            LLVM_SSA_MODULE,
        )
    )
    assert defined == _REAL_TRANSLATED_SYMBOLS
    assert {entry.c_symbol for entry in TRANSLATIONS} == _REAL_TRANSLATED_SYMBOLS
    assert {entry.llvm_symbol for entry in TRANSLATIONS} == _REAL_TRANSLATED_SYMBOLS

    for symbol in _REAL_TRANSLATED_SYMBOLS:
        assert re.search(rf"\b{symbol}\s*\(", extract_c_function(symbol))
        assert re.search(rf"@{symbol}\s*\(", extract_llvm_function(symbol))


def test_external_llvm_declarations_are_extracted_from_canonical_module():
    assert extract_llvm_declaration("acos") == "declare double @acos(double)"
    assert extract_llvm_declaration("llvm.memset.p0.i64").startswith(
        "declare void @llvm.memset.p0.i64("
    )


def test_translation_operations_are_real_abstract_tensor_names():
    assert (
        covered_operations() - PRECOMPILE_INTERNAL_OPERATORS
        <= CANONICAL_ABSTRACT_TENSOR_OPERATORS
    )
    assert [entry.c_symbol for entry in translations_for_operation("matmul")] == [
        "matmul_double",
        "batched_matmul_indexed_double",
    ]


def test_handwritten_fill_ssa_matches_real_c_kernel(llvm_engine):
    values = [-3.5] * 7
    c_output = _c_array([0.0] * len(values))
    C.fill_double(c_output, -3.5, len(values))

    llvm_output = (ctypes.c_double * len(values))()
    function = _llvm_function(
        llvm_engine,
        "fill_double",
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.c_int32,
    )
    function(llvm_output, -3.5, len(values))

    assert list(llvm_output) == [c_output[index] for index in range(len(values))]


@pytest.mark.parametrize(
    ("c_opcode", "llvm_opcode"),
    [
        ("CT_OP_ADD", 0),
        ("CT_OP_SUB", 1),
        ("CT_OP_MUL", 2),
        ("CT_OP_DIV", 3),
        ("CT_OP_LT", 20),
        ("CT_OP_NE", 25),
        ("CT_OP_MAXIMUM", 26),
        ("CT_OP_MINIMUM", 27),
    ],
)
def test_handwritten_binary_ssa_matches_real_c_kernel(
    llvm_engine,
    c_opcode,
    llvm_opcode,
):
    left = [1.5, -2.0, 0.0, 8.0]
    right = [2.0, 4.0, -1.0, 2.0]
    c_left = _c_array(left)
    c_right = _c_array(right)
    c_output = _c_array([0.0] * len(left))
    C.binary_double(
        c_left,
        c_right,
        c_output,
        len(left),
        getattr(C, c_opcode),
    )

    llvm_left = (ctypes.c_double * len(left))(*left)
    llvm_right = (ctypes.c_double * len(right))(*right)
    llvm_output = (ctypes.c_double * len(left))()
    function = _llvm_function(
        llvm_engine,
        "binary_double",
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.c_int32,
    )
    function(llvm_left, llvm_right, llvm_output, len(left), llvm_opcode)

    np.testing.assert_equal(
        np.asarray(llvm_output),
        np.asarray([c_output[index] for index in range(len(left))]),
    )


@pytest.mark.parametrize(
    ("c_opcode", "llvm_opcode"),
    [
        ("CT_OP_BITAND", 42),
        ("CT_OP_BITOR", 43),
        ("CT_OP_BITXOR", 44),
        ("CT_OP_SHL", 45),
        ("CT_OP_SHR", 46),
        ("CT_OP_LOGICAL_AND", 47),
        ("CT_OP_LOGICAL_OR", 48),
    ],
)
def test_handwritten_integer_and_logical_binary_ssa_matches_c(
    llvm_engine, c_opcode, llvm_opcode,
):
    left = [7.0, 12.0, 8.0, 0.0]
    right = [3.0, 5.0, 1.0, 2.0]
    c_left = _c_array(left)
    c_right = _c_array(right)
    c_output = _c_array([0.0] * len(left))
    C.binary_double(
        c_left, c_right, c_output, len(left), getattr(C, c_opcode)
    )
    llvm_output = (ctypes.c_double * len(left))()
    function = _llvm_function(
        llvm_engine, "binary_double",
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.c_int32, ctypes.c_int32,
    )
    function(
        (ctypes.c_double * len(left))(*left),
        (ctypes.c_double * len(right))(*right),
        llvm_output, len(left), llvm_opcode,
    )
    np.testing.assert_equal(
        np.asarray(llvm_output),
        np.asarray([c_output[index] for index in range(len(left))]),
    )


@pytest.mark.parametrize(
    ("c_opcode", "llvm_opcode"),
    [
        ("CT_OP_NEG", 10),
        ("CT_OP_ABS", 11),
        ("CT_OP_ISFINITE", 16),
        ("CT_OP_ISNAN", 17),
        ("CT_OP_ISINF", 18),
        ("CT_OP_LOGICAL_NOT", 19),
    ],
)
def test_handwritten_unary_ssa_matches_real_c_kernel(
    llvm_engine,
    c_opcode,
    llvm_opcode,
):
    values = [float("-inf"), -2.0, -0.0, 1.5, float("inf"), float("nan")]
    c_input = _c_array(values)
    c_output = _c_array([0.0] * len(values))
    C.unary_double(c_input, c_output, len(values), getattr(C, c_opcode))

    llvm_input = (ctypes.c_double * len(values))(*values)
    llvm_output = (ctypes.c_double * len(values))()
    function = _llvm_function(
        llvm_engine,
        "unary_double",
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.c_int32,
    )
    function(llvm_input, llvm_output, len(values), llvm_opcode)

    np.testing.assert_equal(
        np.asarray(llvm_output),
        np.asarray([c_output[index] for index in range(len(values))]),
    )


@pytest.mark.parametrize(
    ("c_opcode", "llvm_opcode"),
    [("CT_OP_SIGN", 40), ("CT_OP_INVERT", 41)],
)
def test_handwritten_sign_and_invert_ssa_match_c(
    llvm_engine, c_opcode, llvm_opcode,
):
    values = [-8.0, -1.0, 0.0, 1.0, 7.0]
    c_input = _c_array(values)
    c_output = _c_array([0.0] * len(values))
    C.unary_double(c_input, c_output, len(values), getattr(C, c_opcode))
    llvm_output = (ctypes.c_double * len(values))()
    function = _llvm_function(
        llvm_engine, "unary_double",
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32, ctypes.c_int32,
    )
    function(
        (ctypes.c_double * len(values))(*values), llvm_output,
        len(values), llvm_opcode,
    )
    np.testing.assert_equal(
        np.asarray(llvm_output),
        np.asarray([c_output[index] for index in range(len(values))]),
    )


def test_handwritten_matmul_ssa_matches_real_c_kernel(llvm_engine):
    left = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    right = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    c_left = _c_array(left)
    c_right = _c_array(right)
    c_output = _c_array([0.0] * 4)
    C.matmul_double(c_left, c_right, c_output, 2, 3, 2)

    llvm_left = (ctypes.c_double * len(left))(*left)
    llvm_right = (ctypes.c_double * len(right))(*right)
    llvm_output = (ctypes.c_double * 4)()
    function = _llvm_function(
        llvm_engine,
        "matmul_double",
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    )
    function(llvm_left, llvm_right, llvm_output, 2, 3, 2)

    np.testing.assert_allclose(
        np.asarray(llvm_output),
        np.asarray([c_output[index] for index in range(4)]),
    )


def test_matmul_inner_product_compensates_cancellation(llvm_engine):
    left = [1.0e16, 1.0, 1.0, -1.0e16]
    right = [1.0, 1.0, 1.0, 1.0]
    c_output = _c_array([0.0])
    C.matmul_double(_c_array(left), _c_array(right), c_output, 1, 4, 1)

    llvm_output = (ctypes.c_double * 1)()
    function = _llvm_function(
        llvm_engine,
        "matmul_double",
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    )
    function(
        (ctypes.c_double * len(left))(*left),
        (ctypes.c_double * len(right))(*right),
        llvm_output, 1, 4, 1,
    )

    assert c_output[0] == 2.0
    assert llvm_output[0] == 2.0


def test_handwritten_where_ssa_matches_real_c_kernel(llvm_engine):
    condition = [0.0, 1.0, -2.0, float("nan")]
    left = [1.0, 2.0, 3.0, 4.0]
    right = [5.0, 6.0, 7.0, 8.0]
    c_output = _c_array([0.0] * 4)
    C.where_double(
        _c_array(condition),
        _c_array(left),
        _c_array(right),
        c_output,
        4,
    )

    llvm_output = (ctypes.c_double * 4)()
    function = _llvm_function(
        llvm_engine,
        "where_double",
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
    )
    function(
        (ctypes.c_double * 4)(*condition),
        (ctypes.c_double * 4)(*left),
        (ctypes.c_double * 4)(*right),
        llvm_output,
        4,
    )
    np.testing.assert_equal(
        np.asarray(llvm_output),
        np.asarray([c_output[index] for index in range(4)]),
    )


def test_handwritten_broadcast_ssa_matches_real_c_kernel(llvm_engine):
    values = [1.0, 2.0, 3.0]
    input_shape = [1, 3]
    output_shape = [2, 3]
    c_output = _c_array([0.0] * 6)
    C.broadcast_double(
        _c_array(values),
        c_output,
        ffi.new("int[]", input_shape),
        len(input_shape),
        ffi.new("int[]", output_shape),
        len(output_shape),
    )

    llvm_output = (ctypes.c_double * 6)()
    _llvm_function(
        llvm_engine,
        "broadcast_double",
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
    )(
        (ctypes.c_double * len(values))(*values),
        llvm_output,
        (ctypes.c_int32 * len(input_shape))(*input_shape),
        len(input_shape),
        (ctypes.c_int32 * len(output_shape))(*output_shape),
        len(output_shape),
    )
    assert list(llvm_output) == [c_output[index] for index in range(6)]


def test_handwritten_unfold_fold_ssa_match_reference_and_are_adjoint(llvm_engine):
    dimensions = (1, 1, 3, 4, 2, 2, 1, 1, 1, 1, 1, 1)
    n, channels, height, width, kh, kw, sh, sw, ph, pw, dh, dw = dimensions
    output_h = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    output_w = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    input_values = np.linspace(-0.7, 1.1, n * channels * height * width)
    column_values = np.linspace(
        0.9, -0.4,
        n * channels * kh * kw * output_h * output_w,
    )

    image = input_values.reshape(n, channels, height, width)
    padded = np.pad(image, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    expected_unfolded = np.empty(
        (n, channels, kh, kw, output_h, output_w), dtype=np.float64,
    )
    for kernel_row in range(kh):
        for kernel_column in range(kw):
            expected_unfolded[:, :, kernel_row, kernel_column] = padded[
                :,
                :,
                kernel_row * dh:kernel_row * dh + sh * output_h:sh,
                kernel_column * dw:kernel_column * dw + sw * output_w:sw,
            ]
    llvm_unfolded = (ctypes.c_double * column_values.size)()
    integer_arguments = (ctypes.c_int32,) * len(dimensions)
    _llvm_function(
        llvm_engine,
        "unfold2d_double",
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        *integer_arguments,
    )(
        (ctypes.c_double * input_values.size)(*input_values),
        llvm_unfolded,
        *dimensions,
    )
    np.testing.assert_allclose(
        np.asarray(llvm_unfolded),
        expected_unfolded.reshape(-1),
    )

    expected_folded_padded = np.zeros_like(padded)
    columns = column_values.reshape(
        n, channels, kh, kw, output_h, output_w,
    )
    for kernel_row in range(kh):
        for kernel_column in range(kw):
            expected_folded_padded[
                :,
                :,
                kernel_row * dh:kernel_row * dh + sh * output_h:sh,
                kernel_column * dw:kernel_column * dw + sw * output_w:sw,
            ] += columns[:, :, kernel_row, kernel_column]
    expected_folded = expected_folded_padded[
        :, :, ph:ph + height, pw:pw + width,
    ].reshape(-1)
    llvm_folded = (ctypes.c_double * input_values.size)()
    _llvm_function(
        llvm_engine,
        "fold2d_double",
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        *integer_arguments,
    )(
        (ctypes.c_double * column_values.size)(*column_values),
        llvm_folded,
        *dimensions,
    )
    np.testing.assert_allclose(
        np.asarray(llvm_folded),
        expected_folded,
    )
    np.testing.assert_allclose(
        np.dot(np.asarray(llvm_unfolded), column_values),
        np.dot(input_values, np.asarray(llvm_folded)),
        rtol=1e-13,
        atol=1e-13,
    )


def test_handwritten_slice_ssa_matches_real_c_kernel(llvm_engine):
    values = list(range(2 * 5 * 3))
    shape = [2, 5, 3]
    output_count = 2 * 2 * 3
    c_output = _c_array([0.0] * output_count)
    C.slice_copy_double(
        _c_array(values),
        c_output,
        ffi.new("int[]", shape),
        3,
        1,
        1,
        2,
        2,
    )

    llvm_output = (ctypes.c_double * output_count)()
    _llvm_function(
        llvm_engine,
        "slice_copy_double",
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    )(
        (ctypes.c_double * len(values))(*values),
        llvm_output,
        (ctypes.c_int32 * 3)(*shape),
        3,
        1,
        1,
        2,
        2,
    )
    assert list(llvm_output) == [
        c_output[index] for index in range(output_count)
    ]


def test_handwritten_index_select_ssa_matches_real_c_kernel(llvm_engine):
    values = list(range(2 * 4 * 2))
    shape = [2, 4, 2]
    indices = [3, 1]
    output_count = 2 * len(indices) * 2
    c_output = _c_array([0.0] * output_count)
    C.index_select_double(
        _c_array(values),
        c_output,
        ffi.new("int[]", shape),
        3,
        1,
        ffi.new("int[]", indices),
        len(indices),
    )

    llvm_output = (ctypes.c_double * output_count)()
    _llvm_function(
        llvm_engine,
        "index_select_double",
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
    )(
        (ctypes.c_double * len(values))(*values),
        llvm_output,
        (ctypes.c_int32 * 3)(*shape),
        3,
        1,
        (ctypes.c_int32 * len(indices))(*indices),
        len(indices),
    )
    assert list(llvm_output) == [
        c_output[index] for index in range(output_count)
    ]


@pytest.mark.parametrize("value_count", [1, 4])
def test_handwritten_index_assign_ssa_matches_real_c_kernel(
    llvm_engine,
    value_count,
):
    target = list(range(3 * 4))
    shape = [3, 4]
    offsets = [0, 2, 4]
    indices = [0, 2, 1, 3]
    values = [99.0] if value_count == 1 else [10.0, 20.0, 30.0, 40.0]
    c_target = _c_array(target)
    C.index_assign_double(
        c_target,
        ffi.new("int[]", shape),
        2,
        ffi.new("int[]", offsets),
        ffi.new("int[]", indices),
        _c_array(values),
        value_count,
    )

    llvm_target = (ctypes.c_double * len(target))(*target)
    _llvm_function(
        llvm_engine,
        "index_assign_double",
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
    )(
        llvm_target,
        (ctypes.c_int32 * 2)(*shape),
        2,
        (ctypes.c_int32 * 3)(*offsets),
        (ctypes.c_int32 * 4)(*indices),
        (ctypes.c_double * len(values))(*values),
        value_count,
    )
    assert list(llvm_target) == [
        c_target[index] for index in range(len(target))
    ]


def test_handwritten_sign_ssa_matches_real_c_kernel(llvm_engine):
    values = [float("nan"), -3.0, -0.0, 0.0, 4.0]
    c_output = _c_array([0.0] * len(values))
    C.sign_double(_c_array(values), c_output, len(values))

    llvm_output = (ctypes.c_double * len(values))()
    _llvm_function(
        llvm_engine,
        "sign_double",
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
    )((ctypes.c_double * len(values))(*values), llvm_output, len(values))
    np.testing.assert_equal(
        np.asarray(llvm_output),
        np.asarray([c_output[index] for index in range(len(values))]),
    )


def test_handwritten_mask_ssa_matches_real_c_kernels(llvm_engine):
    values = [10.0, 20.0, 30.0, 40.0]
    mask = [0.0, 1.0, 0.0, -1.0]

    c_count = C.count_true_double(_c_array(mask), 4)
    llvm_count = _llvm_function(
        llvm_engine,
        "count_true_double",
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        result_type=ctypes.c_int32,
    )((ctypes.c_double * 4)(*mask), 4)
    assert llvm_count == c_count == 2

    c_selected = _c_array([0.0] * c_count)
    C.mask_select_double(_c_array(values), _c_array(mask), c_selected, 4)
    llvm_selected = (ctypes.c_double * llvm_count)()
    _llvm_function(
        llvm_engine,
        "mask_select_double",
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
    )(
        (ctypes.c_double * 4)(*values),
        (ctypes.c_double * 4)(*mask),
        llvm_selected,
        4,
    )
    assert list(llvm_selected) == [
        c_selected[index] for index in range(c_count)
    ]

    c_incremented = _c_array(values)
    C.increment_mask_double(c_incremented, _c_array(mask), 4)
    llvm_incremented = (ctypes.c_double * 4)(*values)
    _llvm_function(
        llvm_engine,
        "increment_mask_double",
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
    )(llvm_incremented, (ctypes.c_double * 4)(*mask), 4)
    assert list(llvm_incremented) == [
        c_incremented[index] for index in range(4)
    ]


@pytest.mark.parametrize(
    ("symbol", "values"),
    [
        ("cast_double_to_int_values", [-2.9, -0.0, 1.1, 9.9]),
        (
            "cast_double_to_float_values",
            [1.0 / 3.0, -123456.75, 1.0e-20, 1.0e20],
        ),
    ],
)
def test_handwritten_cast_ssa_matches_real_c_kernel(
    llvm_engine,
    symbol,
    values,
):
    c_output = _c_array([0.0] * len(values))
    getattr(C, symbol)(_c_array(values), c_output, len(values))
    llvm_output = (ctypes.c_double * len(values))()
    _llvm_function(
        llvm_engine,
        symbol,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
    )((ctypes.c_double * len(values))(*values), llvm_output, len(values))
    np.testing.assert_equal(
        np.asarray(llvm_output),
        np.asarray([c_output[index] for index in range(len(values))]),
    )


def test_handwritten_sum_and_arange_ssa_match_real_c_kernels(llvm_engine):
    values = [1.0, -2.5, 4.25, 8.0]
    c_sum = C.sum_double(_c_array(values), len(values))
    llvm_sum = _llvm_function(
        llvm_engine,
        "sum_double",
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        result_type=ctypes.c_double,
    )((ctypes.c_double * len(values))(*values), len(values))
    assert llvm_sum == c_sum

    c_output = _c_array([0.0] * 6)
    C.create_arange(-2.0, 0.75, 6, c_output)
    llvm_output = (ctypes.c_double * 6)()
    _llvm_function(
        llvm_engine,
        "create_arange",
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double),
    )(-2.0, 0.75, 6, llvm_output)
    assert list(llvm_output) == [c_output[index] for index in range(6)]


def test_abstract_tensor_tape_lowers_directly_to_real_c_kernel_llvm():
    source_values = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    with autograd.forward_capture() as tape:
        source = NumPyTensorOperations.tensor(source_values)
        result = (source + 2.0) * 3.0

    lowered = lower_abstract_tensor_tape_to_llvm_ssa(tape, result)

    assert lowered.complete
    # The tape's own identity token, not a memory address: the lowering
    # looks its nodes up in the tape and must report back in the same
    # currency it looked them up with.
    assert lowered.feed_ids == (tensor_identity(source),)
    assert "@abstract_tensor_tape" in lowered.llvm_ir
    assert "call void @binary_scalar_double" in lowered.llvm_ir
    assert "@tape.constant.0" in lowered.llvm_ir

    module = llvm.parse_assembly(lowered.llvm_ir)
    module.verify()
    target = llvm.Target.from_default_triple()
    engine = llvm.create_mcjit_compiler(module, target.create_target_machine())
    engine.finalize_object()
    function = ctypes.CFUNCTYPE(
        None,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        *(
            ctypes.POINTER(ctypes.c_double)
            for _ in lowered.workspace_sizes
        ),
    )(engine.get_function_address("abstract_tensor_tape"))
    source_buffer = (ctypes.c_double * len(source_values))(*source_values)
    output_buffer = (ctypes.c_double * len(source_values))()
    workspaces = [
        (ctypes.c_double * size)()
        for size in lowered.workspace_sizes
    ]

    function(source_buffer, output_buffer, *workspaces)

    np.testing.assert_allclose(np.asarray(output_buffer), result.tolist())
    assert "alloca double" not in lowered.llvm_ir


def test_direct_tape_lowering_binds_cumsum_to_real_translated_kernel():
    source_values = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    with autograd.forward_capture() as tape:
        source = NumPyTensorOperations.tensor(source_values)
        result = source.cumsum(dim=0)

    lowered = lower_abstract_tensor_tape_to_llvm_ssa(tape, result)

    assert lowered.complete
    assert "call void @cumsum_dim_double" in lowered.llvm_ir
    llvm.parse_assembly(lowered.llvm_ir).verify()


def test_direct_tape_lowering_composes_real_construction_and_reduction_kernels():
    source_values = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    with autograd.forward_capture() as tape:
        source = NumPyTensorOperations.tensor(source_values)
        total = source.sum()
        sequence = NumPyTensorOperations.arange(1, 7, 2)
        filled = NumPyTensorOperations.full((3,), 4.5)

    lowered = lower_abstract_tensor_tape_to_llvm_ssa(
        tape,
        {"total": total, "sequence": sequence, "filled": filled},
    )

    assert lowered.complete
    assert "call double @sum_double" in lowered.llvm_ir
    assert "call void @create_arange" in lowered.llvm_ir
    assert "call void @fill_double" in lowered.llvm_ir

    module = llvm.parse_assembly(lowered.llvm_ir)
    module.verify()
    engine = llvm.create_mcjit_compiler(
        module,
        llvm.Target.from_default_triple().create_target_machine(),
    )
    engine.finalize_object()
    function = ctypes.CFUNCTYPE(
        None,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    )(engine.get_function_address("abstract_tensor_tape"))
    source_buffer = (ctypes.c_double * 3)(*source_values)
    total_buffer = (ctypes.c_double * 1)()
    sequence_buffer = (ctypes.c_double * 3)()
    filled_buffer = (ctypes.c_double * 3)()

    function(
        source_buffer,
        total_buffer,
        sequence_buffer,
        filled_buffer,
    )

    assert total_buffer[0] == total.tolist()
    assert list(sequence_buffer) == sequence.tolist()
    assert list(filled_buffer) == filled.tolist()


@pytest.mark.parametrize(
    ("solver", "selected_symbol"),
    [
        ("lut", "turing_lut_sin_f64"),
        ("continuous", "turing_continuous_sin_f64"),
    ],
)
def test_direct_tape_lowering_selects_epsilon_controlled_trig_solver(
    solver,
    selected_symbol,
):
    source_values = np.linspace(-2.0, 2.0, 17)
    with autograd.forward_capture() as tape:
        source = NumPyTensorOperations.tensor(source_values)
        result = source.sin()

    lowered = lower_abstract_tensor_tape_to_llvm_ssa(
        tape,
        result,
        trig_solver=solver,
        trig_epsilon=1.0e-6,
    )

    assert lowered.complete
    assert lowered.trig_solver == solver
    assert lowered.trig_epsilon == 1.0e-6
    assert f"call double @{selected_symbol}" in lowered.llvm_ir
    module = llvm.parse_assembly(lowered.llvm_ir)
    module.verify()
    engine = llvm.create_mcjit_compiler(
        module,
        llvm.Target.from_default_triple().create_target_machine(),
    )
    engine.finalize_object()
    function = ctypes.CFUNCTYPE(
        None,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    )(engine.get_function_address("abstract_tensor_tape"))
    source_buffer = (ctypes.c_double * len(source_values))(*source_values)
    output_buffer = (ctypes.c_double * len(source_values))()

    function(source_buffer, output_buffer)

    np.testing.assert_allclose(
        np.asarray(output_buffer),
        np.sin(source_values),
        atol=1.1e-6,
        rtol=0.0,
    )
