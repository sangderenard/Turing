"""Canonical public operation vocabulary for :class:`AbstractTensor`.

This is the shared inventory for graph importers and backend translators.  It
describes tensor-producing operations and observable tensor accessors; runtime
management (backend selection, tape management, hooks, and benchmarking) is
deliberately not an operation in a numerical program.
"""

from __future__ import annotations

import inspect

from .fused_ir import ELEMENTWISE_BINARY, ELEMENTWISE_UNARY


CREATION_OPERATORS = frozenset(
    {
        "arange",
        "empty",
        "eye",
        "eye_like",
        "from_nested",
        "full",
        "full_like",
        "get_tensor",
        "hanning",
        "linspace",
        "meshgrid",
        "ones",
        "ones_like",
        "rand_like",
        "randint",
        "randint_like",
        "randn",
        "random_source",
        "random_tensor",
        "randoms",
        "range",
        "sparse_coo_tensor",
        "tensor",
        "tensor_from_list",
        "tensor_like",
        "zeros",
        "zeros_like",
    }
)

SHAPE_AND_INDEX_OPERATORS = frozenset(
    {
        "assign_at_indices",
        "boolean_mask_select",
        "broadcast_rows",
        "broadcast_to",
        "cat",
        "concat",
        "concatenate",
        "copyto",
        "diag",
        "einsum",
        "expand",
        "flatten",
        "fold2d",
        "fold3d",
        "gather",
        "gather_and",
        "increment_at_indices",
        "index_select",
        "interpolate",
        "outer",
        "pad",
        "pad2d",
        "pad_cat",
        "permute",
        "repeat",
        "repeat_interleave",
        "reshape",
        "scatter",
        "scatter_and",
        "scatter_row",
        "searchsorted",
        "select_by_indices",
        "split",
        "squeeze",
        "stack",
        "swapaxes",
        "topk",
        "transpose",
        "unfold2d",
        "unfold3d",
        "unravel_index",
        "unstack",
        "unsqueeze",
        "view",
        "view_flat",
        "where",
    }
)

REDUCTION_AND_LINALG_OPERATORS = frozenset(
    {
        "all",
        "allclose",
        "any",
        "argmax",
        "argmin",
        "argwhere",
        "cholesky",
        "cross",
        "cumsum",
        "det",
        "dot",
        "eigh",
        "inv",
        "inverse",
        "max",
        "mean",
        "min",
        "nonzero",
        "norm",
        "percentile",
        "prod",
        "solve",
        "sum",
        "trace",
    }
)

COMPOSITE_MATH_OPERATORS = frozenset(
    {
        "bitand",
        "bitor",
        "bitxor",
        "shl",
        "shr",
        "cbrt",
        "clamp",
        "clamp_max",
        "clamp_min",
        "clip",
        "coth",
        "cot",
        "csch",
        "csc",
        "deg2rad",
        "erf",
        "imag",
        "invert",
        "isinf",
        "isinfinite",
        "logical_and",
        "logical_or",
        "long_pi",
        "matmul",
        "nan_to_num",
        "pi",
        "rad2deg",
        "real",
        "sec",
        "sech",
        "sign",
        "sinc",
        "softmax",
        "log_softmax",
    }
)

SPECTRAL_OPERATORS = frozenset(
    {
        "fft",
        "fftfreq",
        "ifft",
        "irfft",
        "rfft",
        "rfftfreq",
    }
)

TYPE_AND_DEVICE_OPERATORS = frozenset(
    {
        "astype",
        "bool",
        "cast_like",
        "cpu",
        "cuda",
        "double",
        "float",
        "int",
        "long",
        "long_cast",
        "to",
        "to_backend",
        "to_device",
        "to_dtype",
    }
)

VALUE_LIFECYCLE_OPERATORS = frozenset(
    {
        "clone",
        "copy",
        "detach",
        "ensure_tensor",
    }
)

ACCESSOR_OPERATORS = frozenset(
    {
        "datastring",
        "device",
        "dim",
        "dtype",
        "get_device",
        "get_dtype",
        "get_ndims",
        "get_shape",
        "item",
        "nbytes",
        "ndim",
        "ndims",
        "numel",
        "shape",
        "tensor_type",
    }
)

# ---------------------------------------------------------------------------
# HOST-BOUNDARY SAFETY RULE
#
# This set is for operations whose *implementation is intrinsically external*
# to the tensor/compiler program: filesystem access, foreign host memory, or a
# final materialization crossing.  It must never be used as a convenient way
# to hide a large tensor algorithm from ProcessGraph.
#
# In particular, a codec does not become a host operation merely because its
# terminal value is ``bytes``.  DCT, quantization, coefficient transforms,
# scans, prefix work, and every other tensor/numerical portion must remain
# visible to ProcessGraph so the planner can reduce, partition, and compile
# it.  Only the final, genuinely non-tensor byte publication belongs at the
# host boundary.
#
# Agents have repeatedly wrapped visible tensor work in one Python function,
# added its name here, observed thousands of eager backend calls, and then
# spoken as though the topology had proved impossible to reduce.  That is
# false: adding a name here prevents the planner from seeing the topology at
# all.  It is a compiler bypass, not a compiler result.
#
# Do not add an operator to this set to make ingestion, planning, capture, or
# lowering appear to succeed.  If an operation contains compiler-relevant
# tensor work, expose that work and fix the compiler.  Hiding it here ruins
# the execution boundary and guarantees Python/host churn at runtime.
# ---------------------------------------------------------------------------
HOST_BOUNDARY_OPERATORS = frozenset(
    {
        "avi",
        "jpg",
        "load",
        "mjpeg_frames",
        "numpy",
        "save",
        "tobytes",
        "tolist",
    }
)

# Public compatibility spellings are retained at graph boundaries and resolve
# to one canonical operation before backend planning.
OPERATOR_ALIASES = {
    "abs_": "abs",
    "argmax_": "argmax",
    "argmin_": "argmin",
    "argwhere_": "argwhere",
    "ceil_": "ceil",
    "clamp_": "clamp",
    "clamp_max_": "clamp_max",
    "clamp_min_": "clamp_min",
    "copyto_": "copyto",
    "cumsum_": "cumsum",
    "empty_": "empty",
    "equal_": "equal",
    "exp_": "exp",
    "expand_": "expand",
    "fft_": "fft",
    "fftfreq_": "fftfreq",
    "floor_": "floor",
    "full_": "full",
    "full_like_": "full_like",
    "greater_": "greater",
    "greater_equal_": "greater_equal",
    "ifft_": "ifft",
    "imag_": "imag",
    "invert_": "invert",
    "irfft_": "irfft",
    "less_": "less",
    "less_equal_": "less_equal",
    "log_": "log",
    "log_softmax_": "log_softmax",
    "logical_not_": "logical_not",
    "max_": "max",
    "mean_": "mean",
    "min_": "min",
    "nbytes_": "nbytes",
    "neg_": "neg",
    "not_equal_": "not_equal",
    "ones_": "ones",
    "ones_like_": "ones_like",
    "rand_like": "random_source",
    "randint": "random_source",
    "randint_like": "random_source",
    "randn": "random_source",
    "random_tensor": "random_source",
    "randoms": "random_source",
    "pad2d_": "pad2d",
    "real_": "real",
    "repeat_interleave_": "repeat_interleave",
    "reshape_": "reshape",
    "rfft_": "rfft",
    "rfftfreq_": "rfftfreq",
    "round_": "round",
    "softmax_": "softmax",
    "sqrt_": "sqrt",
    "squeeze_": "squeeze",
    "shape_": "shape",
    "sum_": "sum",
    "tensor_from_list_": "tensor_from_list",
    "tolist_": "tolist",
    "transpose_": "transpose",
    "T": "transpose",
    "trunc_": "trunc",
    "unravel_index_": "unravel_index",
    "zeros_": "zeros",
    "zeros_like_": "zeros_like",
}

CANONICAL_ABSTRACT_TENSOR_OPERATORS = frozenset().union(
    ELEMENTWISE_UNARY,
    ELEMENTWISE_BINARY,
    CREATION_OPERATORS,
    SHAPE_AND_INDEX_OPERATORS,
    REDUCTION_AND_LINALG_OPERATORS,
    COMPOSITE_MATH_OPERATORS,
    SPECTRAL_OPERATORS,
    TYPE_AND_DEVICE_OPERATORS,
    VALUE_LIFECYCLE_OPERATORS,
    ACCESSOR_OPERATORS,
    HOST_BOUNDARY_OPERATORS,
)

PUBLIC_ABSTRACT_TENSOR_OPERATOR_NAMES = frozenset(
    CANONICAL_ABSTRACT_TENSOR_OPERATORS | OPERATOR_ALIASES.keys()
)

# These public names control execution or autograd rather than describe a node
# in a numerical program.  Keeping the exclusion explicit makes API drift
# inspectable instead of silently treating an orchestration method as math.
NON_OPERATOR_PUBLIC_API = frozenset(
    {
        "backend_class_from_backend_data",
        "autograd",
        "backward",
        "benchmark",
        "bool_dtype",
        "bool_dtype_",
        "check_or_build_registry",
        "data_or",
        "F",
        "float_dtype",
        "float_dtype_",
        "get_backward_tool",
        "grad",
        "grad_fn",
        "inf",
        "is_leaf",
        "linalg",
        "long_dtype",
        "long_dtype_",
        "nan",
        "ninf",
        "random",
        "register_hook",
        "requires_grad",
        "requires_grad_",
        "retain_grad",
        "set_default_backend",
        "use_backend",
        "use_tape",
        "zero_grad",
    }
)


def canonical_operator_name(name: str) -> str:
    """Resolve a public compatibility spelling to the canonical vocabulary."""

    return OPERATOR_ALIASES.get(name, name)


def include_ast_parent_outside_abstract_tensor(value) -> bool:
    """Expand Python parents until reaching a canonical tensor operation."""

    target = value.__func__ if inspect.ismethod(value) else value
    name = str(getattr(target, "__name__", ""))
    module = str(getattr(target, "__module__", ""))
    canonical = canonical_operator_name(name)
    tensor_module = (
        module == "src.common.tensors"
        or module.startswith("src.common.tensors.")
    )
    return not (
        tensor_module
        and canonical in CANONICAL_ABSTRACT_TENSOR_OPERATORS
    )


__all__ = [
    "ACCESSOR_OPERATORS",
    "CANONICAL_ABSTRACT_TENSOR_OPERATORS",
    "COMPOSITE_MATH_OPERATORS",
    "CREATION_OPERATORS",
    "HOST_BOUNDARY_OPERATORS",
    "NON_OPERATOR_PUBLIC_API",
    "OPERATOR_ALIASES",
    "PUBLIC_ABSTRACT_TENSOR_OPERATOR_NAMES",
    "REDUCTION_AND_LINALG_OPERATORS",
    "SHAPE_AND_INDEX_OPERATORS",
    "SPECTRAL_OPERATORS",
    "TYPE_AND_DEVICE_OPERATORS",
    "VALUE_LIFECYCLE_OPERATORS",
    "canonical_operator_name",
    "include_ast_parent_outside_abstract_tensor",
]
