"""Dynamic C backend for tensor operations."""

# TENSOR BACKEND IMPLEMENTATION GUIDELINES:
# ----------------------------------------
# 1. OPERATOR IMPLEMENTATION:
#    - DO NOT implement magic methods (__add__, __mul__, etc.)
#    - These are handled by AbstractTensor
#    - Only implement the single designated operator method from the abstract class
#
# 2. TEST COMPLIANCE:
#    - DO NOT create dummy/mock classes to pass tests
#    - DO NOT implement functions just to satisfy test requirements
#    - Either implement full functionality or leave as documented stub
#    - Failed tests are preferable to false implementations
#
# 3. BACKEND RESPONSIBILITIES:
#    - Implement only the core tensor operations defined in AbstractTensor
#    - All operator routing happens through the abstract class
#    - Let test failures expose missing functionality naturally
#
# 4. DEPENDENCIES:
#    - Import only the strictly required packages
#    - Handle import failures gracefully for optional backends
#    - Do not add dummy fallbacks for missing dependencies
#
# Remember: Magic methods and operator overloading are EXCLUSIVELY handled by
# AbstractTensor. Backend implementations provide only the raw
# tensor operations.

import os
import ctypes
import ctypes.util
import json
from typing import Any, Tuple, Optional, List
from cffi import FFI

# The tensor abstraction module was renamed to ``abstraction``. Update imports
# accordingly so the C backend stays in sync with the other backends.
from ..abstraction import (
    AbstractTensor,
    _get_shape,
    _flatten,
    register_backend,
)
from ..abstraction_methods.indexing import lower_basic_index


ffi = FFI()
ffi.cdef("""
    typedef enum CTensorOp {
        CT_OP_ADD, CT_OP_SUB, CT_OP_MUL, CT_OP_DIV, CT_OP_POW,
        CT_OP_MOD, CT_OP_FLOORDIV, CT_OP_SQRT, CT_OP_EXP, CT_OP_LOG,
        CT_OP_NEG, CT_OP_ABS, CT_OP_ROUND, CT_OP_TRUNC, CT_OP_FLOOR,
        CT_OP_CEIL, CT_OP_ISFINITE, CT_OP_ISNAN, CT_OP_ISINF,
        CT_OP_LOGICAL_NOT, CT_OP_LT, CT_OP_LE, CT_OP_GT, CT_OP_GE,
        CT_OP_EQ, CT_OP_NE, CT_OP_MAXIMUM, CT_OP_MINIMUM, ...
    } CTensorOp;
    typedef enum CTensorOperandKind {
        CT_OPERAND_NONE, CT_OPERAND_SLOT, CT_OPERAND_SCALAR
    } CTensorOperandKind;
    typedef struct CTensorPrimitiveInstruction {
        CTensorOp op;
        int out_slot;
        int left_slot;
        CTensorOperandKind right_kind;
        int right_slot;
        double right_scalar;
        int reverse;
    } CTensorPrimitiveInstruction;
    void fill_double(double* out, double value, int n);
    void binary_double(
        const double* a, const double* b, double* out, int n, int op);
    void binary_scalar_double(
        const double* a, double b, double* out, int n, int op, int reverse);
    void matmul_double(const double* a, const double* b, double* out, int m, int n, int p);
    void unary_double(const double* a, double* out, int n, int op);
    int ctensor_execute_primitive_program(
        const CTensorPrimitiveInstruction* instructions,
        int instruction_count,
        const double* const* feeds,
        int feed_count,
        double* workspace,
        int slot_count,
        int element_count,
        int output_slot,
        double* output);
    int ctensor_execute_primitive_program_slots(
        const CTensorPrimitiveInstruction* instructions,
        int instruction_count,
        double* const* slots,
        int slot_count,
        int element_count);
    void reduce_dim_double(
        const double* a, double* out, const int* shape, int ndim,
        int dim, int op);
    void transpose_double(
        const double* a, double* out, const int* shape,
        const int* axes, int ndim);
    void where_double(
        const double* condition, const double* x, const double* y,
        double* out, int n);
    void broadcast_double(
        const double* input, double* output, const int* input_shape,
        int input_ndim, const int* output_shape, int output_ndim);
    void cumsum_dim_double(
        const double* input, double* output, const int* shape,
        int ndim, int dim);
    void argreduce_dim_double(
        const double* input, double* output, const int* shape,
        int ndim, int dim, int find_max);
    void repeat_interleave_double(
        const double* input, double* output, const int* shape,
        int ndim, int dim, int repeats);
    void tile_double(
        const double* input, double* output, const int* input_shape,
        const int* output_shape, int ndim);
    void index_select_double(
        const double* input, double* output, const int* shape,
        int ndim, int dim, const int* indices, int index_count);
    int count_true_double(const double* mask, int n);
    void mask_select_double(
        const double* input, const double* mask, double* output, int n);
    void increment_mask_double(double* input, const double* mask, int n);
    void log_softmax_1d(const double* a, double* out, int n);
    void log_softmax_dim(
        const double* a,
        const int* shape,
        int ndim,
        int dim,
        double* out);

    void pad_double_nd(const double* input, double* output, const int* shape, const int* new_shape, const int* left_pad, int dims, double value);
    void mean_dim(const double* a, double* out, const int* shape, int ndim, int dim);
    void gather_pairs_2d(const double* a, const int* rows, const int* cols,
                         double* out, int n_pairs, int stride);
    double sum_double(const double* a, int n);
    void create_arange(double start, double step, int n, double* out);
    void topk_double(const double* a, int n, int k, int* indices, double* out);
    void topk_double_dim(
        const double* a,
        const int* shape,
        int ndim,
        int dim,
        int k,
        double* indices,
        double* out);
    void for_each_cell_along_dim(
        const double* data,
        const int* shape,
        int ndim,
        int batch_dim,
        void (*callback)(const double*, int, int, void*),
        void* user_data);

    void cast_double_to_int_values(const double* a, double* out, int n);
    void cast_double_to_float_values(const double* a, double* out, int n);
    void stack_double(const double** tensors, int num_tensors, const int* shape, int ndim, int dim, double* out);
    void cat_double(const double** tensors, const int* dim_sizes, int num_tensors, const int* shape, int ndim, int dim, double* out);
""")

from pathlib import Path

# Attempt to load the C implementation from a standalone source file. The
# repository previously embedded a full copy of the C source as a fallback, but
# this redundancy made maintenance difficult. The file must now exist or the
# backend will raise an error. This approach encourages explicit dependency
# management and avoids silent mismatches between versions.
SOURCE_PATH = Path(__file__).with_name("c_backend") / "ctensor_ops.c"
if not SOURCE_PATH.exists():
    raise FileNotFoundError(f"Missing C source: {SOURCE_PATH}")
C_SOURCE = SOURCE_PATH.read_text()

_prebuilt = os.environ.get("TENSOR_CTENSOR_LIB")
if _prebuilt and os.path.exists(_prebuilt):
    C = ffi.dlopen(_prebuilt)
else:
    C = ffi.verify(
        C_SOURCE, include_dirs=[str(SOURCE_PATH.parent)]
    )

# ########## STUB: build_ctensor_with_zig ##########
# PURPOSE: Compile ``ctensor_ops.c`` into a shared library using the Zig
#          toolchain for faster startup and optional precompiled binaries.
# EXPECTED BEHAVIOR: When implemented, this function will invoke Zig's
#          C compiler, output a platform-specific shared object, and return
#          the path. The resulting binary may be cached inside the virtual
#          environment and loaded via ``ffi.dlopen``.
# INPUTS: ``source_path`` (str) pointing to ``ctensor_ops.c`` and an
#         ``out_dir`` for the compiled artifact.
# OUTPUTS: Path to the compiled library.
# KEY ASSUMPTIONS/DEPENDENCIES: Requires the ``ziglang`` package which
#         bundles the Zig binary. Compilation occurs only if no prebuilt
#         library is supplied via ``TENSOR_CTENSOR_LIB``.
# TODO:
#   - Implement the Zig command invocation.
#   - Add caching logic to avoid recompilation.
# NOTES: This stub is the first step toward supporting non-CFFI builds.
# ###########################################################################
def build_ctensor_with_zig(source_path: str, out_dir: str) -> str:
    """Compile ``ctensor_ops.c`` using Zig's embedded clang compiler."""
    from importlib.util import find_spec
    from pathlib import Path
    import subprocess
    import sys

    if find_spec("ziglang") is None:
        raise RuntimeError("ziglang package is required to build ctensor library")

    import ziglang  # type: ignore

    zig_exe = Path(ziglang.__file__).with_name("zig")
    ext = {
        "linux": ".so",
        "darwin": ".dylib",
        "win32": ".dll",
    }.get(sys.platform, ".so")

    out_path = Path(out_dir) / f"ctensor_ops{ext}"
    if not out_path.exists():
        cmd = [
            sys.executable,
            "-m",
            "ziglang",
            "cc",
            "-shared",
            "-O3",
            source_path,
            "-o",
            str(out_path),
        ]
        subprocess.check_call(cmd)

    return str(out_path)

class CTensor:
    """C-backed tensor using cffi buffer."""
    def __init__(self, shape: Tuple[int, ...], buffer=None):
        self.shape = shape
        self.size = 1
        for dim in shape:
            self.size *= dim
        self.buffer = buffer if buffer is not None else ffi.new("double[]", self.size)

    def as_c_ptr(self):
        return self.buffer

    def tolist(self):
        def build(offset: int, shp: Tuple[int, ...]):
            if not shp:
                return float(self.buffer[offset])
            step = 1
            for s in shp[1:]:
                step *= s
            return [build(offset + i * step, shp[1:]) for i in range(shp[0])]

        return build(0, self.shape)

    def __getitem__(self, idx):
        """Return a Python value or CTensor slice using Python-level indexing."""
        data_list = self.tolist()
        result = data_list[idx]
        if isinstance(result, list):
            return CTensor.from_list(result, _get_shape(result))
        return float(result)

    @classmethod
    def from_list(cls, data: list, shape: Tuple[int, ...]):
        flat = []
        def flatten(x):
            if isinstance(x, list):
                for item in x:
                    flatten(item)
            else:
                flat.append(float(x))
        flatten(data)
        buf = ffi.new("double[]", [float(x) for x in flat])
        return cls(shape, buf)

class CTensorOperations(AbstractTensor):
    """C backend using cffi for all arithmetic ops."""

    def _apply_operator__(self, op: str, left: CTensor, right: Any):
        """Lower the canonical AbstractTensor operation vocabulary to C."""
        binary_codes = {
            "add": C.CT_OP_ADD,
            "sub": C.CT_OP_SUB,
            "mul": C.CT_OP_MUL,
            "truediv": C.CT_OP_DIV,
            "pow": C.CT_OP_POW,
            "mod": C.CT_OP_MOD,
            "floordiv": C.CT_OP_FLOORDIV,
            "less": C.CT_OP_LT,
            "less_equal": C.CT_OP_LE,
            "greater": C.CT_OP_GT,
            "greater_equal": C.CT_OP_GE,
            "equal": C.CT_OP_EQ,
            "not_equal": C.CT_OP_NE,
            "maximum": C.CT_OP_MAXIMUM,
            "minimum": C.CT_OP_MINIMUM,
        }
        unary_codes = {
            "sqrt": C.CT_OP_SQRT,
            "exp": C.CT_OP_EXP,
            "log": C.CT_OP_LOG,
            "neg": C.CT_OP_NEG,
            "abs": C.CT_OP_ABS,
            "round": C.CT_OP_ROUND,
            "trunc": C.CT_OP_TRUNC,
            "floor": C.CT_OP_FLOOR,
            "ceil": C.CT_OP_CEIL,
            "isfinite": C.CT_OP_ISFINITE,
            "isnan": C.CT_OP_ISNAN,
            "isinf": C.CT_OP_ISINF,
            "logical_not": C.CT_OP_LOGICAL_NOT,
        }
        if isinstance(left, CTensor) and right is None:
            code = unary_codes.get(op)
            if code is None:
                raise NotImplementedError(
                    f"Unary operator {op} not implemented for C backend"
                )
            out = CTensor(left.shape)
            C.unary_double(
                left.as_c_ptr(), out.as_c_ptr(), left.size, code
            )
            return out
        if isinstance(right, CTensor) and isinstance(left, CTensor):
            if op in ('matmul', 'rmatmul', 'imatmul'):
                a, b = (left, right) if op != 'rmatmul' else (right, left)
                if len(a.shape) != 2 or len(b.shape) != 2:
                    raise ValueError("matmul expects 2D tensors")
                m, n = a.shape
                n2, p = b.shape
                if n != n2:
                    raise ValueError("Shape mismatch for matmul")
                out = CTensor((m, p))
                C.matmul_double(a.as_c_ptr(), b.as_c_ptr(), out.as_c_ptr(), m, n, p)
                if op == 'imatmul':
                    left.buffer = out.buffer
                    left.shape = out.shape
                    left.size = out.size
                    return left
                return out
            if left.shape != right.shape:
                rank = max(len(left.shape), len(right.shape))
                left_shape = (1,) * (rank - len(left.shape)) + left.shape
                right_shape = (1,) * (rank - len(right.shape)) + right.shape
                shape = []
                for left_size, right_size in zip(left_shape, right_shape):
                    if left_size == right_size or left_size == 1:
                        shape.append(right_size)
                    elif right_size == 1:
                        shape.append(left_size)
                    else:
                        raise ValueError("C operands are not broadcastable")
                target = tuple(shape)
                if left.shape != target:
                    temporary = type(self)()
                    temporary.data = left
                    left = temporary.expand_(target)
                if right.shape != target:
                    temporary = type(self)()
                    temporary.data = right
                    right = temporary.expand_(target)
            out = CTensor(left.shape)
            n = left.size
            canonical = op[1:] if op.startswith(("i", "r")) else op
            code = binary_codes.get(canonical)
            if code is None:
                raise NotImplementedError(f"Operator {op} not implemented for C backend.")
            C.binary_double(
                left.as_c_ptr(), right.as_c_ptr(), out.as_c_ptr(), n, code
            )
            return out
        elif isinstance(left, (int, float)) and isinstance(right, CTensor):
            return self._apply_operator__(op, right, left)
        elif isinstance(left, CTensor) and isinstance(right, (int, float)):
            out = CTensor(left.shape)
            n = left.size
            val = float(right)
            reverse = op.startswith("r")
            canonical = op[1:] if op.startswith(("i", "r")) else op
            code = binary_codes.get(canonical)
            if code is None:
                raise NotImplementedError(f"Operator {op} not implemented for C backend.")
            C.binary_scalar_double(
                left.as_c_ptr(), val, out.as_c_ptr(), n, code, int(reverse)
            )
            return out
        else:
            raise TypeError("CTensorOperations only supports CTensor or scalar operands.")

    # Creation ops
    def full_(self, size: Tuple[int, ...], fill_value: Any, dtype: Any, device: Any):
        t = CTensor(size)
        C.fill_double(t.as_c_ptr(), float(fill_value), t.size)
        return t

    def zeros_(self, size: Tuple[int, ...], dtype: Any, device: Any):
        return self.full_(size, 0.0, dtype, device)

    def clone_(self, tensor: CTensor = None) -> CTensor:
        if tensor is None:
            tensor = self.data
        t = CTensor(tensor.shape)
        ffi.memmove(t.buffer, tensor.buffer, tensor.size * ffi.sizeof("double"))
        return t

    def to_device_(self, tensor: CTensor, device: Any) -> CTensor:
        return tensor  # No-op for now

    def arange_(
        self,
        start: int,
        end: Optional[int] = None,
        step: int = 1,
        *,
        dtype: Any = None,
        device: Any = None,
    ) -> CTensor:
        if end is None:
            n = start
            start_val = 0.0
            step_val = 1.0
        else:
            n = int((end - start) // step)
            start_val = float(start)
            step_val = float(step)
        out = CTensor((n,))
        C.create_arange(start_val, step_val, n, out.as_c_ptr())
        return out

    def pow_(self, tensor: Any, exponent: float) -> CTensor:
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))
        return self._apply_operator__("pow", tensor, float(exponent))

    def sqrt_(self, tensor: Any = None) -> CTensor:
        if tensor is None:
            tensor = self.data
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))
        return self._apply_operator__("sqrt", tensor, None)

    def exp_(self):
        return self._apply_operator__("exp", self.data, None)

    def log_(self):
        return self._apply_operator__("log", self.data, None)

    def neg_(self):
        return self._apply_operator__("neg", self.data, None)

    def abs_(self):
        return self._apply_operator__("abs", self.data, None)

    def round_(self, n=None):
        if n not in (None, 0):
            scale = 10.0 ** int(n)
            scaled = self._apply_operator__(
                "mul", self.data, scale
            )
            temporary = type(self)()
            temporary.data = scaled
            rounded = temporary._apply_operator__(
                "round", temporary.data, None
            )
            return self._apply_operator__("truediv", rounded, scale)
        return self._apply_operator__("round", self.data, None)

    def trunc_(self):
        return self._apply_operator__("trunc", self.data, None)

    def floor_(self):
        return self._apply_operator__("floor", self.data, None)

    def ceil_(self):
        return self._apply_operator__("ceil", self.data, None)

    def isfinite_(self):
        return self._apply_operator__("isfinite", self.data, None)

    def isnan_(self):
        return self._apply_operator__("isnan", self.data, None)

    def isinf_(self):
        return self._apply_operator__("isinf", self.data, None)

    def logical_not_(self):
        return self._apply_operator__("logical_not", self.data, None)

    def real_(self):
        return self.clone_()

    def imag_(self):
        return self.zeros_(self.data.shape, None, None)

    def less_(self, value):
        value = value.data if isinstance(value, AbstractTensor) else value
        return self._apply_operator__("less", self.data, value)

    def less_equal_(self, value):
        value = value.data if isinstance(value, AbstractTensor) else value
        return self._apply_operator__("less_equal", self.data, value)

    def greater_(self, value):
        value = value.data if isinstance(value, AbstractTensor) else value
        return self._apply_operator__("greater", self.data, value)

    def greater_equal_(self, value):
        value = value.data if isinstance(value, AbstractTensor) else value
        return self._apply_operator__("greater_equal", self.data, value)

    def equal_(self, value):
        value = value.data if isinstance(value, AbstractTensor) else value
        return self._apply_operator__("equal", self.data, value)

    def not_equal_(self, tensor1, tensor2=None):
        if tensor2 is None:
            tensor1 = (
                tensor1.data
                if isinstance(tensor1, AbstractTensor) else tensor1
            )
            return self._apply_operator__("not_equal", self.data, tensor1)
        previous = self.data
        try:
            self.data = tensor1
            tensor2 = (
                tensor2.data
                if isinstance(tensor2, AbstractTensor) else tensor2
            )
            return self._apply_operator__("not_equal", self.data, tensor2)
        finally:
            self.data = previous

    def maximum_(self, other):
        other = other.data if isinstance(other, AbstractTensor) else other
        return self._apply_operator__("maximum", self.data, other)

    def minimum_(self, other):
        other = other.data if isinstance(other, AbstractTensor) else other
        return self._apply_operator__("minimum", self.data, other)

    def empty_(self, size, dtype=None, device=None):
        return CTensor(tuple(size))

    def reshape_(self, shape):
        shape = list(shape)
        unknown = [index for index, size in enumerate(shape) if size == -1]
        if len(unknown) > 1:
            raise ValueError("only one inferred dimension is permitted")
        known = 1
        for size in shape:
            if size != -1:
                known *= size
        if unknown:
            if known == 0 or self.data.size % known:
                raise ValueError("shape is incompatible with tensor size")
            shape[unknown[0]] = self.data.size // known
        size = 1
        for dimension in shape:
            size *= dimension
        if size != self.data.size:
            raise ValueError("shape is incompatible with tensor size")
        return CTensor(tuple(shape), self.data.buffer)

    def flatten_(self, start_dim=0, end_dim=-1):
        ndim = len(self.data.shape)
        start_dim %= ndim
        end_dim %= ndim
        if start_dim > end_dim:
            raise ValueError("start_dim must not follow end_dim")
        merged = 1
        for size in self.data.shape[start_dim:end_dim + 1]:
            merged *= size
        shape = (
            self.data.shape[:start_dim]
            + (merged,)
            + self.data.shape[end_dim + 1:]
        )
        return CTensor(shape, self.data.buffer)

    def unsqueeze_(self, dim):
        ndim = len(self.data.shape) + 1
        dim %= ndim
        shape = self.data.shape[:dim] + (1,) + self.data.shape[dim:]
        return CTensor(shape, self.data.buffer)

    def squeeze_(self, dim=None):
        shape = self.data.shape
        if dim is None:
            result = tuple(size for size in shape if size != 1)
        else:
            dim %= len(shape)
            result = (
                shape[:dim] + shape[dim + 1:]
                if shape[dim] == 1 else shape
            )
        return CTensor(result, self.data.buffer)

    def permute_(self, dims):
        dims = tuple(axis % len(self.data.shape) for axis in dims)
        if sorted(dims) != list(range(len(self.data.shape))):
            raise ValueError("dims must be a permutation of tensor axes")
        shape = tuple(self.data.shape[axis] for axis in dims)
        out = CTensor(shape)
        C.transpose_double(
            self.data.as_c_ptr(), out.as_c_ptr(),
            ffi.new("int[]", self.data.shape),
            ffi.new("int[]", dims), len(dims),
        )
        return out

    def transpose_(self, dim0, dim1):
        axes = list(range(len(self.data.shape)))
        dim0 %= len(axes)
        dim1 %= len(axes)
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return self.permute_(axes)

    def swapaxes_(self, axis1, axis2):
        return self.transpose_(axis1, axis2)

    def expand_(self, shape):
        target = list(shape)
        if len(target) < len(self.data.shape):
            raise ValueError("cannot expand to fewer dimensions")
        source_shape = (
            (1,) * (len(target) - len(self.data.shape)) + self.data.shape
        )
        for axis, (current, desired) in enumerate(
            zip(source_shape, target)
        ):
            if desired == -1:
                target[axis] = current
            elif desired < 0 or (current != desired and current != 1):
                raise ValueError(
                    f"cannot expand dimension {axis} "
                    f"from {current} to {desired}"
                )
        target_shape = tuple(target)
        out = CTensor(target_shape)
        C.broadcast_double(
            self.data.as_c_ptr(), out.as_c_ptr(),
            ffi.new("int[]", source_shape), len(source_shape),
            ffi.new("int[]", target_shape), len(target_shape),
        )
        return out

    def cumsum_(self, dim=0):
        dim %= len(self.data.shape)
        out = CTensor(self.data.shape)
        C.cumsum_dim_double(
            self.data.as_c_ptr(), out.as_c_ptr(),
            ffi.new("int[]", self.data.shape), len(self.data.shape), dim,
        )
        return out

    def _argreduce_c(self, dim, keepdim, find_max):
        if dim is None:
            source = CTensor((self.data.size,), self.data.buffer)
            shape = source.shape
            dim = 0
            reduce_all = True
        else:
            source = self.data
            shape = source.shape
            dim %= len(shape)
            reduce_all = False
        out_shape = shape[:dim] + shape[dim + 1:]
        out = CTensor(out_shape if out_shape else ())
        C.argreduce_dim_double(
            source.as_c_ptr(), out.as_c_ptr(),
            ffi.new("int[]", shape), len(shape), dim, int(find_max),
        )
        if keepdim:
            target = (
                (1,) * len(self.data.shape)
                if reduce_all
                else self.data.shape[:dim]
                + (1,)
                + self.data.shape[dim + 1:]
            )
            return CTensor(target, out.buffer)
        return out

    def argmin_(self, dim=None, keepdim=False):
        return self._argreduce_c(dim, keepdim, False)

    def argmax_(self, dim=None, keepdim=False):
        return self._argreduce_c(dim, keepdim, True)

    def softmax_(self, dim):
        logged = self.log_softmax_(dim)
        temporary = type(self)()
        temporary.data = logged
        return temporary.exp_()

    def log_softmax_tensor_(self, dim):
        return self.log_softmax_(dim)

    def matmul_(self, tensor: Any, other: Any) -> CTensor:
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))
        if not isinstance(other, CTensor):
            other = CTensor.from_list(other, _get_shape(other))
        if len(tensor.shape) != 2 or len(other.shape) != 2:
            raise ValueError("matmul expects 2D tensors")
        m, n = tensor.shape
        n2, p = other.shape
        if n != n2:
            raise ValueError("Shape mismatch for matmul")
        out = CTensor((m, p))
        C.matmul_double(tensor.as_c_ptr(), other.as_c_ptr(), out.as_c_ptr(), m, n, p)
        return out

    def tensor_from_list_(self, data: List[Any], dtype: Any, device: Any) -> CTensor:
        shape = _get_shape(data)
        return CTensor.from_list(data, shape)

    def get_item_(self, data, index):
        """Lower scalar/slice or shaped first-axis gather into index_select."""
        previous = self.data
        self.data = data
        try:
            return self._get_item_from_data(index)
        finally:
            self.data = previous

    def _get_item_from_data(self, index):
        if isinstance(index, tuple):
            def select(data, axis, indices):
                temporary = type(self)()
                temporary.data = data
                return temporary.index_select_(axis, indices)

            return lower_basic_index(
                self.data,
                index,
                shape_of=lambda data: data.shape,
                index_select=select,
                reshape=lambda data, shape: CTensor(shape, data.buffer),
            )
        if isinstance(index, slice):
            indices = list(range(*index.indices(self.data.shape[0])))
            index_shape = (len(indices),)
            drop_axis = False
        elif isinstance(index, int):
            normalized = index % self.data.shape[0]
            indices = [normalized]
            index_shape = ()
            drop_axis = True
        else:
            raw = index.tolist() if hasattr(index, "tolist") else index
            index_shape = _get_shape(raw)
            indices = [int(value) for value in _flatten(raw)]
            drop_axis = False
        selected = self.index_select_(0, indices)
        if drop_axis:
            return CTensor(self.data.shape[1:], selected.buffer)
        return CTensor(index_shape + self.data.shape[1:], selected.buffer)

    def shape_(self, tensor: CTensor = None) -> Tuple[int, ...]:
        if tensor is None:
            tensor = self.data
        return tensor.shape

    def numel_(self, tensor: CTensor = None) -> int:
        if tensor is None:
            tensor = self.data
        return tensor.size

    def __trunc__(self):
        import math
        tensor = self.data
        if tensor.size != 1:
            raise TypeError("Only scalar tensors can be converted to int")
        return int(math.trunc(tensor.buffer[0]))

    def mean_(
        self,
        tensor: Any = None,
        dim: Optional[int] = None,
        keepdim: bool = False,
    ) -> Any:
        if tensor is None:
            tensor = self.data
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))
        if dim is None:
            value = (
                C.sum_double(tensor.as_c_ptr(), tensor.size) / tensor.size
                if tensor.size else 0.0
            )
            out = CTensor((1,) * len(tensor.shape) if keepdim else ())
            out.buffer[0] = value
            return out

        shape = tensor.shape
        if dim < 0:
            dim += len(shape)
        if dim < 0 or dim >= len(shape):
            raise ValueError("dim out of range")

        out_shape = (
            shape[:dim] + (1,) + shape[dim + 1 :]
            if keepdim
            else shape[:dim] + shape[dim + 1 :]
        )
        out = CTensor(out_shape if out_shape else ())
        shape_arr = ffi.new("int[]", list(shape))
        C.mean_dim(tensor.as_c_ptr(), out.as_c_ptr(), shape_arr, len(shape), dim)
        if not out_shape:
            return out.buffer[0]
        return out

    def _reduce_c(self, dim, keepdim, op):
        tensor = self.data
        reduce_all = dim is None
        if reduce_all:
            source = CTensor((tensor.size,), tensor.buffer)
            shape = source.shape
            dim = 0
        else:
            shape = tensor.shape
            dim %= len(shape)
            source = tensor
        reduced_shape = shape[:dim] + shape[dim + 1:]
        out = CTensor(reduced_shape if reduced_shape else ())
        C.reduce_dim_double(
            source.as_c_ptr(), out.as_c_ptr(),
            ffi.new("int[]", shape), len(shape), dim, op,
        )
        if keepdim:
            if tensor.shape:
                target = (
                    (1,) * len(tensor.shape)
                    if reduce_all
                    else tensor.shape[:dim]
                    + (1,)
                    + tensor.shape[dim + 1:]
                )
            else:
                target = ()
            return CTensor(target, out.buffer)
        return out

    def sum_(self, dim=None, keepdim=False):
        return self._reduce_c(dim, keepdim, 0)

    def prod_(self, dim=None, keepdim=False):
        return self._reduce_c(dim, keepdim, 1)

    def min_(self, dim=None, keepdim=False):
        return self._reduce_c(dim, keepdim, 2)

    def max_(self, dim=None, keepdim=False):
        return self._reduce_c(dim, keepdim, 3)

    def any_(self, dim=None):
        return self._reduce_c(dim, False, 4)

    def all_(self, dim=None):
        return self._reduce_c(dim, False, 5)

    def view_flat_(self, tensor: Any) -> list:
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))
        return _flatten(tensor.tolist())

    def tolist_(self, tensor: Any = None) -> list:
        if tensor is None:
            tensor = self.data
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))
        return tensor.tolist()

    def clamp_(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> CTensor:
        out = self.data
        if min_val is not None:
            out = self._apply_operator__(
                "maximum", self.data, min_val
            )
        if max_val is not None:
            temporary = type(self)()
            temporary.data = out
            out = temporary._apply_operator__(
                "minimum", temporary.data, max_val
            )
        return out

    def clamp_min_(self, min_val):
        return self._apply_operator__("maximum", self.data, min_val)

    def clamp_max_(self, max_val):
        return self._apply_operator__("minimum", self.data, max_val)

    def where_(self, x, y):
        def operand(value):
            value = value.data if isinstance(value, AbstractTensor) else value
            if isinstance(value, CTensor):
                if value.shape != self.data.shape:
                    raise ValueError("C where operand shapes must match")
                return value
            zeros = CTensor(self.data.shape)
            filled = CTensor(self.data.shape)
            C.binary_scalar_double(
                zeros.as_c_ptr(), float(value), filled.as_c_ptr(),
                filled.size, C.CT_OP_ADD, 0,
            )
            return filled

        left = operand(x)
        right = operand(y)
        out = CTensor(self.data.shape)
        C.where_double(
            self.data.as_c_ptr(), left.as_c_ptr(), right.as_c_ptr(),
            out.as_c_ptr(), out.size,
        )
        return out

    def select_by_indices_(self, tensor: CTensor, indices_dim0: Any, indices_dim1: Any) -> Any:
        # ########## STUB: CTensorOperations.select_by_indices ##########
        # PURPOSE: Gather elements from ``tensor`` using two index arrays.
        # EXPECTED BEHAVIOR: Return a 1D CTensor of selected values.
        # INPUTS: ``tensor`` CTensor, ``indices_dim0`` list, ``indices_dim1`` list.
        # OUTPUTS: CTensor with values from ``tensor[indices_dim0[i], indices_dim1[i]]``.
        # KEY ASSUMPTIONS/DEPENDENCIES: Requires stride calculations.
        # TODO:
        #   - Implement efficient index selection.
        # NOTES: Complex indexing left for future work.
        # ############################################################
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))

        rows = list(indices_dim0)

        if isinstance(indices_dim1, slice):
            start, stop, step = indices_dim1.indices(tensor.shape[1])
            cols_range = list(range(start, stop, step))
            row_arr = [r for r in rows for _ in cols_range]
            col_arr = cols_range * len(rows)
            out_shape = (len(rows), len(cols_range))
        else:
            cols = [indices_dim1] * len(rows) if isinstance(indices_dim1, int) else list(indices_dim1)
            if len(rows) != len(cols):
                raise ValueError("Index lists must have same length for element-wise selection")
            row_arr = rows
            col_arr = cols
            out_shape = (len(cols),) if not isinstance(indices_dim1, int) else (len(rows),)

        row_buf = ffi.new("int[]", row_arr)
        col_buf = ffi.new("int[]", col_arr)
        n_pairs = len(row_arr)
        out_buf = ffi.new("double[]", n_pairs)
        C.gather_pairs_2d(tensor.as_c_ptr(), row_buf, col_buf, out_buf, n_pairs, tensor.shape[1])

        return CTensor(out_shape, out_buf)

    def log_softmax_(self, dim: int) -> Any:
        """Compute log softmax along ``dim`` using C routines."""
        tensor = self.data
        ndim = len(tensor.shape)
        if dim < 0:
            dim += ndim
        if dim < 0 or dim >= ndim:
            raise ValueError("dim out of range")
        c_shape = ffi.new("int[]", list(tensor.shape))
        out = CTensor(tensor.shape)
        if ndim == 1:
            C.log_softmax_1d(tensor.as_c_ptr(), out.as_c_ptr(), tensor.size)
        else:
            C.log_softmax_dim(tensor.as_c_ptr(), c_shape, ndim, dim, out.as_c_ptr())
        return out

    def pad_(self, pad: Tuple[int, ...], value: float = 0) -> Any:
        """Pad ``tensor`` with ``value`` according to ``pad`` specification."""
        tensor = self.data

        if len(pad) % 2 != 0:
            raise ValueError("Padding length must be even.")

        dims = len(tensor.shape)
        num_pad_dims = len(pad) // 2
        if num_pad_dims > dims:
            raise ValueError(
                "Padding tuple length implies padding more dimensions than tensor has."
            )

        left = [0] * dims
        right = [0] * dims
        for i in range(num_pad_dims):
            left[dims - num_pad_dims + i] = int(pad[-2 * (i + 1)])
            right[dims - num_pad_dims + i] = int(pad[-2 * (i + 1) + 1])

        new_shape = [
            tensor.shape[i] + left[i] + right[i] for i in range(dims)
        ]

        out = CTensor(tuple(new_shape))
        shape_c = ffi.new("int[]", list(tensor.shape))
        new_shape_c = ffi.new("int[]", new_shape)
        left_c = ffi.new("int[]", left)
        C.pad_double_nd(
            tensor.as_c_ptr(),
            out.as_c_ptr(),
            shape_c,
            new_shape_c,
            left_c,
            dims,
            float(value),
        )
        return out

    def topk_(self, k: int, dim: int) -> Tuple[Any, Any]:
        tensor = self.data
        shape = tensor.shape
        ndim = len(shape)
        if dim < 0:
            dim += ndim
        if dim < 0 or dim >= ndim:
            raise ValueError("dim out of range")

        if k > shape[dim]:
            k = shape[dim]

        c_shape = ffi.new("int[]", list(shape))
        out_shape = list(shape)
        out_shape[dim] = k
        values = CTensor(tuple(out_shape))
        indices = CTensor(tuple(out_shape))
        C.topk_double_dim(
            tensor.as_c_ptr(),
            c_shape,
            ndim,
            dim,
            k,
            indices.as_c_ptr(),
            values.as_c_ptr(),
        )
        return values, indices

    def repeat_interleave_(
        self, repeats: int = 1, dim: Optional[int] = None
    ) -> Any:
        tensor = self.data
        if repeats < 0:
            raise ValueError("repeats must be non-negative")
        if dim is None:
            tensor = CTensor((tensor.size,), tensor.buffer)
            dim = 0
        else:
            dim %= len(tensor.shape)
        shape = list(tensor.shape)
        shape[dim] *= repeats
        out = CTensor(tuple(shape))
        C.repeat_interleave_double(
            tensor.as_c_ptr(), out.as_c_ptr(),
            ffi.new("int[]", tensor.shape), len(tensor.shape),
            dim, repeats,
        )
        return out

    def repeat_(self, repeats: Any = None, dim: int = 0) -> Any:
        if repeats is None:
            raise ValueError("repeats must be specified")
        input_shape = self.data.shape
        if isinstance(repeats, int):
            factors = [1] * len(input_shape)
            factors[dim % len(input_shape)] = repeats
        else:
            factors = list(repeats)
            if len(factors) < len(input_shape):
                factors = [1] * (len(input_shape) - len(factors)) + factors
            if len(factors) != len(input_shape):
                raise ValueError("repeat rank must match tensor rank")
        if any(factor < 0 for factor in factors):
            raise ValueError("repeat factors must be non-negative")
        output_shape = tuple(
            size * factor for size, factor in zip(input_shape, factors)
        )
        out = CTensor(output_shape)
        C.tile_double(
            self.data.as_c_ptr(), out.as_c_ptr(),
            ffi.new("int[]", input_shape),
            ffi.new("int[]", output_shape), len(input_shape),
        )
        return out

    def assign_at_indices_(
        self,
        tensor_to_modify: CTensor,
        indices_dim0: Any,
        indices_dim1: Any,
        values_to_assign: Any,
    ) -> None:
        # ########## STUB: CTensorOperations.assign_at_indices ##########
        # PURPOSE: In-place assignment into ``tensor_to_modify`` at specified indices.
        # EXPECTED BEHAVIOR: Modifies tensor values according to index lists.
        # INPUTS: target CTensor, two index lists, values list.
        # OUTPUTS: None (in-place modification).
        # KEY ASSUMPTIONS/DEPENDENCIES: Requires index math support.
        # TODO:
        #   - Implement multi-dimensional indexing.
        # NOTES: Stub pending full CTensor infrastructure.
        # ############################################################
        raise NotImplementedError("assign_at_indices not implemented for C backend")

    def increment_at_indices_(self, mask: Any):
        mask = mask.data if isinstance(mask, AbstractTensor) else mask
        if not isinstance(mask, CTensor):
            mask = CTensor.from_list(mask, _get_shape(mask))
        if mask.shape != self.data.shape:
            raise ValueError("mask shape must match tensor shape")
        C.increment_mask_double(
            self.data.as_c_ptr(), mask.as_c_ptr(), self.data.size
        )
        return self.data

    def boolean_mask_select_(self, mask: Any) -> Any:
        mask = mask.data if isinstance(mask, AbstractTensor) else mask
        if not isinstance(mask, CTensor):
            mask = CTensor.from_list(mask, _get_shape(mask))
        if mask.shape != self.data.shape:
            raise ValueError("mask shape must match tensor shape")
        count = C.count_true_double(mask.as_c_ptr(), mask.size)
        out = CTensor((count,))
        C.mask_select_double(
            self.data.as_c_ptr(), mask.as_c_ptr(), out.as_c_ptr(),
            self.data.size,
        )
        return out

    def index_select_(self, dim: int, indices: Any) -> Any:
        indices = (
            indices.tolist()
            if isinstance(indices, AbstractTensor)
            else indices.tolist()
            if isinstance(indices, CTensor)
            else list(indices)
        )
        indices = [int(index) for index in indices]
        dim %= len(self.data.shape)
        if any(
            index < 0 or index >= self.data.shape[dim]
            for index in indices
        ):
            raise IndexError("index_select index out of range")
        shape = list(self.data.shape)
        shape[dim] = len(indices)
        out = CTensor(tuple(shape))
        C.index_select_double(
            self.data.as_c_ptr(), out.as_c_ptr(),
            ffi.new("int[]", self.data.shape), len(self.data.shape), dim,
            ffi.new("int[]", indices), len(indices),
        )
        return out

    def interpolate_(self, tensor: CTensor, size: Tuple[int, ...]) -> Any:
        # ########## STUB: CTensorOperations.interpolate ##########
        # PURPOSE: Resize ``tensor`` to ``size`` using linear interpolation.
        # EXPECTED BEHAVIOR: Perform dimension-wise interpolation similar to
        #     other backends.
        # INPUTS: CTensor and target ``size`` tuple.
        # OUTPUTS: CTensor resized to ``size``.
        # KEY ASSUMPTIONS/DEPENDENCIES: Would require new C routines for
        #     interpolation and memory allocation.
        # TODO:
        #   - Add C functions to compute interpolated values.
        # NOTES: Not yet implemented.
        # ############################################################
        raise NotImplementedError("interpolate not implemented for C backend")

    def stack_(self, tensors: list, dim: int = 0) -> Any:
        if not tensors:
            raise ValueError("tensors list cannot be empty")
        tensors = [
            tensor.data if isinstance(tensor, AbstractTensor) else tensor
            for tensor in tensors
        ]
        c_tensors = [
            t if isinstance(t, CTensor) else CTensor.from_list(t, _get_shape(t))
            for t in tensors
        ]
        base_shape = c_tensors[0].shape
        for t in c_tensors:
            if t.shape != base_shape:
                raise ValueError("All tensors must have the same shape")
        ndim = len(base_shape)
        if dim < 0:
            dim += ndim + 1
        if dim < 0 or dim > ndim:
            raise ValueError("dim out of range")
        new_shape = base_shape[:dim] + (len(c_tensors),) + base_shape[dim:]
        out = CTensor(new_shape)
        shape_c = ffi.new("int[]", list(base_shape))
        tensor_ptrs = ffi.new("double*[]", [t.as_c_ptr() for t in c_tensors])
        C.stack_double(tensor_ptrs, len(c_tensors), shape_c, ndim, dim, out.as_c_ptr())
        return out

    def cat_(self, tensors: list, dim: int = 0) -> Any:
        if not tensors:
            raise ValueError("tensors list cannot be empty")
        tensors = [
            tensor.data if isinstance(tensor, AbstractTensor) else tensor
            for tensor in tensors
        ]
        c_tensors = [
            t if isinstance(t, CTensor) else CTensor.from_list(t, _get_shape(t))
            for t in tensors
        ]
        first_shape = list(c_tensors[0].shape)
        ndim = len(first_shape)
        if dim < 0:
            dim += ndim
        if dim < 0 or dim >= ndim:
            raise ValueError("dim out of range")
        for t in c_tensors:
            if len(t.shape) != ndim:
                raise ValueError("All tensors must have the same rank")
            for d in range(ndim):
                if d == dim:
                    continue
                if t.shape[d] != first_shape[d]:
                    raise ValueError("Non-concat dimensions must match")
        dim_sizes = [t.shape[dim] for t in c_tensors]
        out_shape = first_shape[:]
        out_shape[dim] = sum(dim_sizes)
        out = CTensor(tuple(out_shape))
        tensor_ptrs = ffi.new("double*[]", [t.as_c_ptr() for t in c_tensors])
        dim_sizes_c = ffi.new("int[]", dim_sizes)
        shape_c = ffi.new("int[]", first_shape)
        C.cat_double(tensor_ptrs, dim_sizes_c, len(c_tensors), shape_c, ndim, dim, out.as_c_ptr())
        return out

    def unravel_index_(self, shape):
        raise NotImplementedError(
            "unravel_index not implemented for C backend"
        )

    def get_device_(self, tensor: CTensor = None) -> str:
        return "cpu_cffi"

    def get_dtype_(self, tensor: CTensor = None) -> Any:
        return float

    def item_(self, tensor: CTensor = None) -> Any:
        if tensor is None:
            tensor = self.data
        if tensor.size == 1:
            return tensor.buffer[0]
        raise ValueError("Tensor has more than one element")

    def nbytes_(self) -> int:
        return self.data.size * ffi.sizeof("double")

    def long_cast_(self, tensor: CTensor) -> CTensor:
        t = CTensor(tensor.shape)
        for i in range(t.size):
            t.buffer[i] = int(tensor.buffer[i])
        return t

    def save_(self, tensor: CTensor, filepath: str) -> None:
        with open(filepath, "wb") as f:
            f.write(ffi.buffer(tensor.buffer, tensor.size * 8))

    def load_(self, filepath: str, dtype: Any, device: Any) -> CTensor:
        with open(filepath, "rb") as f:
            data = f.read()
        n = len(data) // 8
        buf = ffi.new("double[]", n)
        ffi.memmove(buf, data, len(data))
        # You must provide shape info externally!
        return CTensor((n,), buf)

    @property
    def long_dtype_(self) -> Any:
        return int

    @property
    def bool_dtype_(self) -> Any:
        return bool

    @property
    def float_dtype_(self) -> Any:
        return float

    tensor_type_ = CTensor

    # Implementation hooks required by AbstractTensor
    def get_shape(self) -> tuple[int, ...]:
        t = self.data
        if not isinstance(t, CTensor):
            t = CTensor.from_list(t, _get_shape(t))
        return t.shape

    def get_ndims(self) -> int:
        return len(self.get_shape())

    def to_dtype_(self, dtype: str = "float", tensor=None):
        """Convert ``tensor`` data type using C helpers."""
        # ########## STUB: CTensorOperations.to_dtype_ ##########
        # PURPOSE: Convert CTensor data to specified dtype.
        # EXPECTED BEHAVIOR: Return new CTensor with values cast to dtype.
        # INPUTS: CTensor instance and dtype string.
        # OUTPUTS: CTensor with cast data as contiguous byte array.
        # KEY ASSUMPTIONS/DEPENDENCIES: Only primitive dtypes ``float`` or ``int`` supported.
        # TODO:
        #   - Extend to additional dtypes and integrate with other operations.
        # ############################################################
        if isinstance(dtype, CTensor):
            dtype, tensor = tensor, dtype
        if tensor is None:
            tensor = self.data
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))

        if isinstance(dtype, type):
            dtype = dtype.__name__
        dtype = str(dtype).lower()
        if dtype in {"int", "int32", "int64", "long"}:
            buf = ffi.new("double[]", tensor.size)
            C.cast_double_to_int_values(tensor.as_c_ptr(), buf, tensor.size)
        elif dtype in {"float64", "double"}:
            buf = ffi.new("double[]", tensor.size)
            ffi.memmove(
                buf, tensor.as_c_ptr(),
                tensor.size * ffi.sizeof("double"),
            )
        elif dtype in {"float", "float32"}:
            buf = ffi.new("double[]", tensor.size)
            C.cast_double_to_float_values(
                tensor.as_c_ptr(), buf, tensor.size
            )
        else:
            raise ValueError("C backend supports numeric float/int casts only")

        return CTensor(tensor.shape, buf)

    @staticmethod
    def test() -> None:
        """Simple self-check calling ``sqrt`` from ``libm``."""
        ops = CTensorOperations()
        result = ops.sqrt([4.0, 9.0])
        assert [round(x, 1) for x in result.tolist()] == [2.0, 3.0]


register_backend("c", CTensorOperations)

