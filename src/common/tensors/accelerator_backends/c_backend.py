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
from ..abstraction import AbstractTensor, _get_shape, _flatten


ffi = FFI()
ffi.cdef("""
    void add_double(const double* a, const double* b, double* out, int n);
    void sub_double(const double* a, const double* b, double* out, int n);
    void mul_double(const double* a, const double* b, double* out, int n);
    void div_double(const double* a, const double* b, double* out, int n);
    void pow_double(const double* a, const double* b, double* out, int n);
    void mod_double(const double* a, const double* b, double* out, int n);
    void floordiv_double(const double* a, const double* b, double* out, int n);
    void matmul_double(const double* a, const double* b, double* out, int m, int n, int p);
    // Scalar ops
    void add_scalar(const double* a, double b, double* out, int n);
    void subtract_const(const double* a, double b, double* out, int n);
    void rsubtract_const(const double* a, double b, double* out, int n);
    void mul_scalar(const double* a, double b, double* out, int n);
    void divide_const(const double* a, double b, double* out, int n);
    void rdivide_const(const double* a, double b, double* out, int n);
    void pow_scalar(const double* a, double b, double* out, int n);
    void rpow_scalar(const double* a, double b, double* out, int n);
    void mod_scalar(const double* a, double b, double* out, int n);
    void rmod_scalar(const double* a, double b, double* out, int n);
    void floor_div_const(const double* a, double b, double* out, int n);
    void rfloor_div_const(const double* a, double b, double* out, int n);
    void sqrt_double(const double* a, double* out, int n);
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

    void cast_double_to_int(const double* a, int* out, int n);
    void cast_double_to_float(const double* a, float* out, int n);
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
    C = ffi.verify(C_SOURCE)

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
        """Operate on ``CTensor`` objects or scalars."""
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
                raise ValueError("Shape mismatch")
            out = CTensor(left.shape)
            n = left.size
            if op in ('add', 'iadd'):
                C.add_double(left.as_c_ptr(), right.as_c_ptr(), out.as_c_ptr(), n)
            elif op in ('sub', 'isub'):
                C.sub_double(left.as_c_ptr(), right.as_c_ptr(), out.as_c_ptr(), n)
            elif op in ('mul', 'imul'):
                C.mul_double(left.as_c_ptr(), right.as_c_ptr(), out.as_c_ptr(), n)
            elif op in ('truediv', 'itruediv'):
                C.div_double(left.as_c_ptr(), right.as_c_ptr(), out.as_c_ptr(), n)
            elif op in ('pow', 'ipow'):
                C.pow_double(left.as_c_ptr(), right.as_c_ptr(), out.as_c_ptr(), n)
            elif op in ('mod', 'imod'):
                C.mod_double(left.as_c_ptr(), right.as_c_ptr(), out.as_c_ptr(), n)
            elif op in ('floordiv', 'ifloordiv'):
                C.floordiv_double(left.as_c_ptr(), right.as_c_ptr(), out.as_c_ptr(), n)
            else:
                raise NotImplementedError(f"Operator {op} not implemented for C backend.")
            return out
        elif isinstance(left, CTensor) and isinstance(right, (int, float)):
            out = CTensor(left.shape)
            n = left.size
            val = float(right)
            if op in ('add', 'iadd'):
                C.add_scalar(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op == 'radd':
                C.add_scalar(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op in ('sub', 'isub'):
                C.subtract_const(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op == 'rsub':
                C.rsubtract_const(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op in ('mul', 'imul'):
                C.mul_scalar(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op == 'rmul':
                C.mul_scalar(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op in ('truediv', 'itruediv'):
                C.divide_const(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op == 'rtruediv':
                C.rdivide_const(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op in ('pow', 'ipow'):
                C.pow_scalar(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op == 'rpow':
                C.rpow_scalar(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op in ('mod', 'imod'):
                C.mod_scalar(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op == 'rmod':
                C.rmod_scalar(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op in ('floordiv', 'ifloordiv'):
                C.floor_div_const(left.as_c_ptr(), val, out.as_c_ptr(), n)
            elif op == 'rfloordiv':
                C.rfloor_div_const(left.as_c_ptr(), val, out.as_c_ptr(), n)
            else:
                raise NotImplementedError(f"Operator {op} not implemented for C backend.")
            return out
        else:
            raise TypeError("CTensorOperations only supports CTensor or scalar operands.")

    # Creation ops
    def full_(self, size: Tuple[int, ...], fill_value: Any, dtype: Any, device: Any):
        t = CTensor(size)
        for i in range(t.size):
            t.buffer[i] = float(fill_value)
        return t

    def zeros_(self, size: Tuple[int, ...], dtype: Any, device: Any):
        return self.full(size, 0.0, dtype, device)

    def clone_(self, tensor: CTensor) -> CTensor:
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
        out = CTensor(tensor.shape)
        C.pow_scalar(tensor.as_c_ptr(), float(exponent), out.as_c_ptr(), tensor.size)
        return out

    def sqrt_(self, tensor: Any) -> CTensor:
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))
        out = CTensor(tensor.shape)
        C.sqrt_double(tensor.as_c_ptr(), out.as_c_ptr(), tensor.size)
        return out

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

    def shape_(self, tensor: CTensor) -> Tuple[int, ...]:
        return tensor.shape

    def numel_(self, tensor: CTensor) -> int:
        return tensor.size

    def __trunc__(self):
        import math
        tensor = self.data
        if tensor.size != 1:
            raise TypeError("Only scalar tensors can be converted to int")
        return int(math.trunc(tensor.buffer[0]))

    def mean_(self, tensor: Any, dim: Optional[int] = None) -> Any:
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))
        values = _flatten(tensor.tolist())
        if dim is None:
            return sum(values) / len(values) if values else 0.0

        shape = tensor.shape
        if dim < 0:
            dim += len(shape)
        if dim < 0 or dim >= len(shape):
            raise ValueError("dim out of range")

        out_shape = shape[:dim] + shape[dim + 1 :]
        out = CTensor(out_shape if out_shape else ())
        shape_arr = ffi.new("int[]", list(shape))
        C.mean_dim(tensor.as_c_ptr(), out.as_c_ptr(), shape_arr, len(shape), dim)
        if not out_shape:
            return out.buffer[0]
        return out

    def less_(self, tensor: Any, value: Any) -> list:
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))
        return [tensor.buffer[i] < value for i in range(tensor.size)]

    def view_flat_(self, tensor: Any) -> list:
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))
        return _flatten(tensor.tolist())

    def tolist_(self, tensor: Any) -> list:
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))
        return tensor.tolist()

    def clamp_(
        self,
        tensor: Any,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> CTensor:
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))
        out = CTensor(tensor.shape)
        for i in range(tensor.size):
            val = tensor.buffer[i]
            if min_val is not None and val < min_val:
                val = min_val
            if max_val is not None and val > max_val:
                val = max_val
            out.buffer[i] = val
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

    def log_softmax_(self, tensor: CTensor, dim: int) -> Any:
        """Compute log softmax along ``dim`` using C routines."""
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))
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

    def pad_(self, tensor: CTensor, pad: Tuple[int, ...], value: float = 0) -> Any:
        """Pad ``tensor`` with ``value`` according to ``pad`` specification."""
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))

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

    def topk_(self, tensor: CTensor, k: int, dim: int) -> Tuple[Any, Any]:
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

    def repeat_interleave_(self, tensor: CTensor, repeats: int, dim: Optional[int] = None) -> Any:
        # ########## STUB: CTensorOperations.repeat_interleave ##########
        # PURPOSE: Repeat each element in ``tensor`` ``repeats`` times along ``dim``.
        # EXPECTED BEHAVIOR: New CTensor with expanded dimension.
        # INPUTS: CTensor, repeats int, optional dimension.
        # OUTPUTS: CTensor with repeated values.
        # KEY ASSUMPTIONS/DEPENDENCIES: Requires advanced slicing helpers.
        # TODO:
        #   - Implement axis-aware repetition.
        # NOTES: Only full flattening supported currently.
        # ############################################################
        raise NotImplementedError("repeat_interleave not implemented for C backend")

    def repeat_(self, repeats: Any = None, dim: int = 0) -> Any:
        """Repeat tensor along ``dim`` ``repeats`` times (stub)."""
        raise NotImplementedError("repeat not implemented for C backend")

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

    def increment_at_indices_(self, tensor_to_modify: CTensor, mask: Any) -> None:
        # ########## STUB: CTensorOperations.increment_at_indices ##########
        # PURPOSE: Increment elements of ``tensor_to_modify`` where ``mask`` is True.
        # EXPECTED BEHAVIOR: Element-wise increment using a boolean mask.
        # INPUTS: target CTensor, boolean mask list.
        # OUTPUTS: None (in-place update).
        # KEY ASSUMPTIONS/DEPENDENCIES: Requires broadcasting rules.
        # TODO:
        #   - Implement efficient masked increment.
        # NOTES: Stub to signal incomplete feature.
        # ############################################################
        raise NotImplementedError("increment_at_indices not implemented for C backend")

    def boolean_mask_select_(self, tensor: CTensor, mask: Any) -> Any:
        # ########## STUB: CTensorOperations.boolean_mask_select ##########
        # PURPOSE: Select values from ``tensor`` where ``mask`` is True.
        # EXPECTED BEHAVIOR: Return CTensor or list of filtered values.
        # INPUTS: CTensor, mask list.
        # OUTPUTS: CTensor containing selected elements.
        # KEY ASSUMPTIONS/DEPENDENCIES: Requires shape-aware masking.
        # TODO:
        #   - Implement boolean masking for CTensors.
        # NOTES: Not yet supported in C backend.
        # ############################################################
        raise NotImplementedError("boolean_mask_select not implemented for C backend")

    def index_select_(self, tensor: CTensor, dim: int, indices: Any) -> Any:
        # ########## STUB: CTensorOperations.index_select ##########
        # PURPOSE: Select entries along ``dim`` using ``indices``.
        # EXPECTED BEHAVIOR: Mirror numpy.take along the specified dimension.
        # INPUTS: CTensor, dimension index, index list.
        # OUTPUTS: CTensor with gathered values.
        # KEY ASSUMPTIONS/DEPENDENCIES: Requires advanced indexing support.
        # TODO:
        #   - Implement general index selection.
        # NOTES: Current design does not provide necessary helpers.
        # ############################################################
        raise NotImplementedError("index_select not implemented for C backend")

    def argmin_(self, tensor: CTensor, dim: Optional[int] = None) -> Any:
        # ########## STUB: CTensorOperations.argmin ##########
        # PURPOSE: Placeholder for argmin across dimensions in the C backend.
        # EXPECTED BEHAVIOR: Should mirror numpy.argmin with optional axis.
        # INPUTS: CTensor to examine and optional dimension ``dim``.
        # OUTPUTS: Integer index or CTensor of indices.
        # KEY ASSUMPTIONS/DEPENDENCIES: Requires C helpers for reduction.
        # TODO:
        #   - Implement dimension-aware minimum search.
        # NOTES: Current backend lacks the functionality to compute argmin.
        # ############################################################
        raise NotImplementedError("argmin not implemented for C backend")

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
        # ########## STUB: CTensorOperations.stack ##########
        # PURPOSE: Concatenate a sequence of CTensors along a new dimension.
        # EXPECTED BEHAVIOR: Should return a CTensor representing ``tensors``
        #     stacked along ``dim``.
        # INPUTS: list of CTensors, dimension index.
        # OUTPUTS: New CTensor with increased rank.
        # KEY ASSUMPTIONS/DEPENDENCIES: Requires advanced shape handling.
        # TODO:
        #   - Implement shape validation and memory allocation logic.
        # NOTES: Current C backend lacks generic tensor helpers.
        # ############################################################
        raise NotImplementedError("stack not implemented for C backend")

    def cat_(self, tensors: list, dim: int = 0) -> Any:
        # ########## STUB: CTensorOperations.cat ##########
        # PURPOSE: Concatenate CTensors along an existing dimension.
        # EXPECTED BEHAVIOR: Should join ``tensors`` on ``dim`` similar to
        #     numpy.concatenate.
        # INPUTS: list of CTensors, dimension index.
        # OUTPUTS: New CTensor with combined size.
        # KEY ASSUMPTIONS/DEPENDENCIES: Requires complex stride math.
        # TODO:
        #   - Implement concatenation across arbitrary dimensions.
        # NOTES: Placeholder until a more complete C tensor API exists.
        # ############################################################
        raise NotImplementedError("cat not implemented for C backend")

    def unravel_index_(self, shape):
        raise NotImplementedError(
            "unravel_index not implemented for C backend"
        )

    def get_device_(self, tensor: CTensor) -> str:
        return "cpu_cffi"

    def get_dtype_(self, tensor: CTensor) -> Any:
        return float

    def item_(self, tensor: CTensor) -> Any:
        if tensor.size == 1:
            return tensor.buffer[0]
        raise ValueError("Tensor has more than one element")

    def max_(self, tensor: CTensor) -> float:
        return max(tensor.tolist())

    def long_cast_(self, tensor: CTensor) -> CTensor:
        t = CTensor(tensor.shape)
        for i in range(t.size):
            t.buffer[i] = int(tensor.buffer[i])
        return t

    def not_equal_(self, tensor1: CTensor, tensor2: CTensor) -> list:
        return [tensor1.buffer[i] != tensor2.buffer[i] for i in range(tensor1.size)]

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

    @property
    def tensor_type_(self) -> type:
        return CTensor

    # Implementation hooks required by AbstractTensor
    def get_shape(self) -> tuple[int, ...]:
        t = self.data
        if not isinstance(t, CTensor):
            t = CTensor.from_list(t, _get_shape(t))
        return t.shape

    def get_ndims(self) -> int:
        return len(self.get_shape())

    def to_dtype_(self, tensor, dtype: str = "float"):
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
        if not isinstance(tensor, CTensor):
            tensor = CTensor.from_list(tensor, _get_shape(tensor))

        dtype = dtype.lower()
        if dtype == "int":
            buf = ffi.new("int[]", tensor.size)
            C.cast_double_to_int(tensor.as_c_ptr(), buf, tensor.size)
        elif dtype == "float":
            buf = ffi.new("double[]", tensor.size)
            C.cast_double_to_float(tensor.as_c_ptr(), buf, tensor.size)
        else:
            raise ValueError("dtype must be 'float' or 'int'")

        return CTensor(tensor.shape, buf)

    @staticmethod
    def test() -> None:
        """Simple self-check calling ``sqrt`` from ``libm``."""
        ops = CTensorOperations()
        result = ops.sqrt([4.0, 9.0])
        assert [round(x, 1) for x in result.tolist()] == [2.0, 3.0]

