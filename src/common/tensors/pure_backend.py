"""Pure Python implementation of :class:`AbstractTensor`."""

from __future__ import annotations

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

from typing import Any, Tuple, Optional, List
import math
import json

from .abstraction import _get_shape, _flatten, register_backend



from .abstraction import AbstractTensor

def _to_tuple2(x):
    return (x, x) if isinstance(x, int) else x

class PurePythonTensorOperations(AbstractTensor):
    def allclose_(self, other, rtol=1e-5, atol=1e-8, equal_nan=False):
        import math
        from .abstraction import _flatten, _get_shape
        if not isinstance(other, type(self)):
            other = type(self)(other)
        a = _flatten(self.data)
        b = _flatten(other.data)
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if math.isnan(x) or math.isnan(y):
                if not equal_nan or not (math.isnan(x) and math.isnan(y)):
                    return False
            elif not math.isclose(x, y, rel_tol=rtol, abs_tol=atol):
                return False
        return True
    def isfinite_(self):
        import math
        from .abstraction import _flatten, _get_shape
        data = self.data
        shape = _get_shape(data)
        if len(shape) == 1:
            return [math.isfinite(x) for x in data]
        elif len(shape) == 2:
            return [[math.isfinite(x) for x in row] for row in data]
        else:
            raise NotImplementedError("isfinite_ only implemented for 1D/2D in pure backend")
    def all_(self, dim=None):
        from .abstraction import _flatten
        if dim is None:
            return all(_flatten(self.data))
        # For 2D, all along axis 0 or 1
        data = self.data
        if dim == 0:
            return [all(row[i] for row in data) for i in range(len(data[0]))]
        elif dim == 1:
            return [all(row) for row in data]
        else:
            raise NotImplementedError("all_ only implemented for 1D/2D and dim 0/1 in pure backend")
    def isnan_(self):
        import math
        from .abstraction import _flatten, _get_shape
        data = self.data
        shape = _get_shape(data)
        if len(shape) == 1:
            return [math.isnan(x) for x in data]
        elif len(shape) == 2:
            return [[math.isnan(x) for x in row] for row in data]
        else:
            raise NotImplementedError("isnan_ only implemented for 1D/2D in pure backend")

    def isinf_(self):
        import math
        from .abstraction import _flatten, _get_shape
        data = self.data
        shape = _get_shape(data)
        if len(shape) == 1:
            return [math.isinf(x) for x in data]
        elif len(shape) == 2:
            return [[math.isinf(x) for x in row] for row in data]
        else:
            raise NotImplementedError("isinf_ only implemented for 1D/2D in pure backend")
    def nonzero_(self, as_tuple: bool = False):
        from .abstraction import _flatten, _get_shape
        # Only works for 1D/2D for simplicity
        data = self.data
        shape = _get_shape(data)
        indices = []
        if len(shape) == 1:
            for i, v in enumerate(data):
                if v:
                    indices.append((i,))
        elif len(shape) == 2:
            for i, row in enumerate(data):
                for j, v in enumerate(row):
                    if v:
                        indices.append((i, j))
        else:
            raise NotImplementedError("nonzero_ only implemented for 1D/2D in pure backend")
        if as_tuple:
            if not indices:
                return tuple([] for _ in range(len(shape)))
            return tuple([tuple(idx[dim] for idx in indices) for dim in range(len(shape))])
        return indices
    def any_(self, dim=None):
        from .abstraction import _flatten
        return any(_flatten(self.data)) if dim is None else any(_flatten(row[dim] for row in self.data))
    def max_(self, dim: Optional[int] = None, keepdim: bool = False) -> Any:
        data = self.data
        def _max(lst):
            flat = _flatten(lst)
            return max(flat) if flat else 0.0
        def reduce_dim(lst, d):
            if d == 0:
                if not isinstance(lst[0], list):
                    m = max(lst)
                    return [m] if keepdim else m
                cols = len(lst[0])
                result = []
                for i in range(cols):
                    col = [row[i] for row in lst]
                    m = max(col)
                    result.append(m)
                if keepdim:
                    return [result]
                return result
            else:
                return [reduce_dim(row, d-1) for row in lst]
        if dim is None:
            m = _max(data)
            if keepdim:
                shape = _get_shape(data)
                for _ in range(len(shape)):
                    m = [m]
            return m
        return reduce_dim(data, dim)

    def argmax_(self, dim: Optional[int] = None, keepdim: bool = False) -> Any:
        data = self.data
        def _argmax(lst):
            flat = _flatten(lst)
            return flat.index(max(flat)) if flat else 0
        def reduce_dim(lst, d):
            if d == 0:
                if not isinstance(lst[0], list):
                    idx = lst.index(max(lst))
                    return [idx] if keepdim else idx
                cols = len(lst[0])
                result = []
                for i in range(cols):
                    col = [row[i] for row in lst]
                    idx = col.index(max(col))
                    result.append(idx)
                if keepdim:
                    return [result]
                return result
            else:
                return [reduce_dim(row, d-1) for row in lst]
        if dim is None:
            idx = _argmax(data)
            if keepdim:
                shape = _get_shape(data)
                for _ in range(len(shape)):
                    idx = [idx]
            return idx
        return reduce_dim(data, dim)

    def unravel_index_(self, shape: Tuple[int, ...]):
        idx = self.data
        if isinstance(idx, list):
            idx = idx[0]
        coords = []
        for dim in reversed(shape):
            coords.append(idx % dim)
            idx //= dim
        return tuple(reversed(coords))
    """Educational tensor ops using nested Python lists."""

    def __init__(self, track_time: bool = False):
        super().__init__(track_time=track_time)
        # ########## STUB: PurePythonTensorOperations.__init__ ##########
        # PURPOSE: Placeholder for any future initialization logic needed for
        #          the pure Python backend.
        # EXPECTED BEHAVIOR: Should set up attributes or configuration values
        #          when this backend requires them.
        # INPUTS: None at present.
        # OUTPUTS: None.
        # KEY ASSUMPTIONS/DEPENDENCIES: Currently no dependencies beyond the
        #          abstract interface.
        # TODO:
        #   - Add configurable parameters if performance tuning becomes
        #     relevant.
        # NOTES: Implementation intentionally empty.
        # ###############################################################
        pass

    def _apply_operator__(self, op: str, left: Any, right: Any):
        """Dispatch basic arithmetic for nested lists."""
        from .abstraction import AbstractTensor
        left = self._AbstractTensor__unwrap(left) if isinstance(left, AbstractTensor) else left
        right = self._AbstractTensor__unwrap(right) if isinstance(right, AbstractTensor) else right
        if right is None:
            if isinstance(left, list):
                return self._elementwise_unary(op, left)
            return self._apply_scalar_unary(op, left)
        if op in {"matmul", "rmatmul", "imatmul"}:
            a, b = (left, right) if op != "rmatmul" else (right, left)
            return self._matmul(a, b)
        if isinstance(right, list) and isinstance(left, list):
            return self._elementwise_op(op, left, right)
        if isinstance(right, list):
            return self._elementwise_op_scalar(
                op, right, left
            )  # right is list, treat left as scalar
        if isinstance(left, list):
            return self._elementwise_op_scalar(op, left, right)
        return self._apply_scalar_op(op, left, right)

    def _elementwise_unary(self, op: str, a):
        if not isinstance(a, list):
            return self._apply_scalar_unary(op, a)
        return [self._elementwise_unary(op, ai) for ai in a]

    def _apply_scalar_unary(self, op: str, x):
        if op == "neg":
            return -x
        if op == "abs":
            return abs(x)
        if op == "invert":
            return ~x
        if op == "sin":
            return math.sin(x)
        if op == "cos":
            return math.cos(x)
        if op == "tan":
            return math.tan(x)
        if op == "asin":
            return math.asin(x)
        if op == "acos":
            return math.acos(x)
        if op == "atan":
            return math.atan(x)
        if op == "sinh":
            return math.sinh(x)
        if op == "cosh":
            return math.cosh(x)
        if op == "tanh":
            return math.tanh(x)
        if op == "asinh":
            return math.asinh(x)
        if op == "acosh":
            return math.acosh(x)
        if op == "atanh":
            return math.atanh(x)
        raise NotImplementedError(f"Operator {op} not implemented for pure Python backend.")

    def _elementwise_op(self, op: str, a, b):
        # Recursively apply op to nested lists
        if not isinstance(a, list) and not isinstance(b, list):
            return self._apply_scalar_op(op, a, b)
        return [self._elementwise_op(op, ai, bi) for ai, bi in zip(a, b)]

    def _elementwise_op_scalar(self, op: str, a, scalar):
        if not isinstance(a, list):
            return self._apply_scalar_op(op, a, scalar)
        return [self._elementwise_op_scalar(op, ai, scalar) for ai in a]

    def _apply_scalar_op(self, op: str, x, y):
        if op in ("add", "radd", "iadd"):
            return x + y
        if op in ("sub", "rsub", "isub"):
            return x - y
        if op in ("mul", "rmul", "imul"):
            return x * y
        if op in ("truediv", "rtruediv", "itruediv"):
            return x / y
        if op in ("floordiv", "rfloordiv", "ifloordiv"):
            return x // y
        if op in ("mod", "rmod", "imod"):
            return x % y
        if op in ("pow", "rpow", "ipow"):
            return x**y
        if op in ("matmul", "rmatmul", "imatmul"):
            return self._matmul(x, y)
        raise NotImplementedError(
            f"Operator {op} not implemented for pure Python backend."
        )

    def _matmul(self, a, b):
        """Matrix multiplication for lists."""

        def is_matrix(x):
            return isinstance(x, list) and x and isinstance(x[0], list)

        if is_matrix(a) and is_matrix(b):
            rows, shared, cols = len(a), len(a[0]), len(b[0])
            if len(b) != shared:
                raise ValueError("matmul dimension mismatch")
            return [
                [sum(a[i][k] * b[k][j] for k in range(shared)) for j in range(cols)]
                for i in range(rows)
            ]
        if is_matrix(a) and not is_matrix(b):
            if len(a[0]) != len(b):
                raise ValueError("matmul dimension mismatch")
            return [sum(a[i][k] * b[k] for k in range(len(b))) for i in range(len(a))]
        if not is_matrix(a) and is_matrix(b):
            if len(a) != len(b):
                raise ValueError("matmul dimension mismatch")
            return [
                sum(a[k] * b[k][j] for k in range(len(a))) for j in range(len(b[0]))
            ]
        if not is_matrix(a) and not is_matrix(b):
            if len(a) != len(b):
                raise ValueError("matmul dimension mismatch")
            return sum(a[i] * b[i] for i in range(len(a)))
        raise NotImplementedError("Unsupported types for matmul")

    # Creation ops
    def full_(self, size: Tuple[int, ...], fill_value: Any, dtype: Any, device: Any):
        if not size:
            return fill_value
        return [self.full_(size[1:], fill_value, dtype, device) for _ in range(size[0])]

    def zeros_(self, size: Tuple[int, ...], dtype: Any, device: Any):
        return self.full_(size, 0, dtype, device)

    def clone_(self) -> Any:
        tensor = self.data
        return self._clone_recursive(tensor)

    def _clone_recursive(self, tensor: Any) -> Any:
        if isinstance(tensor, list):
            return [self._clone_recursive(item) for item in tensor]
        return tensor

    def to_device_(self, device: Any) -> Any:
        return self.data

    def stack_(self, tensors: List[Any], dim: int = 0) -> Any:
        if not tensors:
            return []
        tensors = [self._AbstractTensor__unwrap(t) for t in tensors]
        if dim == 0:
            return [self._clone_recursive(t) for t in tensors]
        ref_shape = _get_shape(tensors[0])
        for t in tensors:
            if _get_shape(t) != ref_shape:
                raise ValueError("All tensors must have the same shape")
        return [
            self.stack_([t[i] for t in tensors], dim=dim - 1)
            for i in range(len(tensors[0]))
        ]

    def get_device_(self, tensor: Any) -> Any:
        return "cpu_pure_python"

    def get_dtype_(self, tensor: Any) -> Any:
        if isinstance(tensor, list):
            if not tensor:
                return None
            return self.get_dtype_(tensor[0])
        return type(tensor)

    def item_(self, tensor: Any) -> Any:
        if isinstance(tensor, list) and len(tensor) == 1:
            return tensor[0]
        return tensor

    def max_(self, tensor: Any) -> Any:
        flat = _flatten(tensor)
        return max(flat) if flat else None

    def long_cast_(self, tensor: Any) -> Any:
        if isinstance(tensor, list):
            return [self.long_cast_(item) for item in tensor]
        return int(tensor)

    def float_(self, tensor: Any) -> Any:
        return self.to_dtype_(tensor, "float")

    def double_(self, tensor: Any) -> Any:
        return self.to_dtype_(tensor, "double")

    def int_(self, tensor: Any) -> Any:
        return self.to_dtype_(tensor, "int")

    def long_(self, tensor: Any) -> Any:
        return self.to_dtype_(tensor, "long")

    def bool_(self, tensor: Any) -> Any:
        return self.to_dtype_(tensor, "bool")

    def not_equal_(self, value: Any) -> Any:
        value = value.data if isinstance(value, AbstractTensor) else value

        def rec(a, b):
            if isinstance(a, list) and isinstance(b, list):
                return [rec(x, y) for x, y in zip(a, b)]
            if isinstance(a, list):
                return [rec(x, b) for x in a]
            if isinstance(b, list):
                return [rec(a, y) for y in b]
            return a != b

        return rec(self.data, value)

    def arange_(self, start: int, end: int, step: int = 1, *, dtype: Any = None, device: Any = None) -> Any:
        return list(range(start, end, step))

    def select_by_indices_(self, tensor: Any, indices_dim0: Any, indices_dim1: Any) -> Any:
        if not isinstance(tensor, list) or not isinstance(tensor[0], list):
            raise NotImplementedError("select_by_indices only supports 2D lists for now")
        i0 = self._AbstractTensor__unwrap(indices_dim0)
        i1 = self._AbstractTensor__unwrap(indices_dim1)
        selected_rows = [tensor[i] for i in i0]
        if isinstance(i1, list):
            if len(i0) != len(i1):
                raise ValueError("Index lists must have same length for element-wise selection")
            return [selected_rows[i][i1[i]] for i in range(len(selected_rows))]
        elif isinstance(i1, slice):
            return [row[i1] for row in selected_rows]
        else:
            return [row[i1] for row in selected_rows]

    def log_softmax_tensor_(self, tensor: Any, dim: int) -> Any:
        if dim != -1 and dim != len(_get_shape(tensor)) - 1:
            raise NotImplementedError("log_softmax only implemented for last dimension")
        if not isinstance(tensor, list):
            return math.log(1.0)
        if not isinstance(tensor[0], list):
            max_val = max(tensor)
            exp_tensor = [math.exp(x - max_val) for x in tensor]
            sum_exp = sum(exp_tensor)
            return [math.log(x / sum_exp) for x in exp_tensor]
        return [self.log_softmax_tensor_(sublist, dim=-1) for sublist in tensor]

    def topk_(self, tensor: Any, k: int, dim: int) -> Tuple[Any, Any]:
        shape = _get_shape(tensor)
        if not shape:
            return [tensor], [0]
        if dim < 0:
            dim += len(shape)
        if dim < 0 or dim >= len(shape):
            raise ValueError("dim out of range")
        if len(shape) == 1:
            indexed = sorted(((v, i) for i, v in enumerate(tensor)), reverse=True)[:k]
            values = [v for v, _ in indexed]
            indices = [i for _, i in indexed]
            return values, indices
        if dim == 0:
            transposed = list(zip(*tensor))
            col_vals: List[List[Any]] = []
            col_idxs: List[List[Any]] = []
            for col in transposed:
                vals, idxs = self.topk_(list(col), k, dim=0)
                col_vals.append(vals)
                col_idxs.append(idxs)
            values = [list(v) for v in zip(*col_vals)] if col_vals else []
            indices = [list(i) for i in zip(*col_idxs)] if col_idxs else []
            return values, indices
        values = []
        indices = []
        for sub in tensor:
            vals, idxs = self.topk_(sub, k, dim - 1)
            values.append(vals)
            indices.append(idxs)
        return values, indices

    def transpose_(self, dim0: int, dim1: int):
        data = self.data
        if dim0 == 0 and dim1 == 1:
            if not all(isinstance(row, list) for row in data):
                raise NotImplementedError("transpose_ expects a 2D list")
            return [list(row) for row in zip(*data)]
        raise NotImplementedError("transpose_ only supports dim0=0 and dim1=1")

    def reshape_(self, shape):
        flat = _flatten(self.data)

        total = len(flat)
        shape = list(shape)
        if -1 in shape:
            minus_one_count = shape.count(-1)
            if minus_one_count > 1:
                raise ValueError("Only one dimension can be -1")
            known = 1
            for s in shape:
                if s != -1:
                    known *= s
            if total % known != 0:
                raise ValueError("Cannot reshape array with incompatible size")
            shape[shape.index(-1)] = total // known

        it = iter(flat)

        def build(s):
            if not s:
                return next(it)
            n = s[0]
            return [build(s[1:]) for _ in range(n)]

        return build(shape)

    def squeeze_(self, dim: int | None = None):
        data = self.data

        def squeeze_axis(lst, axis):
            if axis == 0:
                if isinstance(lst, list) and len(lst) == 1:
                    return lst[0]
                return lst
            if not isinstance(lst, list):
                return lst
            return [squeeze_axis(x, axis - 1) for x in lst]

        if dim is not None:
            shape = _get_shape(data)
            if dim < 0:
                dim += len(shape)
            return squeeze_axis(data, dim)

        shape = _get_shape(data)
        for axis in reversed([i for i, s in enumerate(shape) if s == 1]):
            data = squeeze_axis(data, axis)
        return data

    def pad_(self, tensor: Any, pad: Tuple[int, ...], value: float = 0) -> Any:
        if len(pad) != 4:
            raise NotImplementedError("pad only implemented for 2D tensors")
        pad_left, pad_right, pad_top, pad_bottom = pad
        if not isinstance(tensor, list) or not isinstance(tensor[0], list):
            raise ValueError("pad expects a 2D list")
        rows = len(tensor)
        cols = len(tensor[0]) if rows > 0 else 0
        padded_rows = []
        for _ in range(pad_top):
            padded_rows.append([value] * (cols + pad_left + pad_right))
        for row in tensor:
            padded_rows.append([value] * pad_left + row + [value] * pad_right)
        for _ in range(pad_bottom):
            padded_rows.append([value] * (cols + pad_left + pad_right))
        return padded_rows

    def pad2d_(self, pad, value=0.0):
        pad_left, pad_right, pad_top, pad_bottom = pad
        result = []
        for n in self.data:
            n_out = []
            for c in n:
                rows = len(c)
                cols = len(c[0]) if rows > 0 else 0
                width = cols + pad_left + pad_right
                padded = []
                for _ in range(pad_top):
                    padded.append([value] * width)
                for row in c:
                    padded.append([value] * pad_left + row + [value] * pad_right)
                for _ in range(pad_bottom):
                    padded.append([value] * width)
                n_out.append(padded)
            result.append(n_out)
        return result

    def unfold2d_(self, kernel_size, stride=1, padding=0, dilation=1):
        kH, kW = _to_tuple2(kernel_size)
        sH, sW = _to_tuple2(stride)
        pH, pW = _to_tuple2(padding)
        dH, dW = _to_tuple2(dilation)
        x = self.pad2d_((pW, pW, pH, pH), 0.0)
        N = len(x)
        C = len(x[0]) if N > 0 else 0
        H = len(x[0][0]) if C > 0 else 0
        W = len(x[0][0][0]) if H > 0 else 0
        eKH, eKW = (kH - 1) * dH + 1, (kW - 1) * dW + 1
        Hout = (H - eKH) // sH + 1
        Wout = (W - eKW) // sW + 1
        patches = []
        for n in range(N):
            cols = []
            for h in range(Hout):
                for w in range(Wout):
                    patch = []
                    for c in range(C):
                        for i in range(kH):
                            for j in range(kW):
                                hi = h * sH + i * dH
                                wj = w * sW + j * dW
                                patch.append(x[n][c][hi][wj])
                    cols.append(patch)
            # transpose to (C*kH*kW, L)
            cols = [list(col) for col in zip(*cols)]
            patches.append(cols)
        return patches

    def fold2d_(
        self,
        output_size,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
    ):
        N, C, H, W = output_size
        kH, kW = _to_tuple2(kernel_size)
        sH, sW = _to_tuple2(stride)
        pH, pW = _to_tuple2(padding)
        dH, dW = _to_tuple2(dilation)
        Hpad, Wpad = H + 2 * pH, W + 2 * pW
        eKH, eKW = (kH - 1) * dH + 1, (kW - 1) * dW + 1
        Hout = (Hpad - eKH) // sH + 1
        Wout = (Wpad - eKW) // sW + 1
        ypad = [[[ [0.0 for _ in range(Wpad)] for _ in range(Hpad)] for _ in range(C)] for _ in range(N)]
        data = self.data
        for n in range(N):
            for h in range(Hout):
                for w in range(Wout):
                    l = h * Wout + w
                    for c in range(C):
                        for i in range(kH):
                            for j in range(kW):
                                idx = ((c * kH + i) * kW) + j
                                hi = h * sH + i * dH
                                wj = w * sW + j * dW
                                ypad[n][c][hi][wj] += data[n][idx][l]
        out = []
        for n in range(N):
            chans = []
            for c in range(C):
                rows = []
                for h in range(pH, Hpad - pH):
                    rows.append(ypad[n][c][h][pW:Wpad - pW])
                chans.append(rows)
            out.append(chans)
        return out

    def cat_(self, tensors: List[Any], dim: int = 0) -> Any:
        if not tensors:
            return []
        tensors = [self._AbstractTensor__unwrap(t) for t in tensors]
        if dim == 0:
            result = []
            for t in tensors:
                result.extend(t)
            return result
        if dim == 1:
            if not all(len(t) == len(tensors[0]) for t in tensors):
                raise ValueError("Tensors must have same number of rows for dim 1 concatenation")
            result = []
            for i in range(len(tensors[0])):
                combined = []
                for t in tensors:
                    combined.extend(t[i])
                result.append(combined)
            return result
        raise NotImplementedError("cat only implemented for dim 0 and 1")

    def expand_(self, shape):
        out = self.data
        n, d = shape
        if isinstance(out[0], list):
            if len(out) == 1:
                return [out[0][:] for _ in range(n)]
        raise NotImplementedError("expand_ not implemented for this shape")

    def repeat_interleave_(self, repeats: int = 1, dim: Optional[int] = None) -> Any:
        if dim is None or dim == 0:
            tensor = self.data
            if not isinstance(tensor, list):
                return [tensor] * repeats
            result = []
            for item in tensor:
                result.extend([item] * repeats)
            return result
        raise NotImplementedError("repeat_interleave only implemented for dim 0 or None")

    def cumsum_(self, dim: int = 0) -> Any:
        try:
            import numpy as np  # type: ignore
            return np.cumsum(self.data, axis=dim).tolist()
        except Exception:
            if dim != 0:
                raise NotImplementedError("pure backend cumsum_ only supports dim=0 without numpy")
            out = []
            total = 0.0
            for v in self.data:
                total += v
                out.append(total)
            return out

    def repeat_(self, repeats: Any = None, dim: int = 0) -> Any:
        """Repeat tensor along ``dim`` ``repeats`` times (stub)."""
        raise NotImplementedError("repeat not implemented for PurePython backend")

    def view_flat_(self) -> Any:
        return _flatten(self.data)

    def assign_at_indices_(self, indices_dim0: Any, indices_dim1: Any, values_to_assign: Any):
        tensor_to_modify = self.data
        values_to_assign = self._AbstractTensor__unwrap(values_to_assign)
        i0 = self._AbstractTensor__unwrap(indices_dim0)
        i1 = self._AbstractTensor__unwrap(indices_dim1)
        if not isinstance(tensor_to_modify, list) or not isinstance(tensor_to_modify[0], list):
            raise NotImplementedError("assign_at_indices only supports 2D lists for now")
        if not isinstance(i0, list) or not isinstance(i1, list):
            raise ValueError("indices_dim0 and indices_dim1 must be lists")
        if len(i0) != len(i1) or len(i0) != len(values_to_assign):
            raise ValueError("Index lists and values list must have same length")
        for i in range(len(i0)):
            row_idx = i0[i]
            col_idx = i1[i]
            value = values_to_assign[i]
            tensor_to_modify[row_idx][col_idx] = value
        return tensor_to_modify

    def increment_at_indices_(self, mask: Any):
        tensor_to_modify = self.data
        mask = self._AbstractTensor__unwrap(mask)
        if (
            not isinstance(tensor_to_modify, list)
            or not isinstance(mask, list)
            or len(tensor_to_modify) != len(mask)
        ):
            raise NotImplementedError("increment_at_indices only supports flat lists with boolean mask")
        for i in range(len(tensor_to_modify)):
            if mask[i]:
                tensor_to_modify[i] += 1
        return tensor_to_modify

    def clamp_(self, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Any:
        def _clamp(val):
            if isinstance(val, list):
                return [_clamp(v) for v in val]
            if min_val is not None:
                val = max(val, min_val)
            if max_val is not None:
                val = min(val, max_val)
            return val
        return _clamp(self.data)

    def clamp_min_(self, min_val: float) -> Any:
        def _clamp_min(val):
            if isinstance(val, list):
                return [_clamp_min(v) for v in val]
            return val if val >= min_val else min_val
        return _clamp_min(self.data)

    def clamp_max_(self, max_val: float) -> Any:
        def _clamp_max(val):
            if isinstance(val, list):
                return [_clamp_max(v) for v in val]
            return val if val <= max_val else max_val
        return _clamp_max(self.data)

    def shape_(self) -> Tuple[int, ...]:
        return _get_shape(self.data)

    def numel_(self) -> int:
        return len(_flatten(self.data))

    def mean_(self, dim: Optional[int] = None, keepdim: bool = False) -> Any:
        data = self.data
        def _mean(lst):
            flat = _flatten(lst)
            return sum(flat) / len(flat) if flat else 0.0
        def reduce_dim(lst, d):
            if d == 0:
                # mean over axis 0
                if not isinstance(lst[0], list):
                    m = sum(lst) / len(lst) if lst else 0.0
                    return [m] if keepdim else m
                cols = len(lst[0])
                result = []
                for i in range(cols):
                    col = [row[i] for row in lst]
                    m = sum(col) / len(col) if col else 0.0
                    result.append(m)
                if keepdim:
                    return [result]
                return result
            else:
                return [reduce_dim(row, d-1) for row in lst]
        if dim is None:
            m = _mean(data)
            if keepdim:
                shape = _get_shape(data)
                for _ in range(len(shape)):
                    m = [m]
            return m
        return reduce_dim(data, dim)

    def sum_(self, dim: Optional[int] = None, keepdim: bool = False) -> Any:
        data = self.data
        def _sum(lst):
            flat = _flatten(lst)
            return sum(flat)
        def reduce_dim(lst, d):
            if d == 0:
                if not isinstance(lst[0], list):
                    s = sum(lst)
                    return [s] if keepdim else s
                cols = len(lst[0])
                result = []
                for i in range(cols):
                    col = [row[i] for row in lst]
                    s = sum(col)
                    result.append(s)
                if keepdim:
                    return [result]
                return result
            else:
                return [reduce_dim(row, d-1) for row in lst]
        if dim is None:
            s = _sum(data)
            if keepdim:
                shape = _get_shape(data)
                for _ in range(len(shape)):
                    s = [s]
            return s
        return reduce_dim(data, dim)
    def min_(self, dim: Optional[int] = None, keepdim: bool = False) -> Any:
        data = self.data
        def _min(lst):
            flat = _flatten(lst)
            return min(flat) if flat else 0.0
        def reduce_dim(lst, d):
            if d == 0:
                if not isinstance(lst[0], list):
                    m = min(lst)
                    return [m] if keepdim else m
                cols = len(lst[0])
                result = []
                for i in range(cols):
                    col = [row[i] for row in lst]
                    m = min(col)
                    result.append(m)
                if keepdim:
                    return [result]
                return result
            else:
                return [reduce_dim(row, d-1) for row in lst]
        if dim is None:
            m = _min(data)
            if keepdim:
                shape = _get_shape(data)
                for _ in range(len(shape)):
                    m = [m]
            return m
        return reduce_dim(data, dim)

    def pow_(self, tensor: Any, exponent: float) -> Any:
        tensor = self._AbstractTensor__unwrap(tensor)
        if isinstance(tensor, list):
            return [self.pow_(item, exponent) for item in tensor]
        return tensor**exponent

    def sqrt_(self, tensor: Any) -> Any:
        tensor = self._AbstractTensor__unwrap(tensor)
        if isinstance(tensor, list):
            return [self.sqrt_(item) for item in tensor]
        return math.sqrt(tensor)

    def tensor_from_list_(self, data: list, dtype: Any, device: Any) -> Any:
        if not isinstance(data, (list, tuple)):
            try:
                data = data.tolist()
                auto_converted = True
            except Exception:
                auto_converted = False
        else:
            auto_converted = False
        if auto_converted:
            print("[TensorBackend:pure] Auto-converted input to list for tensor_from_list_()")
        return data

    def boolean_mask_select_(self, tensor: Any, mask: Any) -> Any:
        tensor = self._AbstractTensor__unwrap(tensor)
        mask = self._AbstractTensor__unwrap(mask)
        if (
            not isinstance(tensor, list)
            or not isinstance(mask, list)
            or len(tensor) != len(mask)
        ):
            raise NotImplementedError("boolean_mask_select only supports flat lists with boolean mask")
        return [tensor[i] for i in range(len(tensor)) if mask[i]]

    def tolist_(self) -> list:
        return self.clone_()

    def less_(self, value: Any) -> Any:
        value = value.data if isinstance(value, AbstractTensor) else value

        def rec(a, b):
            if isinstance(a, list) and isinstance(b, list):
                return [rec(x, y) for x, y in zip(a, b)]
            if isinstance(a, list):
                return [rec(x, b) for x in a]
            if isinstance(b, list):
                return [rec(a, y) for y in b]
            return a < b

        return rec(self.data, value)

    def index_select_(self, tensor: Any, dim: int, indices: Any) -> Any:
        tensor = self._AbstractTensor__unwrap(tensor)
        idx = self._AbstractTensor__unwrap(indices)
        if dim == 0:
            return [tensor[i] for i in idx]
        if dim == 1:
            return [[row[i] for i in idx] for row in tensor]
        raise NotImplementedError("index_select only implemented for dim 0 or 1")

    def argmin_(self, tensor: Any, dim: Optional[int] = None, keepdim: bool = False) -> Any:
        data = self._AbstractTensor__unwrap(tensor)
        def _argmin(lst):
            flat = _flatten(lst)
            return flat.index(min(flat)) if flat else 0
        def reduce_dim(lst, d):
            if d == 0:
                if not isinstance(lst[0], list):
                    idx = lst.index(min(lst))
                    return [idx] if keepdim else idx
                cols = len(lst[0])
                result = []
                for i in range(cols):
                    col = [row[i] for row in lst]
                    idx = col.index(min(col))
                    result.append(idx)
                if keepdim:
                    return [result]
                return result
            else:
                return [reduce_dim(row, d-1) for row in lst]
        if dim is None:
            idx = _argmin(data)
            if keepdim:
                shape = _get_shape(data)
                for _ in range(len(shape)):
                    idx = [idx]
            return idx
        return reduce_dim(data, dim)

    def diag_(self, offset: int = 0):
        data = self.data
        # If input is 2D, extract diagonal with offset
        if data and isinstance(data[0], list):
            rows = len(data)
            cols = len(data[0])
            diag = []
            for i in range(rows):
                j = i + offset
                if 0 <= j < cols:
                    diag.append(data[i][j])
            return diag
        # Otherwise assume 1D vector and construct diagonal matrix
        n = len(data)
        size = n + abs(offset)
        mat = [[0 for _ in range(size)] for _ in range(size)]
        for idx, val in enumerate(data):
            if offset >= 0:
                i, j = idx, idx + offset
            else:
                i, j = idx - offset, idx
            mat[i][j] = val
        return mat

    def interpolate_(self, tensor: Any, size: Tuple[int, ...]) -> Any:
        tensor = self._AbstractTensor__unwrap(tensor)
        shape = _get_shape(tensor)
        if len(shape) != len(size):
            raise ValueError("size must match tensor dimensions")
        def blend(a, b, frac):
            if not isinstance(a, list):
                return a * (1 - frac) + b * frac
            return [blend(ai, bi, frac) for ai, bi in zip(a, b)]
        def interp_along_dim(data, dim, new_len):
            if dim == 0:
                if not isinstance(data, list):
                    return data
                old_len = len(data)
                if old_len == 1:
                    return [self.clone_(data[0]) for _ in range(new_len)]
                result = []
                for i in range(new_len):
                    pos = (i * (old_len - 1)) / (new_len - 1) if new_len > 1 else 0
                    left = int(math.floor(pos))
                    right = min(left + 1, old_len - 1)
                    frac = pos - left
                    if frac == 0:
                        result.append(self.clone_(data[left]))
                    else:
                        result.append(blend(data[left], data[right], frac))
                return result
            return [interp_along_dim(sub, dim - 1, new_len) for sub in data]
        result = tensor
        for dim_idx in reversed(range(len(size))):
            result = interp_along_dim(result, dim_idx, size[dim_idx])
        return result

    def save_(self, tensor: Any, filepath: str) -> None:
        tensor = self._AbstractTensor__unwrap(tensor)
        with open(filepath, "w") as f:
            json.dump(tensor, f)

    def load_(self, filepath: str, dtype: Any, device: Any) -> Any:
        with open(filepath, "r") as f:
            return json.load(f)

    @property
    def long_dtype_(self) -> Any:
        return int

    @property
    def bool_dtype_(self) -> Any:
        return bool

    @property
    def float_dtype_(self) -> Any:
        return float

    tensor_type_ = list

    @staticmethod
    def test() -> None:
        """Quick smoke test for basic operations."""
        ops = PurePythonTensorOperations()
        stacked = ops.stack([[1, 2], [3, 4]], dim=0)
        assert stacked == [[[1, 2], [3, 4]], [[1, 2], [3, 4]]]
        values, idxs = ops.topk([1, 3, 2, 4], k=2, dim=-1)
        assert values == [4, 3] and idxs == [3, 1]

    @staticmethod
    def from_numpy(source_ops, tensor, target_ops):
        # If already pure python, just return the data
        if isinstance(source_ops, PurePythonTensorOperations):
            data = source_ops.data
        else:
            arr = tensor.data if hasattr(tensor, 'data') else tensor
            data = arr.tolist() if hasattr(arr, 'tolist') else list(arr)
        result = type(target_ops)(track_time=target_ops.track_time)
        result.data = data
        return result

    @staticmethod
    def from_torch(source_ops, tensor, target_ops):
        if isinstance(source_ops, PurePythonTensorOperations):
            data = source_ops.data
        else:
            t = tensor.data if hasattr(tensor, 'data') else tensor
            data = t.detach().cpu().tolist()
        result = type(target_ops)(track_time=target_ops.track_time)
        result.data = data
        return result

    @staticmethod
    def from_pure(source_ops, tensor, target_ops):
        data = tensor.data if hasattr(tensor, 'data') else tensor
        result = type(target_ops)(track_time=target_ops.track_time)
        result.data = data
        return result

    @staticmethod
    def from_jax(source_ops, tensor, target_ops):
        if isinstance(source_ops, PurePythonTensorOperations):
            data = source_ops.data
        else:
            d = tensor.data if hasattr(tensor, 'data') else tensor
            data = d.tolist() if hasattr(d, 'tolist') else list(d)
        result = type(target_ops)(track_time=target_ops.track_time)
        result.data = data
        return result

    def get_shape(self) -> tuple[int, ...]:
        return _get_shape(self.data)

    def get_ndims(self) -> int:
        return len(_get_shape(self.data))

    def _map_unary(self, func, data):
        if isinstance(data, list):
            return [self._map_unary(func, x) for x in data]
        return func(data)

    def neg_(self):
        return self._map_unary(lambda x: -x, self.data)

    def abs_(self):
        return self._map_unary(abs, self.data)

    def invert_(self):
        return self._map_unary(lambda x: ~x, self.data)

    def round_(self, n=None):
        return self._map_unary(lambda x: round(x, n) if n is not None else round(x), self.data)

    def trunc_(self):
        import math
        return self._map_unary(math.trunc, self.data)

    def floor_(self):
        import math
        return self._map_unary(math.floor, self.data)

    def ceil_(self):
        import math
        return self._map_unary(math.ceil, self.data)

    def to_dtype_(self, tensor, dtype: str = "float"):
        # For pure Python, just convert all elements recursively
        def convert(val):
            if dtype in ("float", "float32", "f32"):
                return float(val)
            elif dtype in ("int", "int32", "i32"):
                return int(val)
            elif dtype in ("bool",):
                return bool(val)
            else:
                return float(val)
        if isinstance(tensor, list):
            return [self.to_dtype_(t, dtype) for t in tensor]
        return convert(tensor)

    @classmethod
    def tensor_from_list(cls, data, dtype=None, device=None):
        inst = cls(track_time=False)
        inst.data = inst.tensor_from_list_(data, dtype, device)
        return inst

register_backend("pure_python", PurePythonTensorOperations)
