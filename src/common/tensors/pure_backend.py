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

class PurePythonTensorOperations(AbstractTensor):
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
            return x - y if op != "rsub" else y - x
        if op in ("mul", "rmul", "imul"):
            return x * y
        if op in ("truediv", "rtruediv", "itruediv"):
            return x / y if op != "rtruediv" else y / x
        if op in ("floordiv", "rfloordiv", "ifloordiv"):
            return x // y if op != "rfloordiv" else y // x
        if op in ("mod", "rmod", "imod"):
            return x % y if op != "rmod" else y % x
        if op in ("pow", "rpow", "ipow"):
            return x**y if op != "rpow" else y**x
        if op in ("matmul", "rmatmul", "imatmul"):
            return self._matmul(x, y) if op != "rmatmul" else self._matmul(y, x)
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

    def clone_(self, tensor: Any) -> Any:
        tensor = self._AbstractTensor__unwrap(tensor)
        if not isinstance(tensor, list):
            return tensor
        return [self.clone_(item) for item in tensor]

    def to_device_(self, tensor: Any, device: Any) -> Any:
        return self._AbstractTensor__unwrap(tensor)

    def stack_(self, tensors: List[Any], dim: int = 0) -> Any:
        if not tensors:
            return []
        tensors = [self._AbstractTensor__unwrap(t) for t in tensors]
        if dim == 0:
            return [self.clone_(t) for t in tensors]
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

    def not_equal_(self, tensor1: Any, tensor2: Any) -> Any:
        if isinstance(tensor1, list) and isinstance(tensor2, list):
            return [self.not_equal_(t1, t2) for t1, t2 in zip(tensor1, tensor2)]
        return tensor1 != tensor2

    def arange_(self, start: int, end: Optional[int] = None, step: int = 1, device: Any = None, dtype: Any = None) -> Any:
        if end is None:
            return list(range(start))
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

    def log_softmax_(self, tensor: Any, dim: int) -> Any:
        if dim != -1 and dim != len(_get_shape(tensor)) - 1:
            raise NotImplementedError("log_softmax only implemented for last dimension")
        if not isinstance(tensor, list):
            return math.log(1.0)
        if not isinstance(tensor[0], list):
            max_val = max(tensor)
            exp_tensor = [math.exp(x - max_val) for x in tensor]
            sum_exp = sum(exp_tensor)
            return [math.log(x / sum_exp) for x in exp_tensor]
        return [self.log_softmax_(sublist, dim=-1) for sublist in tensor]

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
        return self.clone_(self.data)

    def less_(self, tensor: Any, value: Any) -> Any:
        tensor = self._AbstractTensor__unwrap(tensor)
        if isinstance(tensor, list):
            return [self.less_(item, value) for item in tensor]
        return tensor < value

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

    @property
    def tensor_type_(self) -> type:
        return list

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
