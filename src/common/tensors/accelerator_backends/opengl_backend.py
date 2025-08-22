"""OpenGL backend skeleton using buffer objects and compute shaders."""
from __future__ import annotations

try:
    from typing import Any, Tuple
    from ..abstraction import AbstractTensor
    import numpy as np
    # PyOpenGL imports are optional at this stage
    from OpenGL import GL  # type: ignore
except ModuleNotFoundError:
    np = None  # type: ignore
    GL = None  # type: ignore
except Exception:
    import sys
    print("OpenGL backend failed to import")
    sys.exit(1)


class GLBuffer:
    """Wrapper for an OpenGL buffer object."""

    def __init__(self, shape: Tuple[int, ...], buffer_id: int | None = None) -> None:
        self.shape = shape
        self.size = int(np.prod(shape)) if np is not None else 0
        self.buffer_id = buffer_id if buffer_id is not None else 0


# ########## STUB: OpenGL compute shader pipeline ##########
# PURPOSE: Provide tensor operations using OpenGL compute shaders.
# EXPECTED BEHAVIOR: This class will allocate GPU buffers, compile
#          compute shaders, dispatch workloads, and read back results
#          as needed. Tensor methods will map to shader invocations
#          or buffer manipulations.
# INPUTS: Shader source strings or file paths, numpy arrays as staging
#         buffers, and configuration parameters such as work group sizes.
# OUTPUTS: Updated ``GLBuffer`` objects representing tensor data.
# KEY ASSUMPTIONS/DEPENDENCIES: Requires a valid OpenGL context and the
#         ``PyOpenGL`` and ``glfw`` packages. Assumes numpy is available
#         for staging data on the CPU.
# TODO:
#   - Implement buffer creation and deletion logic.
#   - Compile and link compute shaders from source.
#   - Dispatch compute workloads for tensor operations.
#   - Support conversion to and from other backends via numpy arrays.
# NOTES: This initial skeleton establishes the API but does not perform
#         real GPU work. Methods raise ``NotImplementedError`` for now.
# ###########################################################################
class OpenGLTensorOperations(AbstractTensor):
    """Stub OpenGL backend using buffers and compute shaders."""

    def __init__(self, track_time: bool = False) -> None:
        super().__init__(track_time=track_time)
        if GL is None:
            raise RuntimeError("PyOpenGL is required for the OpenGL backend")

    def _apply_operator__(self, op: str, left: Any, right: Any):
        raise NotImplementedError("OpenGL operator dispatch not implemented")

    def full_(self, size: Tuple[int, ...], fill_value: Any, dtype: Any, device: Any):
        raise NotImplementedError

    def zeros_(self, size: Tuple[int, ...], dtype: Any, device: Any):
        raise NotImplementedError

    def clone_(self):
        raise NotImplementedError

    def to_device_(self, device: Any):
        raise NotImplementedError

    def get_device_(self):
        raise NotImplementedError

    def get_dtype_(self):
        raise NotImplementedError

    def item_(self):
        raise NotImplementedError

    def max_(self):
        raise NotImplementedError

    def long_cast_(self):
        raise NotImplementedError

    def float_(self):
        raise NotImplementedError

    def double_(self):
        raise NotImplementedError

    def int_(self):
        raise NotImplementedError

    def long_(self):
        raise NotImplementedError

    def bool_(self):
        raise NotImplementedError

    def not_equal_(self, other: Any):
        raise NotImplementedError

    def arange_(self, start: int, end: int | None = None, step: int = 1, device: Any = None, dtype: Any = None):
        raise NotImplementedError

    def select_by_indices_(self, indices_dim0: Any, indices_dim1: Any):
        raise NotImplementedError

    def log_softmax_(self, dim: int):
        raise NotImplementedError

    def pad_(self, pad: Tuple[int, ...], value: float = 0.0):
        raise NotImplementedError

    def cat_(self, tensors: list[Any], dim: int = 0):
        raise NotImplementedError

    def topk_(self, k: int, dim: int):
        raise NotImplementedError

    def stack_(self, tensors: list[Any], dim: int = 0):
        raise NotImplementedError

    def unravel_index_(self, shape):
        raise NotImplementedError(
            "unravel_index not implemented for OpenGL backend"
        )

    def repeat_interleave_(self, repeats: int = 1, dim: int | None = None):
        raise NotImplementedError

    def repeat_(self, repeats: Any = None, dim: int = 0):
        """Repeat tensor along ``dim`` ``repeats`` times (stub)."""
        raise NotImplementedError("repeat not implemented for OpenGL backend")

    def view_flat_(self):
        raise NotImplementedError

    def assign_at_indices_(self, indices_dim0: Any, indices_dim1: Any, values_to_assign: Any):
        raise NotImplementedError

    def increment_at_indices_(self, mask: Any):
        raise NotImplementedError

    def clamp_(self, min_val: float | None = None, max_val: float | None = None):
        raise NotImplementedError

    def numel_(self):
        raise NotImplementedError

    def mean_(self, dim: int | None = None):
        raise NotImplementedError

    def pow_(self, exponent: float):
        raise NotImplementedError

    def sqrt_(self):
        raise NotImplementedError

    def tensor_from_list_(self, data: list[Any], dtype: Any, device: Any):
        raise NotImplementedError

    def boolean_mask_select_(self, mask: Any):
        raise NotImplementedError

    def tolist_(self):
        raise NotImplementedError

    def less_(self, value: Any):
        raise NotImplementedError

    def index_select_(self, dim: int, indices: Any):
        raise NotImplementedError

    def argmin_(self, dim: int | None = None):
        raise NotImplementedError

    def save_(self, filepath: str) -> None:
        """Persist tensor to disk (stub)."""
        raise NotImplementedError("save not implemented for OpenGL backend")

    def load_(self, filepath: str, dtype: Any, device: Any):
        """Load tensor from disk (stub)."""
        raise NotImplementedError("load not implemented for OpenGL backend")

    def to_dtype_(self, dtype: str = "float"):
        """Convert tensor dtype (stub)."""
        raise NotImplementedError("to_dtype not implemented for OpenGL backend")

    def get_shape(self):
        raise NotImplementedError

    def get_ndims(self):
        raise NotImplementedError

    def interpolate_(self, size: Tuple[int, ...]):
        raise NotImplementedError
