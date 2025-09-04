
from __future__ import annotations
from enum import auto
from typing import Any, Tuple, Optional, List, Union, Callable, Dict

from .random import random_generator, RANDOM_KIND, PRNG_ALGO, CSPRNG_ALGO

# --- Unified creation helpers (tape + requires_grad) -----------------------
class _CreationTapeCtx:
    def __init__(self, *, requires_grad: bool = False, tape=None):
        # Lazy import to avoid module cycles
        from .. import autograd as _autograd
        self._autograd = _autograd
        self._prev = None
        self._desired = _autograd.autograd.tape if requires_grad else tape
    def __enter__(self):
        if self._desired is None:
            return None
        self._prev = self._autograd.autograd.tape
        self._autograd.autograd.tape = self._desired
        return self._desired
    def __exit__(self, exc_type, exc, tb):
        if self._desired is not None:
            self._autograd.autograd.tape = self._prev


def _finalize_requires(tensor, requires_grad: bool):
    if requires_grad:
        try:
            tensor.requires_grad_(True)
        except Exception:
            try:
                setattr(tensor, "_requires_grad", True)
            except Exception:
                pass
    return tensor
# Helper to take n items from an iterable and return as a list/array
def take_n(iterable, n):
    return [next(iterable) for _ in range(n)]
def empty(
    size: Tuple[int, ...],
    dtype: Any = None,
    device: Any = None,
    *,
    cls=None,
    requires_grad: bool = False,
    tape=None,
):
    """Create an uninitialized tensor of the given shape using the requested backend.

    Note: requires_grad/tape are used for autograd context and not forwarded elsewhere.
    """
    cls = _resolve_cls(cls)
    with _CreationTapeCtx(requires_grad=requires_grad, tape=tape):
        inst = cls(track_time=False)
        inst.data = inst.empty_(size, dtype, device)
    return _finalize_requires(inst, requires_grad)
# Random tensor creation (fluent with other helpers)
def random_tensor(size: Tuple[int, ...], device: Any = None, scope: Tuple[Any, ...] = [0, 1], *, cls=None, **kwargs):
    """
    Create a tensor of the given shape filled with random values using the requested random generator.
    All arguments for random_generator (kind, algo, seed, dtype, distribution, batch_size, etc) should be passed as kwargs.
    Only tensor-specific arguments (size, device, cls) are explicit.
    """
    # Pull autograd controls out of RNG kwargs
    requires_grad = bool(kwargs.pop("requires_grad", False))
    tape = kwargs.pop("tape", None)
    cls = _resolve_cls(cls)
    total = 1
    for s in size:
        total *= s
    # Always set batch_size for the generator
    kwargs = dict(kwargs)  # copy to avoid mutating caller
    kwargs.setdefault('batch_size', total)
    if 'kind' not in kwargs:
        if kwargs.get('seed') is not None:
            kwargs['kind'] = RANDOM_KIND.PRNG
        else:
            kwargs['kind'] = RANDOM_KIND.CSPRNG
    rng = random_generator(**kwargs)
    
    vals = [next(rng) for _ in range(total)]
    

    with _CreationTapeCtx(requires_grad=requires_grad, tape=tape):
        inst = cls(track_time=False)
        inst.data = inst.tensor_from_list_(vals, kwargs.get('dtype'), device)
        inst = inst.reshape(*size)
    return _finalize_requires(inst, requires_grad)

def eye_like(A, n=None):
    """Create an identity matrix tensor like A, with optional size n."""
    from ..abstraction import AbstractTensor
    n = n if n is not None else A.get_shape()[-1]
    I = AbstractTensor.zeros_like(A)
    for i in range(n):
        I[..., i, i] = (I[..., i, i] * 0) + 1
    return I

def randint(size: Tuple[int, ...], low: int, high: int, device: Any = None, *, cls=None, **kwargs):
    """Create a tensor of the given shape filled with random integers in [low, high)."""
    requires_grad = bool(kwargs.pop("requires_grad", False))
    tape = kwargs.pop("tape", None)
    cls = _resolve_cls(cls)
    total = 1
    for s in size:
        total *= s
    if 'batch_size' not in kwargs:
        kwargs['batch_size'] = total
    if 'dtype' not in kwargs:
        kwargs['dtype'] = 'float'
    if 'kind' not in kwargs:
        if kwargs.get('seed') is not None:
            kwargs['kind'] = RANDOM_KIND.PRNG
        else:
            kwargs['kind'] = RANDOM_KIND.CSPRNG
    rng = random_generator(**kwargs)
    vals = next(rng)
    if not isinstance(vals, list):
        vals = [vals]
    # Map floats in [0,1) to [low, high)
    int_vals = [int(low + (high - low) * v) for v in vals]
    with _CreationTapeCtx(requires_grad=requires_grad, tape=tape):
        inst = cls(track_time=False)
        inst.data = inst.tensor_from_list_(int_vals, 'int', device)
        inst = inst.reshape(*size)
    return _finalize_requires(inst, requires_grad)


def randint_like(tensor, low: int, high: int, device: Any = None, *, cls=None, **kwargs):
    """Return a random integer tensor with the same shape as `tensor` and values in [low, high)."""
    size, _cls = likeness(tensor)
    if cls is None:
        cls = type(tensor)
    return randint(size, low, high, device, cls=cls, **kwargs)


def linspace(start, stop, steps, dtype=None, device=None, *, requires_grad: bool = False, tape=None):
    """Dispatch to backend-aware arange to produce linearly spaced values."""
    from ..abstraction import AbstractTensor  # Local import to avoid circular dependency

    if steps <= 0:
        raise ValueError("steps must be positive")
    i = AbstractTensor.arange(0, steps, 1, dtype=dtype, device=device, requires_grad=requires_grad, tape=tape)
    if steps == 1:
        return i * 0 + start  # length-1 tensor with value `start`
    step_val = (stop - start) / (steps - 1)
    return start + i * step_val


def meshgrid(*vectors, indexing: str = "ij", copy: bool = False, as_class: bool = False):
    """
    Create N-D coordinate matrices from 1-D coordinate vectors.

    Parameters
    ----------
    *vectors : AbstractTensor 1-D, or a single iterable of them
        1-D tensors defining the grid coordinates along each axis.
    indexing : {'ij','xy'}, default 'ij'
        'ij'  -> matrix indexing for any N.
        'xy'  -> Cartesian indexing for the first two axes (N>=2).
    copy : bool, default False
        If True, materialize writable copies (best-effort: uses .clone() or .copy() if available).
        If False, return broadcasted views (memory-efficient).
    as_class : bool, default False
        If True, return a MeshGrid wrapper; else return a tuple.

    Returns
    -------
    tuple[AbstractTensor, ...] or MeshGrid
    """
    from ..abstraction import AbstractTensor, MeshGrid  # Local import to avoid circular dependency

    if len(vectors) == 1 and isinstance(vectors[0], (list, tuple)):
        vectors = tuple(vectors[0])

    if len(vectors) == 0:
        raise ValueError("meshgrid requires at least one 1-D input tensor")

    vecs = []
    sizes = []
    for i, v in enumerate(vectors):
        if not isinstance(v, AbstractTensor):
            v = AbstractTensor.get_tensor(v)
        shp = getattr(v, "shape", None)
        if shp is None or len(shp) != 1:
            raise ValueError(f"meshgrid expects 1-D tensors; arg {i} has shape {shp}")
        vecs.append(v)
        sizes.append(shp[0])

    dims = len(vecs)
    if indexing not in ("ij", "xy"):
        raise ValueError("indexing must be 'ij' or 'xy'")

    base_cls = type(vecs[0])
    if any(type(v) is not base_cls for v in vecs[1:]):
        raise TypeError("meshgrid inputs must be from the same backend/class")

    want_dtype = getattr(vecs[0], "dtype", None)
    want_device = getattr(vecs[0], "device", None)
    for i, v in enumerate(vecs[1:], start=1):
        v_dtype = getattr(v, "dtype", None)
        v_device = getattr(v, "device", None)
        if (want_dtype is not None and v_dtype is not None and v_dtype != want_dtype) or \
           (want_device is not None and v_device is not None and v_device != want_device):
            raise TypeError(
                f"meshgrid inputs must share dtype/device; arg0 has (dtype={want_dtype}, device={want_device}) "
                f"but arg{i} has (dtype={v_dtype}, device={v_device})."
            )

    grids = []
    if indexing == "ij" or dims < 2:
        full_shape = tuple(sizes)
        for i, v in enumerate(vecs):
            shape = [1] * dims
            shape[i] = sizes[i]
            g = v.reshape(shape).expand(full_shape)
            if copy:
                if hasattr(g, "clone"):
                    g = g.clone()
                elif hasattr(g, "copy"):
                    g = g.copy()
            grids.append(g)
    else:
        final_shape = (sizes[1], sizes[0], *sizes[2:])
        for i, v in enumerate(vecs):
            shape = [1] * dims
            if i == 0:
                shape[1] = sizes[0]
            elif i == 1:
                shape[0] = sizes[1]
            else:
                shape[i] = sizes[i]
            g = v.reshape(shape).expand(final_shape)
            if copy:
                if hasattr(g, "clone"):
                    g = g.clone()
                elif hasattr(g, "copy"):
                    g = g.copy()
            grids.append(g)

    return MeshGrid(grids) if as_class else tuple(grids)


# ---------------------------------------------------------------------------
# Basic tensor creation helpers


def _resolve_cls(cls):
    """Best-effort backend class resolution.

    Mirrors the lazy backend selection used in :func:`AbstractTensor.arange`.
    """
    if cls is not None:
        return cls
    from ..abstraction import BACKEND_REGISTRY  # Local import to avoid circular dependency

    for backend_name in ("numpy", "torch", "pure_python"):
        backend_cls = BACKEND_REGISTRY.get(backend_name)
        if backend_cls is not None:
            return backend_cls
        
    try:
        from .. import numpy_backend
        backend_cls = BACKEND_REGISTRY.get("numpy")
        if backend_cls is not None:
            return backend_cls

    except ModuleNotFoundError:
        try:
            from .. import pure_python_backend
            backend_cls = BACKEND_REGISTRY.get("pure_python")
            if backend_cls is not None:
                return backend_cls
        except ModuleNotFoundError:
            print("[ERROR] Could not import pure Python backend. Ensure you have the necessary dependencies installed.")
            
    raise RuntimeError("No tensor backend available for tensor creation.")

def zero(cls, dtype: Any = None, device: Any = None):
    from ..abstraction import AbstractTensor  # Local import to avoid circular dependency

    cls = _resolve_cls(cls)
    return AbstractTensor.get_tensor([0.0], dtype=dtype, device=device, cls=cls)

def one(cls, dtype: Any = None, device: Any = None):
    """Create a tensor filled with ones using the requested backend."""
    from ..abstraction import AbstractTensor  # Local import to avoid circular dependency

    cls = _resolve_cls(cls)
    inst = zero(cls, dtype=dtype, device=device) + 1

    return inst

def zeros(size: Tuple[int, ...], dtype: Any = None, device: Any = None, *, cls=None, requires_grad: bool = False, tape=None):
    """Create a tensor filled with zeros using the requested backend."""
    from ..abstraction import AbstractTensor  # Local import to avoid circular dependency

    with _CreationTapeCtx(requires_grad=requires_grad, tape=tape):
        out = zero(cls, dtype=dtype, device=device).repeat(size)
    return _finalize_requires(out, requires_grad)


def randoms(size: Tuple[int, ...], device: Any = None, *, cls=None, **kwargs):
    """Alias for random_tensor for API symmetry with zeros/ones/full."""
    return random_tensor(size, device, cls=cls, **kwargs)


def ones(size: Tuple[int, ...], dtype: Any = None, device: Any = None, *, cls=None, requires_grad: bool = False, tape=None):
    """Create a tensor filled with ones using the requested backend."""
    from ..abstraction import AbstractTensor  # Local import to avoid circular dependency

    with _CreationTapeCtx(requires_grad=requires_grad, tape=tape):
        out = one(cls, dtype=dtype, device=device).repeat(size)
    return _finalize_requires(out, requires_grad)


def full(
    size: Tuple[int, ...],
    fill_value: Any,
    dtype: Any = None,
    device: Any = None,
    *,
    cls=None,
    requires_grad: bool = False,
    tape=None,
):
    """Create a tensor of ``size`` filled with ``fill_value`` using the backend."""
    from ..abstraction import AbstractTensor  # Local import to avoid circular dependency

    cls = _resolve_cls(cls)
    with _CreationTapeCtx(requires_grad=requires_grad, tape=tape):
        inst = AbstractTensor.get_tensor([fill_value], dtype=dtype, device=device, cls=cls)
        out = inst.repeat(size)
    return _finalize_requires(out, requires_grad)

def likeness(tensor):
    return tensor.shape, likeclass(tensor)

def likeclass(tensor, dtype: Any = None, device: Any = None):
    from ..abstraction import AbstractTensor  # Local import to avoid circular dependency
    """Return a tensor with the same shape and dtype as ``tensor``."""
    cls = AbstractTensor.backend_class_from_backend_data(tensor)
    if cls is None:
        cls = AbstractTensor.check_or_build_registry()
    return cls



def zeros_like(tensor, dtype: Any = None, device: Any = None):
    """Return a zeros tensor with the same shape as ``tensor``."""
    from ..abstraction import AbstractTensor  # Local import to avoid circular dependency

    size, cls = likeness(tensor)
    if dtype is None:
        dtype = tensor.get_dtype() if isinstance(tensor, AbstractTensor) else getattr(tensor, "dtype", AbstractTensor.float_dtype_)
    if device is None:
        device = tensor.get_device() if isinstance(tensor, AbstractTensor) else getattr(tensor, "device", None)
    return full(size, 0, dtype, device, cls=cls)


def ones_like(tensor, dtype: Any = None, device: Any = None):
    """Return a ones tensor with the same shape as ``tensor``."""
    from ..abstraction import AbstractTensor  # Local import to avoid circular dependency

    size, cls = likeness(tensor)
    if dtype is None:
        dtype = tensor.get_dtype() if isinstance(tensor, AbstractTensor) else getattr(tensor, "dtype", AbstractTensor.float_dtype_)
    if device is None:
        device = tensor.get_device() if isinstance(tensor, AbstractTensor) else getattr(tensor, "device", None)
    return full(size, 1, dtype, device, cls=cls)


def full_like(tensor, fill_value: Any, dtype: Any = None, device: Any = None):
    """Return a tensor filled with ``fill_value`` and the same shape as ``tensor``."""
    from ..abstraction import AbstractTensor  # Local import to avoid circular dependency

    size, cls = likeness(tensor)
    if dtype is None:
        dtype = tensor.get_dtype() if isinstance(tensor, AbstractTensor) else getattr(tensor, "dtype", AbstractTensor.float_dtype_)
    if device is None:
        device = tensor.get_device() if isinstance(tensor, AbstractTensor) else getattr(tensor, "device", None)
    return full(size, fill_value, dtype, device, cls=cls)

def rand_like(tensor, device: Any = None, *, cls=None, **kwargs):
    """Return a random tensor with the same shape as `tensor`."""
    size, _cls = likeness(tensor)
    if cls is None:
        cls = type(tensor)
    return random_tensor(size, device, cls=cls, **kwargs)
# Standard normal (mean=0, std=1) tensor creation using scalar gauss from Random
def randn(size: Tuple[int, ...], device: Any = None, *, cls=None, **kwargs):
    """
    Create a tensor of the given shape filled with standard normal (mean=0, std=1) values.
    Uses the scalar gauss algorithm from the Random class.
    """
    from ..abstraction import AbstractTensor  # Local import to avoid circular dependency
    # Pull autograd controls out of RNG kwargs
    requires_grad = bool(kwargs.pop("requires_grad", False))
    tape = kwargs.pop("tape", None)
    cls = _resolve_cls(cls)
    total = 1
    for s in size:
        total *= s
    if 'batch_size' not in kwargs:
        kwargs = {**kwargs, 'batch_size': total}
    if 'distribution' not in kwargs:
        kwargs = {**kwargs, 'distribution': 'normal'}
    if 'kind' not in kwargs:
        if kwargs.get('seed') is not None:
            kwargs = {**kwargs, 'kind': RANDOM_KIND.PRNG}
        else:
            kwargs = {**kwargs, 'kind': RANDOM_KIND.CSPRNG}
    rng = random_generator(**kwargs)
    vals = next(rng)
    if not isinstance(vals, list):
        vals = [vals]
    with _CreationTapeCtx(requires_grad=requires_grad, tape=tape):
        inst = cls(track_time=False)
        inst.data = inst.tensor_from_list_(vals, kwargs.get('dtype'), device)
        inst = inst.reshape(*size)
    return _finalize_requires(inst, requires_grad)
