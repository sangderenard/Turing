from __future__ import annotations

from typing import Any, Tuple, Optional, List, Union, Callable, Dict


def linspace(start, stop, steps, dtype=None, device=None):
    """Dispatch to backend-aware arange to produce linearly spaced values."""
    from ..abstraction import AbstractTensor  # Local import to avoid circular dependency

    if steps <= 0:
        raise ValueError("steps must be positive")
    i = AbstractTensor.arange(0, steps, 1, dtype=dtype, device=device)
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
            g = v.reshape(*shape).expand(full_shape)
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
            g = v.reshape(*shape).expand(final_shape)
            if copy:
                if hasattr(g, "clone"):
                    g = g.clone()
                elif hasattr(g, "copy"):
                    g = g.copy()
            grids.append(g)

    return MeshGrid(grids) if as_class else tuple(grids)
