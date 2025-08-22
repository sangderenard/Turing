from __future__ import annotations
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .abstraction import AbstractTensor


def _is_seq(x: Any) -> bool:
    return isinstance(x, (list, tuple))


def _same_shape(shapes: List[Tuple[int, ...]]) -> bool:
    if not shapes:
        return True
    s0 = shapes[0]
    return all(s == s0 for s in shapes[1:])


def _maybe_cast_dtype_device(x: "AbstractTensor", *, dtype, device) -> "AbstractTensor":
    """Best-effort cast to dtype/device without assuming exact method names."""
    if dtype is not None:
        # Prefer common names; fall back if needed
        if hasattr(x, "astype") and callable(getattr(x, "astype")):
            x = x.astype(dtype)
        elif hasattr(x, "to") and callable(getattr(x, "to")):
            x = x.to(dtype=dtype)
    if device is not None:
        if hasattr(x, "to") and callable(getattr(x, "to")):
            # Torch-compatible signature often supports .to(device)
            try:
                x = x.to(device=device)
            except TypeError:
                # Some backends may expose .to_device
                if hasattr(x, "to_device"):
                    x = x.to_device(device)
        elif hasattr(x, "to_device") and callable(getattr(x, "to_device")):
            x = x.to_device(device)
    return x


def _leaf_to_abstract(
    leaf: Any, *, dtype, device, cls
) -> "AbstractTensor":
    """
    Coerce a single non-sequence leaf to an AbstractTensor on (dtype, device).
    This uses AbstractTensor.tensor() for leaves (safe: not a list), which correctly
    handles:
      - AbstractTensor -> short-circuit / backend conversion
      - torch.Tensor / np.ndarray / python scalars -> proper wrapping
    """
    out = cls.tensor(leaf, dtype=dtype, device=device)
    # Ensure dtype/device if cls.tensor didn’t enforce perfectly
    out = _maybe_cast_dtype_device(out, dtype=dtype, device=device)
    # Python scalars often materialize as shape (1,); squeeze to scalar
    if not hasattr(leaf, "__len__") and getattr(out, "get_shape", lambda: ())() == (1,):
        try:
            out = out.squeeze()
        except Exception:
            pass
    return out


def _pack_recursive(
    data: Any, *, dtype, device, cls
) -> "AbstractTensor":
    """
    Recursively convert arbitrarily-nested sequences into a single AbstractTensor by
    stacking along a new leading axis at each sequence level.

    Key properties:
      - Pre-existing AbstractTensors are respected (no re-wrapping into lists).
      - Mixed content (numbers, numpy/torch tensors, AbstractTensors) is promoted to AbstractTensor leaves.
      - Autograd provenance is preserved via stack/cat edges (no .tolist()/.item()).
      - Ragged structures raise an explicit error.
    """
    if _is_seq(data):
        if all(not _is_seq(elem) and not isinstance(elem, cls) for elem in data):
            out = cls.tensor(list(data), dtype=dtype, device=device)
            return _maybe_cast_dtype_device(out, dtype=dtype, device=device)

        # Recurse on children -> list of AbstractTensor
        children: List["AbstractTensor"] = [
            _pack_recursive(elem, dtype=dtype, device=device, cls=cls) for elem in data
        ]
        if not children:
            raise ValueError("Cannot pack an empty list/tuple into a tensor (ambiguous shape).")

        # Ensure all children shapes match for stacking
        shapes = [tuple(ch.get_shape()) for ch in children]
        if not _same_shape(shapes):
            raise ValueError(f"Ragged nested structure: encountered differing child shapes {shapes}.")

        # Ensure all are on same backend/dtype/device; cast if needed
        # Use the type of the first child to call stack
        head = children[0]
        children = [_maybe_cast_dtype_device(ch, dtype=dtype, device=device) for ch in children]

        # Prefer class-level stack to keep your abstraction consistent
        # AbstractTensor.stack([...], dim=0) should exist in your API (you’ve used it elsewhere).
        packed = cls.stack(children, dim=0)

        # Make contiguous if backend exposes it (avoid cost if not present)
        if hasattr(packed, "contiguous"):
            try:
                packed = packed.contiguous()
            except Exception:
                pass
        return packed

    # Leaf
    return _leaf_to_abstract(data, dtype=dtype, device=device, cls=cls)


def pack_nested_to_tensor(
    data: Any, *, dtype=None, device=None, cls=None
) -> "AbstractTensor":
    """
    Public API: recursively packs `data` into a single AbstractTensor.
    - Sequences become stacked tensors with a new leading dimension at each level.
    - Leaves are coerced to AbstractTensors with (dtype, device) if provided.
    - Preserves autograd provenance (no materialization to python lists).

    Example:
        t = pack_nested_to_tensor([a, b, c], dtype=a.dtype, device=a.device, cls=AbstractTensor)

    """
    # Fast-path: already an AbstractTensor and dtype/device match constraints
    if cls is None:
        from .abstraction import AbstractTensor as _AT  # local import to avoid cycle
        cls = _AT

    if isinstance(data, cls):
        out = data
        out = _maybe_cast_dtype_device(out, dtype=dtype, device=device)
        return out

    return _pack_recursive(data, dtype=dtype, device=device, cls=cls)
