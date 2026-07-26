from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np


@dataclass(frozen=True)
class NormalizedIndexAxis:
    indices: tuple[int, ...]
    drop_axis: bool
    index_shape: tuple[int, ...] | None = None

    @property
    def advanced(self) -> bool:
        return self.index_shape is not None


def normalize_index(index: Any, shape: Tuple[int, ...]):
    """Normalize basic indexing plus one shaped integer-array index."""

    items = list(index) if isinstance(index, tuple) else [index]
    ellipses = sum(item is Ellipsis for item in items)
    if ellipses > 1:
        raise IndexError("an index can only have a single ellipsis")
    if ellipses:
        location = next(
            position for position, item in enumerate(items)
            if item is Ellipsis
        )
        missing = len(shape) - (len(items) - 1)
        if missing < 0:
            raise IndexError("too many indices for tensor")
        items[location:location + 1] = [slice(None)] * missing
    if len(items) > len(shape):
        raise IndexError("too many indices for tensor")
    items.extend([slice(None)] * (len(shape) - len(items)))

    axes: list[NormalizedIndexAxis] = []
    output_shape: list[int] = []
    advanced_count = 0
    for axis_size, item in zip(shape, items):
        if isinstance(item, (int, np.integer)):
            value = int(item)
            if value < -axis_size or value >= axis_size:
                raise IndexError("tensor index out of range")
            axes.append(NormalizedIndexAxis((value % axis_size,), True))
        elif isinstance(item, slice):
            indices = tuple(range(*item.indices(axis_size)))
            axes.append(NormalizedIndexAxis(indices, False))
            output_shape.append(len(indices))
        else:
            raw = item.tolist() if hasattr(item, "tolist") else item
            array = np.asarray(raw)
            if array.dtype.kind not in "iu":
                raise TypeError("advanced tensor indices must be integers")
            advanced_count += 1
            if advanced_count > 1:
                raise NotImplementedError(
                    "at most one integer-array index is currently supported"
                )
            normalized = array.astype(np.int64, copy=False)
            normalized = np.where(
                normalized < 0, normalized + axis_size, normalized
            )
            if np.any((normalized < 0) | (normalized >= axis_size)):
                raise IndexError("tensor index out of range")
            index_shape = tuple(int(size) for size in normalized.shape)
            axes.append(
                NormalizedIndexAxis(
                    tuple(int(value) for value in normalized.reshape(-1)),
                    False,
                    index_shape,
                )
            )
            output_shape.extend(index_shape)
    return tuple(axes), tuple(output_shape)


def normalize_basic_index(index: Any, shape: Tuple[int, ...]):
    """Normalize basic indexing into per-axis indices and output shape.

    Integer axes are retained as one-element selections but omitted from the
    logical output shape. This gives read and write lowerings one shared
    interpretation of integers, slices, negative indices, and ellipses.
    """
    normalized, output_shape = normalize_index(index, shape)
    axes = []
    for axis in normalized:
        if axis.advanced:
            raise NotImplementedError(
                "basic indexing supports integers, slices, and ellipses"
            )
        axes.append((list(axis.indices), axis.drop_axis))
    return axes, output_shape


def flat_index_ids(index: Any, shape: Tuple[int, ...]) -> np.ndarray:
    """Return row-major source offsets selected by a normalized index."""

    axes, output_shape = normalize_index(index, shape)
    strides = []
    running = 1
    for size in reversed(shape):
        strides.append(running)
        running *= size
    strides.reverse()
    offsets = np.zeros(output_shape, dtype=np.int64)
    output_axis = 0
    for axis, stride in zip(axes, strides):
        if axis.drop_axis:
            offsets += axis.indices[0] * stride
            continue
        local_shape = axis.index_shape or (len(axis.indices),)
        coordinate = np.asarray(axis.indices, dtype=np.int64).reshape(
            local_shape
        )
        broadcast_shape = (
            (1,) * output_axis
            + local_shape
            + (1,) * (len(output_shape) - output_axis - len(local_shape))
        )
        offsets += coordinate.reshape(broadcast_shape) * stride
        output_axis += len(local_shape)
    return offsets


def lower_basic_index(
    data: Any,
    index: Any,
    *,
    shape_of,
    index_select,
    reshape,
):
    """Compose basic tuple indexing from index-select and metadata reshape.

    Backends with rich native indexing may bypass this helper. Lowering
    targets can reuse it so integers, slices, negative indices, and ellipses
    share one AbstractTensor-level policy while numerical gathering remains a
    backend primitive.
    """
    axes, _ = normalize_index(index, tuple(shape_of(data)))
    current = data
    output_axis = 0
    for axis in axes:
        indices = list(axis.indices)
        if axis.drop_axis:
            selected = index_select(current, output_axis, indices)
            selected_shape = shape_of(selected)
            current = reshape(
                selected,
                selected_shape[:output_axis]
                + selected_shape[output_axis + 1:],
            )
        elif axis.advanced:
            selected = index_select(current, output_axis, indices)
            selected_shape = shape_of(selected)
            current = reshape(
                selected,
                selected_shape[:output_axis]
                + axis.index_shape
                + selected_shape[output_axis + 1:],
            )
            output_axis += len(axis.index_shape)
        else:
            current = index_select(current, output_axis, indices)
            output_axis += 1
    return current


def unravel_index(indices: Any, shape: Tuple[int, ...]):
    """Map flat ``indices`` into coordinates for a tensor of ``shape``.

    Delegates to the backend-specific implementation ``unravel_index_`` after
    converting ``indices`` to an ``AbstractTensor`` instance.
    """
    from ..abstraction import AbstractTensor
    if not isinstance(indices, AbstractTensor):
        indices = AbstractTensor.get_tensor(indices).to_dtype(
            AbstractTensor.long_dtype_
        )
    return indices.unravel_index_(shape)
    
    
def gather(x: Any, index: Any, dim: int = 0):
    """Gather elements from x along axis dim using integer indices."""
    # build index tuple
    nd = x.ndims()
    axis = dim if dim >= 0 else nd + dim
    indexer = [slice(None)] * nd
    indexer[axis] = index
    # select
    out = x[tuple(indexer)]
    # record autograd
    from ..abstraction import AbstractTensor
    finalize = AbstractTensor._pre_autograd('gather', [x, index], params={'dim': dim})
    return finalize(out)
   
def scatter(x: Any, index: Any, src: Any, dim: int = 0):
    """Scatter-add src into x along axis dim at positions given by index."""
    # build index tuple
    nd = x.ndims()
    axis = dim if dim >= 0 else nd + dim
    indexer = [slice(None)] * nd
    indexer[axis] = index
    # perform in-place scatter-add
    # record autograd before update
    from ..abstraction import AbstractTensor
    finalize = AbstractTensor._pre_autograd('scatter', [x, index, src], params={'dim': dim})
    # fetch existing values and add
    old = x[tuple(indexer)]
    x[tuple(indexer)] = old + src
    return finalize(x)

