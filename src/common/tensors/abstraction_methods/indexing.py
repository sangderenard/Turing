from __future__ import annotations

from typing import Any, Tuple


def normalize_basic_index(index: Any, shape: Tuple[int, ...]):
    """Normalize basic indexing into per-axis indices and output shape.

    Integer axes are retained as one-element selections but omitted from the
    logical output shape. This gives read and write lowerings one shared
    interpretation of integers, slices, negative indices, and ellipses.
    """
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

    axes = []
    output_shape = []
    for axis_size, item in zip(shape, items):
        if isinstance(item, int):
            if item < -axis_size or item >= axis_size:
                raise IndexError("tensor index out of range")
            axes.append(([item % axis_size], True))
        elif isinstance(item, slice):
            indices = list(range(*item.indices(axis_size)))
            axes.append((indices, False))
            output_shape.append(len(indices))
        else:
            raise NotImplementedError(
                "basic indexing supports integers, slices, and ellipses"
            )
    return axes, tuple(output_shape)


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
    axes, _ = normalize_basic_index(index, tuple(shape_of(data)))
    current = data
    output_axis = 0
    for indices, drop_axis in axes:
        if drop_axis:
            selected = index_select(current, output_axis, indices)
            selected_shape = shape_of(selected)
            current = reshape(
                selected,
                selected_shape[:output_axis]
                + selected_shape[output_axis + 1:],
            )
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

