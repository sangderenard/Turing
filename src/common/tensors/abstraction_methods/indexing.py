from __future__ import annotations

from typing import Any, Tuple


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
    items = list(index) if isinstance(index, tuple) else [index]
    if any(item is Ellipsis for item in items):
        location = next(
            position for position, item in enumerate(items)
            if item is Ellipsis
        )
        missing = len(shape_of(data)) - (len(items) - 1)
        items[location:location + 1] = [slice(None)] * missing
    items.extend([slice(None)] * (len(shape_of(data)) - len(items)))
    current = data
    output_axis = 0
    for item in items:
        current_shape = shape_of(current)
        axis_size = current_shape[output_axis]
        if isinstance(item, int):
            selected = index_select(
                current, output_axis, [item % axis_size]
            )
            selected_shape = shape_of(selected)
            current = reshape(
                selected,
                selected_shape[:output_axis]
                + selected_shape[output_axis + 1:],
            )
        elif isinstance(item, slice):
            indices = list(range(*item.indices(axis_size)))
            current = index_select(current, output_axis, indices)
            output_axis += 1
        else:
            raise NotImplementedError(
                "basic indexing supports integers, slices, and ellipses"
            )
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

