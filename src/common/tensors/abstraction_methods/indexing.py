from __future__ import annotations

from typing import Any, Tuple


def unravel_index(indices: Any, shape: Tuple[int, ...]):
    """Map flat ``indices`` into coordinates for a tensor of ``shape``.

    Delegates to the backend-specific implementation ``unravel_index_`` after
    converting ``indices`` to an ``AbstractTensor`` instance.
    """
    # assume indices is already an AbstractTensor
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

