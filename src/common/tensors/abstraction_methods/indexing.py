from __future__ import annotations

from typing import Any, Tuple


def unravel_index(indices: Any, shape: Tuple[int, ...]):
    """Map flat ``indices`` into coordinates for a tensor of ``shape``.

    Delegates to the backend-specific implementation ``unravel_index_`` after
    converting ``indices`` to an ``AbstractTensor`` instance.
    """
    from ..abstraction import AbstractTensor

    tensor = AbstractTensor.get_tensor(indices)
    return tensor.unravel_index_(shape)
