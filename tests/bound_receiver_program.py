"""An authored class whose method is compiled from its own source alone.

Shaped like the x86 read head: ``step`` calls a private method on ``self``,
which unpacks a tensor's shape, branches on the receiver's class, views and
gathers.  Only ``step``'s source is handed to the compiler.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.tensors import AbstractTensor


@dataclass(frozen=True)
class Batch:
    octets: AbstractTensor


@dataclass(frozen=True)
class PreparedBatch:
    octets: AbstractTensor
    flat_octets: AbstractTensor


class Reader:
    def _peek(self, batch, cursor):
        lanes, capacity = (int(size) for size in batch.octets.shape)
        safe_cursor = cursor.maximum(0).minimum(max(0, capacity - 1))
        if isinstance(batch, PreparedBatch):
            flat_octets = batch.flat_octets
        else:
            flat_octets = batch.octets.reshape((-1,))
        lane = AbstractTensor.arange(lanes, cls=type(batch.octets))
        flat_index = lane * capacity + safe_cursor
        return flat_octets.gather(flat_index.to_dtype("int64"), dim=0)

    def step(self, batch, cursor):
        return self._peek(batch, cursor) + cursor
