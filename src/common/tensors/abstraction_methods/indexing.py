from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class NormalizedIndexAxis:
    indices: Sequence[int] | np.ndarray
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
            # Keep a slice symbolic and constant-size. Expanding a full image
            # axis into a Python tuple made GPU indexing spend more time
            # manufacturing host integers than dispatching its gather.
            indices = range(*item.indices(axis_size))
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
                    normalized.reshape(-1),
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

    Returns one tensor per axis, row-major, for scalar or array ``indices``.
    The default ``unravel_index_`` is built from ``%`` and ``//`` alone, so
    this works on every backend; a backend only overrides it to go faster.
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
   
def _integer_positions(index: Any):
    """``index`` as a flat list of ints, or ``None`` if it is not one.

    Slices, boolean masks and anything else a caller may legitimately
    hand to fancy indexing come back as ``None`` so they keep taking the
    single-pass path they always took.
    """
    if isinstance(index, slice):
        return None
    raw = index.tolist() if hasattr(index, "tolist") else index
    if not isinstance(raw, (list, tuple)):
        return None
    positions = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if isinstance(value, float) and value != int(value):
            return None
        positions.append(int(value))
    return positions


def _rounds_without_repeats(positions):
    """Group positions so that no round names the same destination twice.

    Round ``r`` holds the ``r``-th occurrence of each destination, so
    every round can be applied in one vectorised pass and the number of
    rounds is the largest multiplicity present -- a vertex's degree, for
    the graph and mesh cases this exists to serve.
    """
    seen = {}
    rounds = []
    for position, destination in enumerate(positions):
        rank = seen.get(destination, 0)
        seen[destination] = rank + 1
        if rank == len(rounds):
            rounds.append([])
        rounds[rank].append(position)
    return rounds


#: The aggregation vocabulary is not invented here.  ``SegmentReduce`` in
#: ``abstract_graph_core`` already declares exactly these four for the
#: graph tier's ``coalesce_edges``; this is the same policy applied to a
#: dense scatter, spelled the same way so there is one word per policy in
#: the tree rather than two.
SCATTER_REDUCTIONS = ("sum", "mean", "max", "min")


def scatter(x: Any, index: Any, src: Any, dim: int = 0, *, reduce: str = "sum"):
    """Scatter ``src`` into ``x`` along ``dim`` at positions ``index``.

    ``reduce`` says what happens where several sources name one
    destination, using the ``SegmentReduce`` vocabulary declared in
    ``abstract_graph_core``: ``"sum"`` (the default), ``"mean"``,
    ``"max"`` or ``"min"``.  Every policy INCLUDES the value already in
    ``x`` as one of the values being combined, so ``"sum"`` is
    ``x_i + sum(src)``, ``"max"`` is ``max(x_i, src...)``, and ``"mean"``
    averages ``x_i`` together with the contributions.

    Repeated destinations therefore ACCUMULATE under the default, which
    is what the name has always promised, what this module's backward
    rule in ``backward_registry`` was already written for
    (``y_i = x_i + src_j``, adjoint ``gsrc = g[index]``), and what the
    GLSL backend's ``scatter_snippet`` has always done on the GPU.  Plain
    fancy-index assignment cannot do it -- ``a[idx] = v`` keeps only the
    last write -- so destinations are grouped into rounds that each name
    every destination at most once, and each round is one vectorised
    pass.  The number of rounds is the largest multiplicity present: a
    vertex's degree, for the graph and mesh cases this exists to serve.

    With ``reduce="sum"`` and no repeated destination there is exactly
    one round and the result is identical, entry for entry, to the single
    assignment this used to perform; the same holds when ``index`` is a
    slice or a mask rather than a list of positions.  So only genuinely
    repeated destinations change, and for those the previous answer
    silently dropped every contribution but one.
    """
    if reduce not in SCATTER_REDUCTIONS:
        raise ValueError(
            f"scatter reduce={reduce!r} is not one of {SCATTER_REDUCTIONS}")
    nd = x.ndims()
    axis = dim if dim >= 0 else nd + dim
    from ..abstraction import AbstractTensor
    finalize = AbstractTensor._pre_autograd('scatter', [x, index, src], params={'dim': dim})
    result = x.clone()

    positions = _integer_positions(index)
    if reduce == "sum" and (positions is None
                            or len(set(positions)) == len(positions)):
        indexer = [slice(None)] * nd
        indexer[axis] = index
        result[tuple(indexer)] = x[tuple(indexer)] + src
        return finalize(result)
    if positions is None:
        raise ValueError(
            f"scatter reduce={reduce!r} needs an integer index, "
            f"not {type(index).__name__}")

    # ``src`` normally carries one entry per destination along ``axis``;
    # anything else is a broadcast value that every round reuses whole.
    src_tensor = src if isinstance(src, AbstractTensor) else AbstractTensor.get_tensor(src)
    shape = tuple(src_tensor.get_shape())
    per_destination = len(shape) == nd and shape[axis] == len(positions)

    for group in _rounds_without_repeats(positions):
        indexer = [slice(None)] * nd
        indexer[axis] = [positions[position] for position in group]
        if per_destination:
            picker = [slice(None)] * nd
            picker[axis] = group
            piece = src_tensor[tuple(picker)]
        else:
            piece = src_tensor
        # Read from ``result``, not ``x``: later rounds must land on top
        # of what earlier rounds already combined.
        standing = result[tuple(indexer)]
        if reduce == "max":
            result[tuple(indexer)] = standing.maximum(piece)
        elif reduce == "min":
            result[tuple(indexer)] = standing.minimum(piece)
        else:
            result[tuple(indexer)] = standing + piece

    if reduce == "mean":
        multiplicity = {}
        for destination in positions:
            multiplicity[destination] = multiplicity.get(destination, 0) + 1
        targets = sorted(multiplicity)
        # ``x``'s own value is one of the values being averaged, so the
        # divisor is one more than the number of contributions.
        divisor = AbstractTensor.get_tensor(
            [float(1 + multiplicity[target]) for target in targets])
        spread = [1] * nd
        spread[axis] = len(targets)
        indexer = [slice(None)] * nd
        indexer[axis] = targets
        result[tuple(indexer)] = (
            result[tuple(indexer)] / divisor.reshape(tuple(spread)))
    return finalize(result)

