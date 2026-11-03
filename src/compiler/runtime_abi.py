"""The calling contract of a ``void entry(void **buffers, <int> *extents)``
artifact, as a host that owns its own tensor handles needs it.

The C and LLVM module lanes share one public ABI: a vector of buffer pointers
in ``buffer_order`` and a vector of runtime extents in ``extent_order``.  A
host that holds tensors (a row stack, a handle table) must know, for every
buffer, its element type, its rank independently of any concrete shape, and
where each runtime extent comes from -- and, for a runtime-sized result, which
input's extents it is allocated with.  Guessing any of these is not a wrong
number but a buffer overrun.

``runtime_abi_description`` states exactly that, from the artifact's own
emitted fields (the same tuples the emitter produced and the execution helpers
read), so it is a description of what was generated, never a second source of
truth.  Plain data: it serializes as JSON or YAML unchanged.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

SCHEMA = "turing-runtime-abi-v1"
COMPUTED_OUTPUT_SCHEMA = "turing-runtime-abi-v2"

#: A stored callable field crosses the native boundary as two method inputs,
#: a C function pointer and an opaque context handle, which the host supplies
#: on every call (docs/compiler_convergence/41_host_callback_abi_2026_10_24.md).
HOST_CALLBACK_SCHEMA = "turing-host-callback-v1"
HOST_CALLBACK_OPERATIONS = frozenset({"HostCallback", "HostCallbackPresent"})


def refuse_host_callbacks(module: Any, function_name: str, backend: str) -> None:
    """Refuse a host callback in a lane whose entry ABI cannot carry one.

    Only the Fortran class surface passes the callback's function pointer
    and context as entry inputs.  The buffer-vector entry of the C and LLVM
    lanes has no such slot, and an unknown operation there would be dropped
    rather than called.
    """

    pending, seen = [str(function_name)], set()
    while pending:
        name = pending.pop()
        if name in seen or name not in module.functions:
            continue
        seen.add(name)
        for block in module.functions[name].blocks.values():
            for instruction in block.instrs:
                if str(instruction.op) in HOST_CALLBACK_OPERATIONS:
                    field = instruction.attributes.get("host_callback_field")
                    raise ValueError(
                        f"{backend} emission cannot call host callback "
                        f"{field!r} in {name!r}: only the Fortran class "
                        f"surface carries {HOST_CALLBACK_SCHEMA} inputs"
                    )
                if str(instruction.op) == "Call":
                    pending.append(str(instruction.attributes.get("callee")))


def runtime_abi_description(
    *,
    entry: str,
    backend: str,
    extent_type: str,
    buffer_order: Iterable[int],
    buffer_dtypes: Iterable[str],
    buffer_shapes: Iterable[Iterable[Any]],
    buffer_ranks: Iterable[int],
    buffer_roles: Iterable[str],
    extent_order: Iterable[tuple[int, str, int | None]],
    output_extent_sources: Iterable[tuple[int, int]] = (),
    output_capacity_sources: Iterable[tuple[int, int]] = (),
    output_length_sources: Iterable[tuple[int, int]] = (),
    output_shape_sources: Iterable[tuple[int, int]] = (),
) -> dict[str, Any]:
    """Describe the buffer and extent vectors of one compiled entry.

    Every buffer states ``rank``.  ``shape`` is its static extents, or None
    when it is runtime-sized (rank known, extents supplied per call).  A
    runtime-sized input is measured by the host into the extents vector at
    the slots ``extents.order`` names; a runtime-sized output names the input
    buffer whose extents it takes, axis for axis (``extents_from``), and the
    host allocates it with them.
    """

    order = tuple(int(value_id) for value_id in buffer_order)
    dtypes = tuple(str(dtype) for dtype in buffer_dtypes)
    shapes = tuple(tuple(shape or ()) for shape in buffer_shapes)
    ranks = tuple(int(rank) for rank in buffer_ranks)
    roles = tuple(str(role) for role in buffer_roles)
    if not (len(order) == len(dtypes) == len(shapes) == len(ranks) == len(roles)):
        raise ValueError("runtime ABI buffer vectors disagree in length")
    index_of = {value_id: index for index, value_id in enumerate(order)}
    sources: Mapping[int, int] = {
        int(output): int(source) for output, source in output_extent_sources
    }
    def relations(pairs):
        result = {}
        for output, source in pairs:
            output, source = int(output), int(source)
            if output in result and result[output] != source:
                raise ValueError("runtime output identity has conflicting sources")
            result[output] = source
        return result
    capacities = relations(output_capacity_sources)
    lengths = relations(output_length_sources)
    result_shapes = relations(output_shape_sources)
    if set(lengths) & set(result_shapes):
        raise ValueError("a result cannot take both a logical length and a shape vector")
    from ..transmogrifier.dtype_layout import layout_for
    for output, source in (*capacities.items(), *lengths.items(), *result_shapes.items()):
        if output not in index_of or source not in index_of or output == source:
            raise ValueError("runtime output relation requires distinct public buffer identities")
        if ranks[index_of[output]] <= 0:
            raise ValueError("runtime output relation requires a ranked data buffer")
    for output, source in capacities.items():
        if (ranks[index_of[output]] != ranks[index_of[source]]
                or roles[index_of[source]] not in {"input", "inout"}):
            raise ValueError("runtime output capacity requires a same-rank public input")
    for output, source in (*lengths.items(), *result_shapes.items()):
        source_index = index_of[source]
        if layout_for(dtypes[source_index]).kind not in {"int", "uint"}:
            raise ValueError("native output extents require an integer buffer")
        expected_count = 1 if output in lengths else ranks[index_of[output]]
        shape = shapes[source_index]
        if not ((ranks[source_index] == 0 and shape == () and expected_count == 1)
                or (ranks[source_index] == 1 and shape == (expected_count,))):
            raise ValueError("native output extent port has the wrong fixed size")
    buffers = []
    for index, value_id in enumerate(order):
        static = shapes[index]
        runtime_sized = ranks[index] > len(static)
        record: dict[str, Any] = {
            "index": index,
            "value_id": value_id,
            "role": roles[index],
            "dtype": dtypes[index],
            "rank": ranks[index],
            "shape": None if runtime_sized else [int(item) for item in static],
        }
        if value_id in capacities:
            record["capacity_from"] = index_of[capacities[value_id]]
        if value_id in lengths:
            record["logical_length_from"] = index_of[lengths[value_id]]
        if value_id in result_shapes:
            record["logical_shape_from"] = index_of[result_shapes[value_id]]
        if runtime_sized and value_id in sources:
            record["extents_from"] = index_of[sources[value_id]]
        elif runtime_sized and value_id not in capacities:
            record["extents_from"] = "fed"
        buffers.append(record)
    extents = []
    for slot, (value_id, kind, axis) in enumerate(extent_order):
        if int(value_id) not in index_of:
            raise ValueError(
                f"extent slot {slot} measures value {value_id}, which is no "
                "public buffer"
            )
        extents.append({
            "slot": slot,
            "buffer": index_of[int(value_id)],
            "kind": str(kind),
            "axis": None if axis is None else int(axis),
        })
    return {
        "schema": (COMPUTED_OUTPUT_SCHEMA
                   if capacities or lengths or result_shapes else SCHEMA),
        "entry": str(entry),
        "backend": str(backend),
        "signature": f"void {entry}(void **buffers, {extent_type} *extents)",
        "buffers": buffers,
        "extents": {"type": str(extent_type), "order": extents},
    }


def apply_native_result_extents(
    buffers: dict[int, Any], capacities: Mapping[int, Any], *,
    length_sources: Iterable[tuple[int, int]] = (),
    shape_sources: Iterable[tuple[int, int]] = (),
) -> None:
    """Present native-written logical extents over caller-owned capacities.

    A byte slice may reserve its source's capacity while returning fewer rows.
    Reading its native length cell preserves that distinction; evaluating the
    source slice again on the host would create a second implementation.
    """
    import numpy as np

    views = {}
    for output, source in length_sources:
        output, source = int(output), int(source)
        allocated = capacities[output]
        native = np.asarray(buffers[source])
        if native.dtype.kind not in {"i", "u"} or native.size != 1:
            raise ValueError("native result length is not one integer")
        count = int(native.reshape(-1)[0])
        if allocated.ndim < 1 or not 0 <= count <= allocated.shape[0]:
            raise ValueError("native result length exceeds its proven capacity")
        views[output] = allocated[:count]
    for output, source in shape_sources:
        output, source = int(output), int(source)
        allocated = capacities[output]
        native = np.asarray(buffers[source])
        if native.dtype.kind not in {"i", "u"} or native.size != allocated.ndim:
            raise ValueError("native result shape has the wrong rank or element type")
        shape = tuple(int(value) for value in native.reshape(-1))
        if any(extent < 0 for extent in shape):
            raise ValueError("native result shape has a negative extent")
        count = 1
        for extent in shape:
            count *= extent
        if count > allocated.size:
            raise ValueError("native result shape exceeds its proven capacity")
        views[output] = allocated.reshape(-1)[:count].reshape(shape)
    buffers.update(views)


def public_buffer_contract(
    order: Iterable[int], inputs: Iterable[Any], outputs: Iterable[Any],
    rank_of,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """``(ranks, roles)`` of the public buffers, from the root's own values.

    A buffer that is both a root argument and a result is ``inout``; one that
    is neither (compiler-owned storage the wrapper exposes) is ``other``.
    """

    by_id: dict[int, Any] = {}
    input_ids = set()
    output_ids = set()
    for value in inputs:
        by_id.setdefault(int(value.id), value)
        input_ids.add(int(value.id))
    for value in outputs:
        by_id.setdefault(int(value.id), value)
        output_ids.add(int(value.id))
    ranks = []
    roles = []
    for value_id in order:
        value_id = int(value_id)
        value = by_id.get(value_id)
        ranks.append(0 if value is None else int(rank_of(value)))
        roles.append(
            "inout" if value_id in input_ids and value_id in output_ids
            else "input" if value_id in input_ids
            else "output" if value_id in output_ids
            else "other"
        )
    return tuple(ranks), tuple(roles)
