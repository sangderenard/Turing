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
        if runtime_sized and value_id in sources:
            record["extents_from"] = index_of[sources[value_id]]
        elif runtime_sized:
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
        "schema": SCHEMA,
        "entry": str(entry),
        "backend": str(backend),
        "signature": f"void {entry}(void **buffers, {extent_type} *extents)",
        "buffers": buffers,
        "extents": {"type": str(extent_type), "order": extents},
    }


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
