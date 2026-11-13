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

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

SCHEMA = "turing-runtime-abi-v1"
COMPUTED_OUTPUT_SCHEMA = "turing-runtime-abi-v2"
OWNED_OUTPUT_SCHEMA = "turing-runtime-abi-v3"

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
    output_heap_sequences: Iterable[tuple[int, int, int, int, str]] = (),
    release_entry: str | None = None,
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
    owned = tuple((int(data), int(length), int(capacity), int(owner), str(dtype))
                  for data, length, capacity, owner, dtype in output_heap_sequences)
    owned_ids = {item[0] for item in owned}
    if len(owned_ids) != len(owned) or owned_ids & (set(capacities) | set(lengths) | set(result_shapes)):
        raise ValueError("owned output identity has conflicting allocation contracts")
    if bool(owned) != bool(release_entry):
        raise ValueError("owned outputs require their native release entry")
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
    owned_records = {}
    for data, length, capacity, owner, dtype in owned:
        if len({data, length, capacity, owner}) != 4 or any(
            value_id not in index_of for value_id in (data, length, capacity, owner)
        ):
            raise ValueError("owned output requires four distinct public descriptor ports")
        for value_id in (data, length, capacity, owner):
            index = index_of[value_id]
            if roles[index] not in {"output", "inout"} or shapes[index] not in {(), (1,)}:
                raise ValueError("owned output descriptor port is not one writable native cell")
        if any(dtypes[index_of[value_id]] not in {"ptr", "uintp"} for value_id in (data, owner)):
            raise ValueError("owned output data and owner must be native pointer cells")
        if any(layout_for(dtypes[index_of[value_id]]).kind not in {"int", "uint"}
               for value_id in (length, capacity)):
            raise ValueError("owned output length and capacity must be integers")
        layout = layout_for(dtype)
        if layout.kind not in {"bool", "int", "uint", "float", "complex"} or layout.byte_size > 8:
            raise ValueError("owned output element layout is not a fixed native value")
        owned_records[data] = {
            "element_dtype": dtype, "length_buffer": index_of[length],
            "capacity_buffer": index_of[capacity], "owner_buffer": index_of[owner],
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
        if value_id in capacities:
            record["capacity_from"] = index_of[capacities[value_id]]
        if value_id in lengths:
            record["logical_length_from"] = index_of[lengths[value_id]]
        if value_id in result_shapes:
            record["logical_shape_from"] = index_of[result_shapes[value_id]]
        if value_id in owned_records:
            record["owned_sequence"] = owned_records[value_id]
            record["owned_sequence"]["standalone_length_prefix"] = "uint64"
            record["storage"] = "owned_heap_pointer_cell"
            record["shape"] = [1]
        if runtime_sized and value_id in sources:
            record["extents_from"] = index_of[sources[value_id]]
        elif runtime_sized and value_id not in capacities and value_id not in owned_ids:
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
    description = {
        "schema": (OWNED_OUTPUT_SCHEMA if owned else COMPUTED_OUTPUT_SCHEMA
                   if capacities or lengths or result_shapes else SCHEMA),
        "entry": str(entry),
        "backend": str(backend),
        "signature": f"void {entry}(void **buffers, {extent_type} *extents)",
        "buffers": buffers,
        "extents": {"type": str(extent_type), "order": extents},
    }
    if owned:
        description["release_entry"] = {
            "entry": str(release_entry),
            "signature": f"void {release_entry}(void **buffers)",
            "lifetime": "consume results before releasing their native owners",
        }
    return description


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


@dataclass(frozen=True)
class OwnedGlyphSequenceCopy:
    """Exact returned child-pool ports supplied by a validated compiler receipt.

    The outer arena contains int64 row handles; the child arena contains int64
    glyph tokens, NOT Unicode code points. Capacity is flattened glyph storage,
    while stride is the per-row width. This is a low-level copy contract, not
    permission to export a descriptor or a Python string decoder.
    """

    data_id: int
    length_id: int
    capacity_id: int
    owner_id: int
    glyph_data_id: int
    glyph_lengths_id: int
    glyph_capacity_id: int
    glyph_stride_id: int

    def __post_init__(self):
        ids = tuple(vars(self).values())
        if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("owned glyph ports must be distinct nonnegative identities")


def _copy_owned_glyph_rows(spec, integer_cell, count, capacity, address, owner):
    """Copy validated row payloads while the caller retains the native owner."""
    import ctypes
    import numpy as np

    glyph_address = integer_cell(spec.glyph_data_id)
    lengths_address = integer_cell(spec.glyph_lengths_id)
    flattened = integer_cell(spec.glyph_capacity_id)
    stride = integer_cell(spec.glyph_stride_id)
    pointer_max, extent_max = int(np.iinfo(np.uintp).max), int(np.iinfo(np.intp).max)
    itemsize = np.dtype("int64").itemsize

    def arena(pointer, extent):
        if not 0 <= extent <= extent_max // itemsize:
            raise ValueError("owned glyph arena extent exceeds host address width")
        if not 0 <= pointer <= pointer_max:
            raise ValueError("owned glyph pointer exceeds native pointer width")
        size = extent * itemsize
        if size and (pointer == 0 or pointer > pointer_max - (size - 1)):
            raise ValueError("owned glyph arena has null or overflowing address")
        if pointer and pointer % ctypes.alignment(ctypes.c_int64):
            raise ValueError("owned glyph arena is not int64 aligned")

    if (not 0 <= stride <= extent_max // itemsize or capacity > extent_max // itemsize
            or flattened < 0 or capacity * stride > flattened):
        raise ValueError("owned glyph dimensions exceed flattened capacity")
    # Every scalar and arena bound is checked before reading any native row.
    arena(address, count)
    # Existing allocation reserves the outer arena independently but creates
    # child arenas at zero extent until the first append. An empty result
    # therefore need not have a child-length arena even with outer capacity.
    arena(lengths_address, capacity if count else 0)
    arena(glyph_address, flattened)
    if count and not owner:
        raise ValueError("owned glyph rows have no live owner")
    if not count:
        return ()

    handles = np.ctypeslib.as_array((ctypes.c_int64 * count).from_address(address)).copy()
    if any(not 0 <= int(handle) < capacity for handle in handles):
        raise ValueError("owned glyph handle exceeds child length arena")
    lengths = []
    for handle in handles:
        length = ctypes.c_int64.from_address(lengths_address + int(handle) * itemsize).value
        if not 0 <= length <= stride or int(handle) * stride + length > flattened:
            raise ValueError("owned glyph row length exceeds its stride or arena")
        lengths.append(length)
    # No payload read occurs until every selected handle and length validates.
    result = []
    for handle, length in zip(handles, lengths):
        if length:
            row = (ctypes.c_int64 * length).from_address(
                glyph_address + int(handle) * stride * itemsize)
            result.append(np.ctypeslib.as_array(row).copy())
        else:
            result.append(np.empty((0,), dtype=np.int64))
    return tuple(result)


def copy_owned_sequence_results(
    buffers: Mapping[int, Any],
    sequences: Iterable[tuple[int, int, int, int, str]],
    *, glyph_sequences: Iterable[OwnedGlyphSequenceCopy] = (),
) -> dict[int, Any]:
    """Read an incumbent owned-heap sequence before releasing its owner.

    Ports are data-pointer cell, length, capacity, owner and element dtype,
    derived from the returned sequence descriptor. The caller keeps the
    native owner alive through this copy and releases it even on failure.
    Optional glyph specs replace only the selected data result with a tuple of
    independent int64 glyph arrays, never decoded Python strings. Numeric
    address/extent checks do not prove pointer validity: validated compiler
    provenance and a live native allocation are caller prerequisites.
    """
    import ctypes
    import numpy as np

    sequences = tuple(sequences)
    glyphs = {}
    for spec in glyph_sequences:
        if not isinstance(spec, OwnedGlyphSequenceCopy):
            raise ValueError("owned glyph copy requires typed ports")
        if spec.data_id in glyphs:
            raise ValueError("duplicate owned glyph result")
        matches = [ports for ports in sequences if ports[:4] == (
            spec.data_id, spec.length_id, spec.capacity_id, spec.owner_id)]
        if len(matches) != 1 or np.dtype(matches[0][4]) != np.dtype("int64"):
            raise ValueError("owned glyph ports do not match one int64 outer result")
        output_ids = {spec.data_id, spec.length_id, spec.capacity_id}
        if any(output_ids.intersection(ports[:3]) for ports in sequences if ports is not matches[0]):
            raise ValueError("owned glyph result collides with another output")
        glyphs[spec.data_id] = spec

    def integer_cell(value_id):
        cell = np.asarray(buffers[int(value_id)])
        if cell.size != 1 or cell.dtype.kind not in {"i", "u"}:
            raise ValueError("owned sequence metadata is not one integer cell")
        return int(cell.reshape(-1)[0])

    copied = {}
    for data_id, length_id, capacity_id, owner_id, dtype in sequences:
        count, capacity = integer_cell(length_id), integer_cell(capacity_id)
        address, owner = integer_cell(data_id), integer_cell(owner_id)
        element = np.dtype(dtype)
        if (element.hasobject or element.kind not in {"b", "i", "u", "f", "c"}
                or element.itemsize > 8):
            raise ValueError("owned sequence has no fixed native element layout")
        if not 0 <= count <= capacity:
            raise ValueError("owned sequence length exceeds its native capacity")
        if not 0 <= address <= np.iinfo(np.uintp).max or not 0 <= owner <= np.iinfo(np.uintp).max:
            raise ValueError("owned sequence address exceeds native pointer width")
        if count > np.iinfo(np.intp).max // element.itemsize:
            raise ValueError("owned sequence byte count exceeds host address width")
        if count and (address == 0 or owner == 0):
            raise ValueError("nonempty owned sequence has no live arena and owner")
        if data_id in glyphs:
            copied[int(data_id)] = _copy_owned_glyph_rows(
                glyphs[data_id], integer_cell, count, capacity, address, owner)
        elif count:
            octets = (ctypes.c_ubyte * (count * element.itemsize)).from_address(address)
            copied[int(data_id)] = np.ctypeslib.as_array(octets).view(element).copy()
        else:
            copied[int(data_id)] = np.empty((0,), dtype=element)
        for value_id in (length_id, capacity_id):
            copied[int(value_id)] = np.asarray(buffers[int(value_id)]).copy()
    return copied


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
