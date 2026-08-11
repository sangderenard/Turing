"""Backend-neutral recognition of container (dict/list) subscript ops.

A shapeless subscript (a value with no tensor shape) is a dict/list container --
the decoder's opcode maps and token tables, keyed by unbounded RVAs/addresses or
by string names -- not a strided tensor. Recognising these, classifying their
keys, and interning string keys to a stable i64 are IR-level facts, independent
of how any one backend lowers them. This module holds that analysis so every
backend (WASM, C/PE, GLSL, Fortran, LLVM) shares one definition instead of each
re-deriving it; a backend imports these and only supplies its own lowering.
"""
from __future__ import annotations

from typing import Any, Sequence

from ..common.tensors.fused_ir import FusedProgram, OpStep


def is_shapeless(program: FusedProgram, value_id: int) -> bool:
    """A value with no tensor shape (``()`` or absent) -- a scalar reference into
    a dict/list container, not a shaped tensor buffer."""

    meta = (program.meta or {}).get(int(value_id))
    shape = tuple(meta.shape) if meta is not None and getattr(meta, "shape", None) else None
    return not shape


# FNV-1a 64-bit: a stable compile-time string hash so the same dict key interns
# to the same i64 map key in every region and every backend, without a shared
# intern table. Runtime-hashed names (see wasm_sequence.emit_hash_delimited_prefix)
# must fold with the same constants so a runtime name and a constant name collide.
FNV64_OFFSET = 0xCBF29CE484222325
FNV64_PRIME = 0x100000001B3


def fnv1a_64(text) -> int:
    # A word or a raw byte buffer hash the same way -- a str is its UTF-8 bytes,
    # so ``b'abc'`` and ``'abc'`` are one content identity.
    data = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    h = FNV64_OFFSET
    for byte in data:
        h = ((h ^ byte) * FNV64_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h - 2 ** 64 if h >= 2 ** 63 else h  # fold to signed i64


def string_constant(step: OpStep | None) -> str | None:
    """The Python string a ``tensor_from_list`` step materialises, else None."""
    if step is None or step.op_name != "tensor_from_list":
        return None
    values = step.attrs.get("values")
    return values if isinstance(values, str) else None


def constant_scalar_index(slices: Any) -> int | None:
    """The single integer subscript in an ``index_set``/``Indexed`` ``slices``
    attribute, or ``None`` if it is not a compile-time scalar.

    ``slices`` may be a plain int, an ``AbstractTensor``/ndarray wrapping one
    element, or a nested wrapper. A non-scalar (fancy or runtime index) returns
    ``None`` so the caller records an honest shortfall rather than guessing.
    """

    candidates = [slices, getattr(slices, "data", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            pass
        item = getattr(candidate, "item", None)
        if callable(item):
            try:
                return int(item())
            except Exception:
                pass
    try:
        import numpy as np

        array = np.asarray(slices)
        if array.size == 1:
            return int(array.reshape(()))
    except Exception:
        pass
    return None


def slice_selectors(step: OpStep):
    """Defunctionalized subscript descriptor -- one shape for both capture
    conventions of ``target[slices] = value``.

    ``index_set`` (from ``AbstractTensor.__setitem__``) carries inputs
    ``[data, value]`` with the subscript a constant in the ``slices`` attr.
    ``IndexedStore`` (the reference ``SubscriptStore`` path) carries the
    subscript as dataflow operands: ``[data, index0, (index1, ...), value]``.

    Returns ``(data_id, value_id, selectors)`` where each selector is
    ``("const", int)`` or ``("runtime", value_id)``, or ``None`` when the store
    is not (yet) lowerable this way -- e.g. a runtime ``index_set`` whose index
    lives only in the attr, never threaded as an operand.
    """

    op = step.op_name
    if op == "index_set" and len(step.input_ids) == 2:
        index = constant_scalar_index(step.attrs.get("slices"))
        if index is None:
            return None
        return step.input_ids[0], step.input_ids[1], [("const", index)]
    if op == "IndexedStore" and len(step.input_ids) >= 3:
        return (
            step.input_ids[0],
            step.input_ids[-1],
            [("runtime", value_id) for value_id in step.input_ids[1:-1]],
        )
    return None


def container_key_spec(program, by_result, feeds, ref):
    """Normalise a subscript operand to ('imm', i64) | ('feed', id), plus whether
    it was a string key. A string constant interns to its content token so a
    runtime-derived name and a constant name resolve to the same map slot --
    whether the token was assigned here (a raw ``str`` constant) or upstream by
    the string-interning fold (a ``string_token`` op)."""
    producer = by_result.get(ref)
    if producer is not None and producer.op_name == "string_token":
        return ("imm", int(producer.attrs["token"])), True
    text = string_constant(producer)
    if text is not None:
        return ("imm", fnv1a_64(text)), True
    if ref in feeds:
        return ("feed", ref), False
    if producer is not None and producer.op_name == "tensor_from_list":
        scalar = constant_scalar_index(producer.attrs.get("values"))
        if scalar is not None:
            return ("imm", int(scalar)), False
    return None, False


def pure_container_store(program: FusedProgram, live: Sequence[OpStep]):
    """A region that is one *shapeless* subscript store into a dict/list.

    Returns ``(data_id, value_spec, key_specs, result_id)`` or ``None``. Each
    spec is ``("imm", i64)`` (a constant int subscript, or a string key/value
    hashed to i64) or ``("feed", value_id)`` (read from a field at run time).
    Shapeless means the target is a container, lowered to a heap map rather than
    a strided buffer scatter.

    The store may be surrounded by ``tensor_from_list`` constant steps (its
    string/number key or value materialised inline). Two subscripts are always a
    nested container; a single subscript is a container only when its key is a
    string (a numeric single subscript is ambiguous with a 1-D buffer scatter,
    so it stays on the strided path).
    """

    stores = [s for s in live if s.op_name in ("index_set", "IndexedStore")]
    if len(stores) != 1:
        return None
    step = stores[0]
    by_result = {s.result_id: s for s in live}
    for other in live:
        if other is not step and other.op_name not in ("tensor_from_list", "string_token"):
            return None
    descriptor = slice_selectors(step)
    if descriptor is None:
        return None
    data_id, value_id, selectors = descriptor
    if not is_shapeless(program, step.result_id) or not is_shapeless(program, data_id):
        return None
    if list(program.outputs.values()) != [step.result_id]:
        return None
    feeds = set(program.feeds)
    if data_id not in feeds:
        return None

    key_specs = []
    has_string_key = False
    for kind, ref in selectors:
        if kind == "const":
            key_specs.append(("imm", int(ref)))
            continue
        spec, is_string = container_key_spec(program, by_result, feeds, ref)
        if spec is None:
            return None
        key_specs.append(spec)
        has_string_key = has_string_key or is_string
    value_spec, _ = container_key_spec(program, by_result, feeds, value_id)
    if value_spec is None:
        return None

    if len(key_specs) == 2 or (len(key_specs) == 1 and has_string_key):
        return data_id, value_spec, key_specs, step.result_id
    return None


def pure_container_read(program: FusedProgram, live: Sequence[OpStep]):
    """A region that is one *shapeless* subscript read from a dict/list.

    Returns ``(container_id, key_specs, result_id)`` or ``None``. The read
    counterpart of ``pure_container_store``: a chain of ``Indexed`` gathers
    ``table[gx]`` (single) or ``table[gx][gy]`` (nested) rooted at a shapeless
    container feed, ending at the sole region output, surrounded only by
    ``tensor_from_list`` key constants. Two levels are always a container; a
    single level only when the key is a string.
    """

    gathers = [s for s in live if s.op_name in ("Indexed", "gather")]
    if not gathers:
        return None
    for other in live:
        if other not in gathers and other.op_name not in ("tensor_from_list", "string_token"):
            return None
    outputs = list(program.outputs.values())
    if len(outputs) != 1:
        return None
    by_result = {s.result_id: s for s in live}
    feeds = set(program.feeds)

    chain: list[OpStep] = []
    current = by_result.get(outputs[0])
    while current is not None and current.op_name in ("Indexed", "gather"):
        if len(current.input_ids) != 2:
            return None
        chain.append(current)
        current = by_result.get(current.input_ids[0])
    chain.reverse()
    if not (1 <= len(chain) <= 2) or len(chain) != len(gathers):
        return None
    for parent, child in zip(chain, chain[1:]):
        if child.input_ids[0] != parent.result_id:
            return None
    container_id = chain[0].input_ids[0]
    if container_id not in feeds:
        return None
    if not is_shapeless(program, container_id) or not is_shapeless(program, outputs[0]):
        return None
    key_specs = []
    has_string = False
    for gather in chain:
        spec, is_string = container_key_spec(program, by_result, feeds, gather.input_ids[1])
        if spec is None:
            return None
        key_specs.append(spec)
        has_string = has_string or is_string
    if len(key_specs) == 2 or (len(key_specs) == 1 and has_string):
        return container_id, key_specs, outputs[0]
    return None
