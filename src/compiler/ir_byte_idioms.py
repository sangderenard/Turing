"""Backend-neutral IR folding of byte-string idioms.

A region kernel holds scalars, but source code does real byte-string work. Rather
than teach every backend to pattern-match the multi-step shape, the recognition
happens ONCE here, over the shared ``FusedProgram`` IR: an idiom collapses to a
single general op that every backend (WASM, C/PE, GLSL, Fortran, LLVM) lowers its
own way.

Currently folded:

* ``hash_delimited_prefix`` -- ``subject[start].split(delim, 1)[0]``, the
  null-terminated name extraction (PE section names, binary_ingestion.py). Its
  result is a name used downstream as a dict key, so it is the FNV-1a hash of the
  bytes before the first delimiter, bounded by the field width. Inputs
  ``[subject, start]``; attrs ``{delim, maxlen}``.
"""
from __future__ import annotations

from ..common.tensors.fused_ir import FusedProgram, OpStep
from .ir_container_ops import fnv1a_64  # noqa: F401 -- shared key identity

HASH_DELIMITED_PREFIX = "hash_delimited_prefix"

# PE section-name fields are 8 bytes, null-padded; this is the only split-on-null
# idiom in the codebase, so the bound is fixed.
_DEFAULT_NAME_WIDTH = 8


def _const_int(step: OpStep | None):
    if step is None or step.op_name != "tensor_from_list":
        return None
    values = step.attrs.get("values")
    try:
        return int(values)
    except (TypeError, ValueError):
        return None


def _bytes_first_byte(step: OpStep | None):
    if step is None or step.op_name != "tensor_from_list":
        return None
    values = step.attrs.get("values")
    if isinstance(values, (bytes, bytearray)) and len(values) >= 1:
        return int(values[0])
    return None


def fold_byte_string_idioms(program: FusedProgram) -> FusedProgram:
    """Return ``program`` with recognised byte-string idioms replaced by a single
    general op, or the program unchanged if none match. Value ids are preserved
    (the fold reuses the idiom's output id), so cross-region wiring is intact.
    """

    steps = list(program.steps)
    by_result = {s.result_id: s for s in steps}
    dropped: set[int] = set()
    added: list[OpStep] = []

    for split in steps:
        if split.op_name != "split" or len(split.input_ids) < 2:
            continue
        sliced = by_result.get(split.input_ids[0])
        if (sliced is None or sliced.op_name not in ("Indexed", "gather")
                or len(sliced.input_ids) != 2):
            continue
        delim = _bytes_first_byte(by_result.get(split.input_ids[1]))
        if delim is None:
            continue
        # The result must be taken as element [0] of the split.
        final = next(
            (t for t in steps
             if t.op_name in ("Indexed", "gather") and len(t.input_ids) == 2
             and t.input_ids[0] == split.result_id
             and _const_int(by_result.get(t.input_ids[1])) == 0),
            None,
        )
        if final is None:
            continue
        subject_id, start_id = sliced.input_ids
        # Drop the idiom's intermediate steps and its constant materialisations;
        # keep the subject/start feeds.
        dropped.update({sliced.result_id, split.result_id, final.result_id,
                        split.input_ids[1], final.input_ids[1]})
        if len(split.input_ids) >= 3:
            dropped.add(split.input_ids[2])
        added.append(OpStep(
            step_id=final.step_id, op_name=HASH_DELIMITED_PREFIX,
            input_ids=[subject_id, start_id],
            attrs={"delim": int(delim), "maxlen": _DEFAULT_NAME_WIDTH},
            result_id=final.result_id,
        ))

    if not added:
        return program
    kept = [s for s in steps if s.result_id not in dropped]
    kept.extend(added)
    kept.sort(key=lambda s: s.step_id)
    return FusedProgram(
        version=program.version,
        feeds=set(program.feeds),
        steps=kept,
        outputs=dict(program.outputs),
        state_in=program.state_in,
        meta=program.meta,
        extras=program.extras,
    )
