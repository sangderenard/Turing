"""Backend-neutral folding of string words into tokens.

Runs over the shared ``FusedProgram`` IR, after the byte-string idiom fold (which
consumes literal ``bytes`` delimiters) and before container detection: every
``tensor_from_list`` holding a Python ``str`` becomes a ``string_token`` op
carrying its universal token (``string_table.string_token``), and every
comparison of a token is tagged so a backend compares the 64-bit identities
rather than the float value they are held as. A constant + constant string
concatenation folds to the interned concatenation's token. Runtime string
building (constant + a runtime string) is left untouched -- it needs a real
string builder, not interning.

The token assignment is universal; only the lowering (a token constant, a token
comparison) is per-backend, exactly like the container ops.
"""
from __future__ import annotations

from ..common.tensors.fused_ir import FusedProgram, OpStep
from .ir_string_ops import STRING_SPLIT_PART_HASH
from .string_table import string_token as _string_token

STRING_TOKEN = "string_token"


def tokenize_ssa_string_constants(functions, table=None) -> None:
    """Tokenize string constants across SSA functions, in place.

    The SSA-level analogue of :func:`intern_string_constants` (which folds a
    ``FusedProgram``), for the non-fused whole-object emission path. Every
    ``Const`` holding a Python ``str`` becomes a ``string_token`` op carrying the
    word's fnv1a-64 token; every ``equal``/``not_equal`` that consumes such a
    token is tagged ``string_compare`` so a backend compares the 64-bit
    identities rather than the float bits the token is held as. ``table`` (an
    optional ``StringTable``) records token -> word for reverse lookup; the token
    itself is content-addressed, so tokenizing never depends on it.

    ``functions`` is a mapping of name -> repository SSA ``Function``.
    """

    import dataclasses

    _COMPARES = ("equal", "not_equal", "Eq", "Ne")
    for function in functions.values():
        token_result_ids: set[int] = set()
        for block in function.blocks.values():
            rewritten = []
            for instruction in block.instrs:
                value = instruction.attributes.get("value")
                if value is None:
                    value = instruction.attributes.get("constant")
                if instruction.op in ("Const", "const") and isinstance(value, str):
                    token = table.intern(value) if table is not None else _string_token(value)
                    attributes = dict(instruction.attributes)
                    attributes.pop("value", None)
                    attributes.pop("constant", None)
                    attributes["token"] = int(token)
                    attributes["text"] = value
                    rewritten.append(
                        dataclasses.replace(
                            instruction, op=STRING_TOKEN, attributes=attributes
                        )
                    )
                    if instruction.res is not None:
                        token_result_ids.add(int(instruction.res.id))
                else:
                    rewritten.append(instruction)
            block.instrs = rewritten
        if not token_result_ids:
            continue
        for block in function.blocks.values():
            for index, instruction in enumerate(block.instrs):
                if instruction.op in _COMPARES and any(
                    int(argument.id) in token_result_ids
                    for argument in instruction.args
                ):
                    attributes = dict(instruction.attributes)
                    attributes["string_compare"] = True
                    block.instrs[index] = dataclasses.replace(
                        instruction, attributes=attributes
                    )


def _delim_byte(step: OpStep | None):
    """The delimiter byte of a split's separator constant -- a one-char str or a
    one-byte bytes."""
    if step is None or step.op_name != "tensor_from_list":
        return None
    v = step.attrs.get("values")
    if isinstance(v, (bytes, bytearray)) and len(v) == 1:
        return int(v[0])
    if isinstance(v, str) and len(v) == 1:
        return ord(v)
    return None


def _const_int(step: OpStep | None):
    if step is None or step.op_name != "tensor_from_list":
        return None
    try:
        return int(step.attrs.get("values"))
    except (TypeError, ValueError):
        return None


def fold_string_split(program: FusedProgram) -> FusedProgram:
    """Collapse ``x.split(delim, 1)[part]`` (part in {0,1}) into one
    ``string_split_part_hash`` op over ``x`` as a string view. ``delim`` may be a
    one-char str or one-byte bytes constant. The split subject ``x`` flows as a
    fat-pointer view (like a container base flows as a heap address); the
    coordinator seeds it, the op dereferences it. Value ids are preserved (the
    op reuses the idiom's output id) so cross-region wiring is intact.
    """

    steps = list(program.steps)
    by_result = {s.result_id: s for s in steps}
    dropped: set[int] = set()
    added: list[OpStep] = []
    for split in steps:
        if split.op_name != "split" or len(split.input_ids) < 2:
            continue
        delim = _delim_byte(by_result.get(split.input_ids[1]))
        if delim is None:
            continue
        view_id = split.input_ids[0]
        final = next(
            (t for t in steps
             if t.op_name in ("Indexed", "gather") and len(t.input_ids) == 2
             and t.input_ids[0] == split.result_id
             and _const_int(by_result.get(t.input_ids[1])) in (0, 1)),
            None,
        )
        if final is None:
            continue
        part = _const_int(by_result.get(final.input_ids[1]))
        dropped.update({split.result_id, final.result_id,
                        split.input_ids[1], final.input_ids[1]})
        if len(split.input_ids) >= 3:
            dropped.add(split.input_ids[2])
        added.append(OpStep(step_id=final.step_id, op_name=STRING_SPLIT_PART_HASH,
                            input_ids=[view_id], attrs={"delim": int(delim), "part": int(part)},
                            result_id=final.result_id))
    if not added:
        return program
    kept = [s for s in steps if s.result_id not in dropped]
    kept.extend(added)
    kept.sort(key=lambda s: s.step_id)
    return FusedProgram(
        version=program.version, feeds=set(program.feeds), steps=kept,
        outputs=dict(program.outputs), state_in=program.state_in,
        meta=program.meta, extras=program.extras,
    )


def _string_of(step: OpStep | None):
    if step is None or step.op_name == STRING_TOKEN:
        return step.attrs.get("text") if step is not None else None
    if step.op_name == "tensor_from_list" and isinstance(step.attrs.get("values"), str):
        return step.attrs["values"]
    return None


def intern_string_constants(program: FusedProgram, table) -> FusedProgram:
    """Return ``program`` with string constants interned to ``string_token`` ops
    and token comparisons tagged, or unchanged if it holds no string constants.
    ``table`` is a ``StringTable`` (records token -> string for reverse lookup).
    """

    steps = list(program.steps)
    if not any(s.op_name == "tensor_from_list" and isinstance(s.attrs.get("values"), str)
               for s in steps):
        return program

    by_result = {s.result_id: s for s in steps}
    token_of: dict[int, int] = {}

    # Constant string concatenation folds to one interned token: add(a, b) where
    # both are string constants becomes the token of a+b. (Runtime concat -- a
    # constant plus a runtime string -- is left for a real string builder.)
    concat_fold: dict[int, str] = {}
    for step in steps:
        if step.op_name != "add" or len(step.input_ids) != 2:
            continue
        left, right = (_string_of(by_result.get(i)) for i in step.input_ids)
        if left is not None and right is not None:
            concat_fold[step.result_id] = left + right

    new_steps: list[OpStep] = []
    for step in steps:
        if step.result_id in concat_fold:
            text = concat_fold[step.result_id]
            token = table.intern(text)
            token_of[step.result_id] = token
            new_steps.append(OpStep(step.step_id, STRING_TOKEN, [],
                                    {"token": int(token), "text": text}, step.result_id))
        elif step.op_name == "tensor_from_list" and isinstance(step.attrs.get("values"), str):
            text = step.attrs["values"]
            token = table.intern(text)
            token_of[step.result_id] = token
            new_steps.append(OpStep(step.step_id, STRING_TOKEN, [],
                                    {"token": int(token), "text": text}, step.result_id))
        else:
            new_steps.append(step)

    # Drop the constant string operands that fed a folded concat (now dead), and
    # tag comparisons that consume a token so the backend compares identities.
    consumed_by_concat = {
        i for step in steps if step.result_id in concat_fold for i in step.input_ids
    }
    tagged: list[OpStep] = []
    for step in new_steps:
        if step.result_id in concat_fold:
            tagged.append(step)
            continue
        if step.op_name in ("equal", "not_equal") and any(
            i in token_of for i in step.input_ids
        ):
            attrs = dict(step.attrs)
            attrs["string_compare"] = True
            tagged.append(OpStep(step.step_id, step.op_name, list(step.input_ids),
                                 attrs, step.result_id))
        else:
            tagged.append(step)
    tagged = [s for s in tagged
              if not (s.result_id in consumed_by_concat and s.op_name == STRING_TOKEN
                      and s.result_id not in concat_fold)]

    return FusedProgram(
        version=program.version, feeds=set(program.feeds), steps=tagged,
        outputs=dict(program.outputs), state_in=program.state_in,
        meta=program.meta, extras=program.extras,
    )
