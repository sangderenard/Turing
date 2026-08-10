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

STRING_TOKEN = "string_token"


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
