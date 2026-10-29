"""A helper called inside a loop over a bound tuple keeps its call record.

Unrolling ``for b in bits`` copies the call to ``helper`` once per element.
Each copy is the authored call: its exact return is read through the
authored call's reachability row (``unrolled_from``), not refused as a call
no source reaches.
"""

from __future__ import annotations

from src.common.tensors import AbstractTensor as AT
from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot


SOURCE = """
def helper(x, b):
    return x + b

def f(x, bits):
    acc = x < 0
    for b in bits:
        acc = acc | (helper(x, b) == 0)
    return acc
"""


def _calls(plan):
    for item in getattr(plan, "items", None) or ():
        callee = getattr(item, "callee", None)
        if callee is not None:
            yield callee
            yield from _calls(callee)
        yield from _calls(item)


def test_each_unrolled_call_is_planned_from_its_authored_call():
    compilation = compile_ast_aot(
        SOURCE,
        "f",
        {"x": AT.get_tensor([1, -5], dtype="int64"), "bits": (5, 7)},
        precompile_only=True,
        mutable_parameters=("x",),
    )

    helpers = [
        callee for callee in _calls(compilation.hierarchy_plan)
        if getattr(callee, "name", None) == "helper"
    ]
    # One call per element of ``bits``.
    assert len(helpers) == 2
