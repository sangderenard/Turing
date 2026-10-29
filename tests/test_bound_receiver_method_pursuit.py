"""A bound receiver's methods are pursued from the entry's own source.

The entry's caller is its host, so the host's actuals name each formal's
class, as an internal call edge does: ``self._peek`` resolves into the
receiver class's source even though only ``step`` is submitted.
"""

from __future__ import annotations

import inspect
import textwrap

from src.common.tensors import AbstractTensor as AT
from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot
from tests import bound_receiver_program as program


def _compiled():
    batch = program.Batch(
        octets=AT.get_tensor([[7, 8, 9], [4, 5, 6]], dtype="int64"),
    )
    return compile_ast_aot(
        textwrap.dedent(inspect.getsource(program.Reader.step)),
        "step",
        {
            "self": program.Reader(),
            "batch": batch,
            "cursor": AT.get_tensor([1, 5], dtype="int64"),
        },
        python_bindings={
            name: value
            for name, value in vars(program).items()
            if not name.startswith("__")
        },
        precompile_only=True,
        mutable_parameters=("cursor",),
    )


def _closure(plan, name):
    if getattr(plan, "name", None) == name:
        return plan
    for item in (getattr(plan, "items", None) or ()):
        found = _closure(item, name)
        if found is not None:
            return found
    callee = getattr(plan, "callee", None)
    return None if callee is None else _closure(callee, name)


def _lines(plan):
    for item in (getattr(plan, "items", None) or ()):
        if hasattr(item, "opcode"):
            yield item
        yield from _lines(item)


def _shapes(plan):
    shapes = {}
    for item in (getattr(plan, "items", None) or ()):
        for value_id, shape, dtype in getattr(item, "value_shapes", ()) or ():
            shapes[int(value_id)] = (tuple(shape), dtype)
        shapes.update(_shapes(item))
    return shapes


def test_a_method_called_on_self_is_compiled_from_the_receiver_class():
    peek = _closure(_compiled().hierarchy_plan, "_peek")

    assert peek is not None
    opcodes = [line.opcode for line in _lines(peek)]
    assert "gather" in opcodes and "to_dtype" in opcodes


def test_isinstance_on_a_bound_batch_prunes_the_other_arm():
    peek = _closure(_compiled().hierarchy_plan, "_peek")
    lines = list(_lines(peek))

    # ``batch`` is a ``Batch``: the ``PreparedBatch`` arm is gone and the
    # view is the one taken.
    assert [line.opcode for line in lines].count("reshape") == 1
    assert all(
        dict(line.attributes).get("attribute") != "flat_octets"
        for line in lines
    )


def test_a_view_and_a_gather_keep_their_extents_and_element_class():
    peek = _closure(_compiled().hierarchy_plan, "_peek")
    shapes = _shapes(peek)
    by_opcode = {
        line.opcode: shapes[int(line.outputs[0])] for line in _lines(peek)
        if line.opcode in {"reshape", "gather", "to_dtype"}
    }

    assert by_opcode["reshape"] == ((6,), "int64")
    assert by_opcode["gather"] == ((2,), "int64")
    assert by_opcode["to_dtype"] == ((2,), "int64")
