"""``Cls.method(...)`` through a submitted class binds only a static method."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = str(ROOT / "extraction_contracts" / "program_extraction.yaml")


def _lower(source: str):
    return lower_ast_source_to_ssa(
        source, "root", name="cq", extraction_contract=CONTRACT,
        progress=lambda _: None,
    )


def test_instance_method_named_like_an_op_through_the_class_refuses():
    # Python binds ``self=n, a=4`` and returns 5.  A receiver edge cannot
    # state that binding, and left unlinked the call lowered as the ``min``
    # reduction over ``(Ops, n, 4)``.
    with pytest.raises(ValueError, match="non-static method has no receiver"):
        _lower(
            "class Ops:\n"
            "    def min(self, a: int) -> int:\n"
            "        return a + 1\n"
            "\n"
            "\n"
            "def root(n: int) -> int:\n"
            "    return Ops.min(n, 4)\n"
        )


def test_static_method_named_like_an_op_links_to_its_body():
    module, _outputs, exports = _lower(
        "class Ops:\n"
        "    @staticmethod\n"
        "    def min(a: int, b: int) -> int:\n"
        "        return a - b\n"
        "\n"
        "\n"
        "def root(n: int) -> int:\n"
        "    return Ops.min(n, 1)\n"
    )
    operations = {
        str(instruction.op).casefold()
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    }
    assert "sub" in operations
    assert "min" not in operations
