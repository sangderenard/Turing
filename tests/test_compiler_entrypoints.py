import warnings

from src.compiler.compiler_entrypoints import CANONICAL_SOURCE_COMPILER
from src.compiler.fortran_c_shell import (
    compile_ast_fortran_c_shell,
    lower_ast_source_to_ssa,
)
from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot


def test_complete_source_ssa_is_the_declared_canonical_compiler():
    assert CANONICAL_SOURCE_COMPILER.endswith(".lower_ast_source_to_ssa")
    assert lower_ast_source_to_ssa.__canonical_source_compiler__ is True


def test_legacy_whole_source_entries_are_deprecated_before_they_do_work():
    for function, arguments in (
        (compile_ast_aot, ("", "missing", {})),
        (compile_ast_fortran_c_shell, ("", "missing", {}, ".")),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                function(*arguments)
            except Exception:
                pass
        messages = tuple(str(item.message) for item in caught)
        assert any(
            "deprecated source-compilation entry point" in message
            and "lower_ast_source_to_ssa" in message
            for message in messages
        )


def test_omitted_entrypoint_lowers_complete_class_definition():
    module, _outputs, exports = lower_ast_source_to_ssa("""
class Pair:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def total(self):
        return self.left + self.right

def helper(value):
    return value + 1
""")

    pair = module.class_table.by_identity("Pair")
    assert pair is not None
    assert [field.name for field in pair.fields] == ["left", "right"]
    assert {method.name for method in pair.methods} == {"__init__", "total"}
    assert all(method.function_name in module.functions for method in pair.methods)
    assert any(name.endswith("__helper") for name in exports)
