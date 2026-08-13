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
