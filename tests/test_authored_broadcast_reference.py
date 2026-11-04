"""The authored extent law remains a completed SSA dependency component."""

from types import SimpleNamespace

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.authored_broadcast_reference import (
    authored_broadcast_call_actuals, with_authored_broadcast_extents,
)
from src.compiler.frame_identity_book import _frame_value_cell
from src.compiler.identity_concordance import (
    begin_identity_book,
    end_identity_book,
)


def test_authored_broadcast_reference_preserves_c_kernels_and_imports_owner():
    base = c_backend_repository_ssa_reference()
    result = with_authored_broadcast_extents(base)
    operation = result.operation("broadcast_shape_extents")
    assert operation is not None
    assert len(operation.entrypoints) == 1
    root = result.module.functions[operation.entrypoints[0]]
    component = root.metadata["accepted_ssa_component"]
    closure = result.dependency_closure(*operation.entrypoints)
    assert len(closure) > 1
    assert all(
        function.metadata["accepted_ssa_component"]["identity_book"]
        is component["identity_book"]
        for function in closure.values()
    )
    assert result.operation("topk") == base.operation("topk")
    assert result.operation("zeros") == base.operation("zeros")
    assert base.operation("broadcast_shape_extents") is None
    ambient = SimpleNamespace(metadata={"tensor_ssa_reference": result})
    assert with_authored_broadcast_extents(base, module=ambient) is result

    book, token = begin_identity_book()
    try:
        helper, actuals, formals = authored_broadcast_call_actuals(
            result,
            {name: name for name in (
                "rank", "start", "left", "right", "output",
            )},
        )
        assert helper is root
        assert actuals == ("rank", "start", "left", "right", "output")
        assert len(formals) == len(root.args)
        imported = _frame_value_cell(root, root.args[0].id)
        assert imported is not None
        assert imported.page.name == "concordance_imported_cell"
        assert imported.row[0] == component["artifact_sha256"]
        assert book.latest_ref(imported.page, imported.row) == imported
    finally:
        end_identity_book(token)
