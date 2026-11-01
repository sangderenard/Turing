"""Actual authored synchronization through SSA, C emission and native execution."""

import pytest

from tests.test_source_ssa_synchronization import compile_source
from src.compiler.concordance_declarations import C_SYNCHRONIZATION_POLICY
from src.compiler.identity_concordance import concordance_report
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


@pytest.mark.parametrize("operation,intrinsic", [
    ("compiler_fence", "atomic_signal_fence"),
    ("memory_barrier", "atomic_thread_fence"),
])
def test_source_fences_compile_and_execute_with_exact_c11_policy(tmp_path, operation, intrinsic):
    calls = []
    scopes = (4,) if operation == "compiler_fence" else (4, 256, 257)
    for scope in scopes:
        for semantics in (2, 4, 8, 16):
            operands = f"{semantics}, 1" if operation == "compiler_fence" else f"{scope}, {semantics}, 1"
            calls.append(f"    {operation}({operands})")
    source = (f"from src.compiler.synchronization import {operation}\ndef root():\n"
              + "\n".join(calls) + "\n    return 41\n")
    module = compile_source(source)
    assert "0 finding(s)" in concordance_report(module)
    artifact = emit_ssa_module_to_c(module, "root__root")
    assert artifact.complete, artifact.shortfalls
    page = module.metadata["identity_book"].page(C_SYNCHRONIZATION_POLICY)
    policies = [page.latest(row) for row in page.rows()]
    assert len(policies) == len(calls)
    assert {policy[0] for policy in policies} == {intrinsic}
    assert {policy[2] for policy in policies} == set(scopes)
    assert {policy[3] for policy in policies} == {2, 4, 8, 16}
    assert artifact.source.count(f"{intrinsic}(") == len(calls)
    assert "0 finding(s)" in concordance_report(module)
    artifact.compile(tmp_path / operation)
    execution = artifact.prepare_execution({}).run()
    assert execution.buffers[artifact.buffer_order[-1]].item() == 41
    assert "0 finding(s)" in concordance_report(module)


@pytest.mark.parametrize("scope,semantics,reason", [
    (1, 2, "memory scope 1"),
    (2, 2, "memory scope 2"),
    (256, 0x102, "memory semantics mask 0x102"),
    (256, 0x4002, "memory semantics mask 0x4002"),
])
def test_c_refuses_unsupported_scope_or_visibility_without_emitting_a_fence(scope, semantics, reason):
    module = compile_source(
        "from src.compiler.synchronization import memory_barrier\n"
        f"def root():\n    memory_barrier({scope}, {semantics}, 1)\n    return 1\n"
    )
    artifact = emit_ssa_module_to_c(module, "root__root")
    assert not artifact.complete
    assert any(item.operation == "MemoryBarrier" and reason in item.reason for item in artifact.shortfalls)
    assert "atomic_thread_fence(" not in artifact.source
    assert not tuple(module.metadata["identity_book"].page(C_SYNCHRONIZATION_POLICY).rows())
