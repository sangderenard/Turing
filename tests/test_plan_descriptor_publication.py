"""Read-only plan freezing must not revalidate all edges for every value."""
import inspect
from pathlib import Path
import sys
import pytest

from src.compiler import identity_concordance as concordance
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler import glsl_deployment_strategy as strategy


def test_frozen_plan_publishes_abi_once_not_once_per_descriptor(monkeypatch):
    original = concordance.publish_program_abi_graph_identities
    calls = {}

    def observed(graph, **kwargs):
        frame = inspect.currentframe().f_back
        while frame is not None:
            if frame.f_code.co_name == "_fold_callsite_structural_values":
                break  # mutable fold, not this read-only planning boundary
            if frame.f_code.co_name == "_build_shell_hierarchy_plan":
                calls[frame] = calls.get(frame, 0) + 1
                break
            frame = frame.f_back
        return original(graph, **kwargs)

    monkeypatch.setattr(concordance, "publish_program_abi_graph_identities", observed)
    source = "def root(x):\n" + "".join(
        f"    y{n} = {'x' if n == 0 else 'y' + str(n - 1)} + {n + 1}.0\n"
        for n in range(48)
    ) + "    return y47\n"
    module, outputs, _ = lower_ast_source_to_ssa(
        source, "root", name="plan_publication_scaling", progress=lambda message: None,
        extraction_contract=Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml",
    )
    assert module.functions and outputs
    assert calls, "the real public lowering must reach plan freezing"
    assert max(calls.values()) == 1, list(calls.values())


def test_mutable_fold_batches_reads_between_actual_operand_writes(monkeypatch):
    original = concordance.publish_program_abi_graph_identities
    operands = strategy._set_operands
    calls, writes, stacks = {}, {}, {}

    def fold_frame():
        frame = inspect.currentframe().f_back
        while frame is not None:
            if frame.f_code.co_name == "_fold_callsite_structural_values":
                return frame
            frame = frame.f_back

    def observed(graph, **kwargs):
        frame = fold_frame()
        if frame is not None:
            calls[frame] = calls.get(frame, 0) + 1
            caller = inspect.currentframe().f_back
            path = []
            while caller is not None and caller is not frame:
                path.append(caller.f_code.co_name)
                caller = caller.f_back
            stacks[tuple(path)] = stacks.get(tuple(path), 0) + 1
        return original(graph, **kwargs)

    def written(*args, **kwargs):
        frame = fold_frame()
        if frame is not None:
            writes[frame] = writes.get(frame, 0) + 1
        return operands(*args, **kwargs)

    monkeypatch.setattr(concordance, "publish_program_abi_graph_identities", observed)
    monkeypatch.setattr(strategy, "_set_operands", written)
    source = "def root(x):\n" + "".join(
        f"    y{n} = {'x' if n == 0 else 'y' + str(n - 1)} + {n + 1}.0\n"
        for n in range(48)
    ) + "    return y47\n"
    lower_ast_source_to_ssa(source, "root", name="fold_publication_scaling",
        progress=lambda message: None,
        extraction_contract=Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml")
    assert calls
    assert all(count <= writes.get(frame, 0) + 4 for frame, count in calls.items()), stacks


def test_settlement_and_carving_validate_once_per_frozen_graph(monkeypatch):
    original = concordance.publish_program_abi_graph_identities
    settlements, carvings, held_indexes = {}, {}, {}

    def observed(graph, **kwargs):
        frame = inspect.currentframe().f_back
        while frame is not None:
            name = frame.f_code.co_name
            if name == "_fold_callsite_structural_values":
                break
            if name == "_propagate_callsite_tensor_specializations":
                settlements[frame] = settlements.get(frame, 0) + 1
                break
            if name == "_dispatch_subgraph":
                index = frame.f_locals["graph_index"]
                held_indexes[id(index)] = index
                carvings[id(index)] = carvings.get(id(index), 0) + 1
                break
            frame = frame.f_back
        return original(graph, **kwargs)

    monkeypatch.setattr(concordance, "publish_program_abi_graph_identities", observed)
    source = "def root(x):\n" + "".join(
        f"    y{n} = {'x' if n == 0 else 'y' + str(n - 1)} + {n + 1}.0\n"
        for n in range(48)
    ) + "    return y47\n"
    lower_ast_source_to_ssa(source, "root", name="settlement_carving_scaling",
        progress=lambda message: None,
        extraction_contract=Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml")
    assert settlements and carvings
    budgets = [(count, len(frame.f_locals["graphs"]) * frame.f_locals["round_index"])
               for frame, count in settlements.items()]
    assert all(count <= budget for count, budget in budgets), (budgets, list(carvings.values()))
    assert max(carvings.values()) == 1, list(carvings.values())


@pytest.mark.parametrize("case", ["view", "toplevel", "energy", "controller",
                                 "controller_untyped", "mapping", "oscillator"])
def test_frozen_reads_equal_original_fresh_publication(case, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import audit_identity_concordance as audit
    original = strategy._tensor_descriptor
    compared = []

    def checked(graph, node, seen=None):
        answer = original(graph, node, seen)
        caller = inspect.currentframe().f_back
        if caller.f_code.co_name == "_operand_descriptor":
            caller = caller.f_back
        if caller.f_code.co_name == "planned_descriptor":
            fresh = original(graph, node)
            assert answer == fresh, (node, answer, fresh)
            compared.append(node)
        return answer

    monkeypatch.setattr(strategy, "_tensor_descriptor", checked)
    audit.CASES[case]()
    assert compared, "audit source must exercise the real frozen plan's reads"
