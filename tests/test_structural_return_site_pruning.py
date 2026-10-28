import ast

import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import Unresolved, current_identity_book
from src.compiler.identity_concordance import concordance_report
from src.compiler.concordance_declarations import (
    RETURN_SITE_SLOT,
    RETURN_SITE_UNREACHABLE,
)
import src.compiler.glsl_deployment_strategy as deployment
from src.compiler.loop_composer import LoopComposer


SOURCE = '''
DOMAIN_TEXT = 1
DOMAIN_BYTES = 2
STATUS_INVALID_DOMAIN = 3
STATUS_INVALID_OFFSETS = 4
STATUS_LIMIT_EXCEEDED = 5
STATUS_OK = 0

class Envelope:
    def __init__(self, limit: int):
        self.max_batch_items = limit

    def _valid_domain(self, domain: int) -> bool:
        return domain == DOMAIN_TEXT or domain == DOMAIN_BYTES

    def validate_batch_envelope(
        self, domain: int, values: list[int], offsets: list[int]
    ) -> tuple[int, int]:
        if not self._valid_domain(domain):
            return STATUS_INVALID_DOMAIN, 0
        if len(offsets) == 0 or offsets[0] != 0:
            return STATUS_INVALID_OFFSETS, 0
        item_count = len(offsets) - 1
        if item_count > self.max_batch_items:
            return STATUS_LIMIT_EXCEEDED, 0
        previous = 0
        for offset in offsets:
            if not isinstance(offset, int) or isinstance(offset, bool):
                return STATUS_INVALID_OFFSETS, 0
            if offset < previous or offset > len(values):
                return STATUS_INVALID_OFFSETS, 0
            previous = offset
        if offsets[-1] != len(values):
            return STATUS_INVALID_OFFSETS, 0
        return STATUS_OK, item_count

def root(domain: int, values: list[int], offsets: list[int], limit: int):
    envelope = Envelope(limit)
    return envelope.validate_batch_envelope(domain, values, offsets)
'''


def test_typed_span_fold_retires_rejected_loop_return_slot(monkeypatch):
    tree = ast.parse(SOURCE)
    envelope = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Envelope"
    )
    validator = next(
        node for node in envelope.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_batch_envelope"
    )
    type_check = next(
        node for node in ast.walk(validator)
        if isinstance(node, ast.If)
        and "isinstance" in ast.dump(node.test)
        and any(
            isinstance(child, ast.Name) and child.id == "bool"
            for child in ast.walk(node.test)
        )
    )
    rejected_return = next(
        node for node in ast.walk(type_check)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple)
    )
    rejected_span = (
        rejected_return.value.lineno,
        rejected_return.value.col_offset,
        rejected_return.value.end_lineno,
        rejected_return.value.end_col_offset,
    )
    abi = {
        "records": {},
        "bindings": [],
        "values": [
            {
                "function": "root", "parameter": "domain",
                "storage": "scalar", "dtype": "int64", "rank": 0,
                "python_type": "builtins.int",
            },
            {
                "function": "root", "parameter": "values",
                "storage": "span", "dtype": "int64", "rank": 1,
                "python_type": "builtins.list",
            },
            {
                "function": "root", "parameter": "offsets",
                "storage": "span", "dtype": "int64", "rank": 1,
                "python_type": "builtins.list",
            },
            {
                "function": "root", "parameter": "limit",
                "storage": "scalar", "dtype": "int64", "rank": 0,
                "python_type": "builtins.int",
            },
        ],
    }
    policy = ExtractionContract(
        "extraction_contracts/program_extraction.yaml"
    ).with_program_abi(abi)

    original = deployment._fold_callsite_structural_values
    original_describe = LoopComposer.describe
    captured = {}
    loop_descriptors = []

    def observe_fold(graph, *args, **kwargs):
        before = dict(graph.G.graph.get("return_slot_values") or {})
        ingestion_scope = graph.G.graph.get("ingestion_value_scope")
        site_cell = next((
            cell for cell, positions in (
                graph.G.graph.get("return_site_spans") or {}
            ).items()
            if tuple(positions) == rejected_span
            and cell.page.name == "ingestion_value"
            and tuple(cell.row)[0] == ingestion_scope
        ), None)
        result = original(graph, *args, **kwargs)
        if graph.G.graph.get("function_name") != "validate_batch_envelope":
            return result
        # The annotation-only function-template fold is intentionally generic:
        # ``list[int]`` does not exclude bool. The explicit root int64 span
        # becomes exact only on the callsite copy, where its loop target has
        # the derived physical-class receipt.
        if not any(
            (data.get("attributes") or {}).get(
                "callsite_iterable_element_class_fact"
            ) == (
                "builtins.int", 1, "callsite iterable element class",
            )
            for _node_id, data in graph.G.nodes(data=True)
        ):
            return result
        if rejected_span not in before or before[rejected_span] == (None, None):
            return result

        after = dict(graph.G.graph.get("return_slot_values") or {})
        assert site_cell is not None
        # ``return_slot_values`` is a compatibility projection. The
        # authoritative RETURN_SITE_SLOT page is what the fold revises.
        assert not any(
            tuple(
                (data.get("source_span") or {}).get(key, -1)
                for key in ("line", "column", "end_line", "end_column")
            ) == rejected_span
            for _node_id, data in graph.G.nodes(data=True)
        )

        book = current_identity_book()
        slot_page = book.pages[RETURN_SITE_SLOT.name]
        scope = tuple(graph.G.graph["lexical_read_scope"])
        assert tuple(graph.G.graph["return_site_spans"][site_cell]) == rejected_span
        rows = tuple(
            row for row in slot_page.scope_rows(scope)
            if row[1] == site_cell
        )
        assert len(rows) == 2
        for row in rows:
            fact = slot_page.latest(row)
            assert isinstance(fact, Unresolved)
            assert fact.reason == RETURN_SITE_UNREACHABLE

        captured["site_cell"] = site_cell
        captured["before"] = before
        captured["after"] = after
        return result

    def observe_loop_describe(composer, graph, node_id):
        descriptor = original_describe(composer, graph, node_id)
        if graph.G.graph.get("function_name") == "validate_batch_envelope":
            site_cell = captured.get("site_cell")
            if site_cell is not None and descriptor.source_type == "For":
                assert all(
                    control[3][0] != site_cell
                    for control in descriptor.return_controls
                )
                assert any(
                    control[3][0] != site_cell
                    for control in descriptor.return_controls
                )
                loop_descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(
        deployment, "_fold_callsite_structural_values", observe_fold
    )
    monkeypatch.setattr(LoopComposer, "describe", observe_loop_describe)
    module, _outputs, _exports = lower_ast_source_to_ssa(
        SOURCE, "root", name="structural_return_site_probe",
        extraction_contract=policy, progress=lambda _: None,
        progress_bars=False,
    )
    assert loop_descriptors
    report = concordance_report(module)
    assert "0 finding(s)" in report, report

    lowered = next(
        function for name, function in module.functions.items()
        if "validate_batch_envelope__specialized" in name
        and "__planned_region_" not in name
    )
    assert all(argument.dtype not in (None, "", "unknown")
               for argument in lowered.args)


def test_mixed_loop_element_classes_refuse_untracked_native_abi():
    source = '''
def check(offsets: list[int]) -> tuple[int, int]:
    for offset in offsets:
        if not isinstance(offset, int) or isinstance(offset, bool):
            return 4, 0
    return 0, len(offsets) - 1

def root():
    integer_result = check([0, 1])
    mixed_result = check([0, True])
    return (
        integer_result[0], integer_result[1],
        mixed_result[0], mixed_result[1],
    )
'''
    contract = ExtractionContract(
        "extraction_contracts/program_extraction.yaml"
    )
    with pytest.raises(
        ValueError,
        match=(
            r"unsupported mixed-element native type ABI: .*"
            r"exact callsite element classes \('builtins.bool', "
            r"'builtins.int'\).*VALUE_TYPE_ABI declares "
            r"\('unknown',\).*no per-element class tag"
        ),
    ):
        lower_ast_source_to_ssa(
            source,
            "root",
            name="mixed_element_type_abi_probe",
            extraction_contract=contract,
            progress=lambda _: None,
            progress_bars=False,
        )


@pytest.mark.parametrize("lane", ("C", "LLVM", "Fortran"))
def test_homogeneous_callsite_element_class_reaches_native_span_abi(
    tmp_path, lane,
):

    source = '''
def check(offsets: list[int]) -> tuple[int, int]:
    for offset in offsets:
        if not isinstance(offset, int) or isinstance(offset, bool):
            return 4, 0
    return 0, len(offsets) - 1

def root():
    integers = check([0, 1])
    booleans = check([False, True])
    return (
        integers[0], integers[1],
        booleans[0], booleans[1],
    )
'''
    module, outputs, exports = lower_ast_source_to_ssa(
        source,
        "root",
        name="homogeneous_callsite_span_abi_probe",
        extraction_contract=ExtractionContract(
            "extraction_contracts/program_extraction.yaml"
        ),
        progress=lambda _: None,
        progress_bars=False,
    )
    assert "0 finding(s)" in concordance_report(module)

    entry = exports[0]
    if lane == "C":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, entry)
        assert artifact.complete, artifact.shortfalls
        native = artifact.compile(tmp_path / "c")
        execution = native.prepare_execution({}).run()
        output_ids = artifact.buffer_order
    elif lane == "LLVM":
        from src.compiler.ssa_llvm_backend import (
            compile_artifact,
            emit_ssa_function_to_llvm,
            prepare_artifact_execution,
        )

        artifact = emit_ssa_function_to_llvm(module, entry)
        assert artifact.complete, artifact.shortfalls
        native = compile_artifact(artifact, directory=tmp_path / "llvm")
        execution = prepare_artifact_execution(native, {}).run()
        output_ids = artifact.buffer_order
    else:
        from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
        from src.compiler.ssa_fortran_backend import (
            FortranCoreNative, emit_module, fortran_compiler,
        )

        if fortran_compiler() is None:
            pytest.skip("native Fortran compiler unavailable")
        artifact = emit_module(
            module,
            name="homogeneous_callsite_span_abi_probe",
            outputs={entry: outputs[entry]},
            extra_roots=(entry,),
            progress=lambda _: None,
        )
        assert artifact.complete, artifact.shortfalls
        built = compile_fortran_module_c_shell(
            artifact, {}, tmp_path / "fortran", entrypoint=entry,
            library=True,
        )
        core = FortranCoreNative(
            built.executable_path, artifact.api.entry_point(entry),
            artifact.api,
        )
        execution = core.prepare_execution({}).run()
        output_ids = tuple(
            int(getattr(value, "id", value)) for value in outputs[entry]
        )
    assert tuple(
        int(execution.buffers[value_id].item())
        for value_id in output_ids
    ) == (0, 1, 4, 0)
