"""A specialized formal keeps the exact caller class, not its generic annotation."""
import networkx as nx
import pytest
from src.common.tensors.topological_reducer import specialize_python_precision_widths
from src.compiler.identity_concordance import begin_identity_book, end_identity_book, Novel, Derived, Mode
from src.compiler.concordance_declarations import (
    FUNCTION_ADDRESS, SOURCE_VALUE_CLASS, CANONICAL_VALUE, INGESTION, INGEST_SOURCE,
)


def _input(book, authored_class, actual_class, proof=True):
    scope = ("annotation_fixture", "read")
    book.post(FUNCTION_ADDRESS, ("annotation_fixture.read",), 7,
              stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
    lexical_scope = (f"lexical_reads:{scope}", 0)
    formal = book.post(CANONICAL_VALUE, (lexical_scope, 0), ("formal",),
                       stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
    caller = book.post(CANONICAL_VALUE, (("caller_reads", 0), 9), ("actual",),
                       stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
    generic = book.post(SOURCE_VALUE_CLASS, (scope, 0), (authored_class, 1, "parameter annotation value"),
                        stage=INGESTION, provenance=Derived((formal,)), mode=Mode.CONCORD)
    actual = book.post(SOURCE_VALUE_CLASS, (("annotation_fixture", "caller"), 9), (actual_class, 1, "caller value"),
                       stage=INGESTION, provenance=Derived((caller,)), mode=Mode.CONCORD)
    graph = nx.DiGraph()
    graph.add_node(0, type="Input", attributes={"binding_name": "value"}, parents=[], children=[])
    graph.graph.update(authored_function_identity=scope, function_ref=7,
                       canonical_value_ids=True, lexical_read_scope=lexical_scope,
                       planner_parameter_classes={"value": (actual_class, 1)},
                       planner_parameter_class_cells={"value": (actual,)} if proof else {})
    return graph, generic, actual


@pytest.mark.parametrize("authored,actual", [("fixture.View", "fixture.Full"), ("builtins.int", "builtins.bool"), ("builtins.int", "builtins.int")])
def test_specialized_parameter_is_born_from_its_exact_actual_class(authored, actual):
    book, token = begin_identity_book()
    try:
        graph, generic, source = _input(book, authored, actual)
        specialize_python_precision_widths(graph)
        row = (graph.graph["source_numeric_scope"], 0)
        history = book.page(SOURCE_VALUE_CLASS).history(row)
        assert len(history) == 1 and history[0][1][:2] == (actual, 1)
        ref = book.latest_ref(SOURCE_VALUE_CLASS, row)
        causes = {cell for cell, _ in book.edges_into(ref)}
        assert source in causes and generic not in causes
        assert book.page(SOURCE_VALUE_CLASS).latest(generic.row)[0] == authored
        specialize_python_precision_widths(graph)
        assert book.latest_ref(SOURCE_VALUE_CLASS, row) == ref
    finally:
        end_identity_book(token)


def test_absent_exact_caller_proof_does_not_displace_generic_class():
    book, token = begin_identity_book()
    try:
        graph, generic, source = _input(book, "fixture.View", "fixture.Full", proof=False)
        with pytest.raises(ValueError, match="source value class concordance disagreement"):
            specialize_python_precision_widths(graph)
    finally:
        end_identity_book(token)


@pytest.mark.parametrize("backend", ["c", "llvm", "fortran"])
def test_native_parameter_annotation_does_not_replace_actual_nominal_class(backend, tmp_path):
    from pathlib import Path
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import concordance_report
    source = """def classify(value: int):
    return isinstance(value, bool)
def root():
    return classify(True), classify(1)
"""
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="actual_nominal_class",
        extraction_contract=Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml",
        progress=lambda _: None, progress_bars=False,
    )
    assert "0 finding(s)" in concordance_report(module)
    root = exports[0]
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c
        artifact = emit_ssa_to_c(module, root)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path)
        execution = artifact.prepare_execution({})
    elif backend == "llvm":
        from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
        artifact = emit_ssa_function_to_llvm(module, root)
        assert artifact.complete, artifact.shortfalls
        execution = prepare_artifact_execution(compile_artifact(artifact, directory=tmp_path), {})
    else:
        from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative, fortran_compiler
        from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
        if fortran_compiler() is None:
            pytest.skip("Fortran compiler unavailable")
        artifact = emit_module(module, name="actual_nominal_class", outputs={root: outputs[root]}, extra_roots=(root,), progress=lambda _: None)
        assert artifact.complete, artifact.shortfalls
        native = compile_fortran_module_c_shell(artifact, {}, tmp_path, entrypoint=root, library=True)
        execution = FortranCoreNative(native.executable_path, artifact.api.entry_point(root), artifact.api).prepare_execution({})
    execution.run()
    assert tuple(bool(execution.buffers[value.id].item()) for value in outputs[root]) == (True, False)


def _loop_target(book, graph, scope=None, parent=None):
    original_scope = graph.graph["lexical_read_scope"]
    scope = original_scope if scope is None else scope
    if scope != original_scope:
        formal = book.latest_ref(CANONICAL_VALUE, (original_scope, 0))
        book.post(CANONICAL_VALUE, (scope, 0), book.page(CANONICAL_VALUE).cell(formal.row, formal.column),
                  stage=book.registry.declare_stage("read_scope_fork"), provenance=Derived((formal,)), mode=Mode.CONCORD)
    graph.add_node(1, type="Input", value_id=1, attributes={"binding_kind": "loop"}, parents=[], children=[])
    graph.graph.update(canonical_value_ids=True, lexical_read_scope=scope)
    return book.post(CANONICAL_VALUE, (scope, 1), ("target",),
                     stage=INGESTION if parent is None else book.registry.declare_stage("read_scope_fork"),
                     provenance=Novel(INGEST_SOURCE, ()) if parent is None else Derived((parent,)), mode=Mode.CONCORD)


@pytest.mark.parametrize("actual_class", ["builtins.bool", ""])
@pytest.mark.parametrize("has_generic", [False, True])
def test_iterated_class_projection_reads_exact_existing_target_fact(actual_class, has_generic):
    book, token = begin_identity_book()
    try:
        graph, generic, source = _input(book, "builtins.list", "builtins.list")
        authored = graph.graph["authored_function_identity"]
        if has_generic:
            book.post(SOURCE_VALUE_CLASS, (authored, 1), ("builtins.int", 1, "declared sequence element projection"),
                      stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
        target = _loop_target(book, graph)
        actual = book.post(SOURCE_VALUE_CLASS, (("prior_callsite", "read"), 1),
                           (actual_class, 1, "callsite iterable element class"),
                           stage=INGESTION, provenance=Derived((target,)), mode=Mode.CONCORD)
        graph.nodes[1]["attributes"]["callsite_iterable_element_class_cell"] = actual
        specialize_python_precision_widths(graph)
        row = (graph.graph["source_numeric_scope"], 1)
        assert book.page(SOURCE_VALUE_CLASS).latest(row) == book.page(SOURCE_VALUE_CLASS).cell(actual.row, actual.column)
        causes = {ref for ref, _ in book.edges_into(book.latest_ref(SOURCE_VALUE_CLASS, row))}
        assert actual in causes
    finally:
        end_identity_book(token)


def test_mismatched_iterated_target_class_ref_is_refused():
    book, token = begin_identity_book()
    try:
        graph, generic, actual = _input(book, "builtins.list", "builtins.list")
        graph.add_node(1, type="Input", value_id=1, attributes={"callsite_iterable_element_class_cell": actual}, parents=[], children=[])
        with pytest.raises(ValueError, match="does not name its canonical target"):
            specialize_python_precision_widths(graph)
    finally:
        end_identity_book(token)


def test_late_exact_iterated_class_is_part_of_specialization_identity():
    book, token = begin_identity_book()
    try:
        graph, generic, source = _input(book, "builtins.list", "builtins.list")
        authored = graph.graph["authored_function_identity"]
        book.post(SOURCE_VALUE_CLASS, (authored, 1), ("builtins.int", 1, "declared sequence element projection"),
                  stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
        target = _loop_target(book, graph)
        specialize_python_precision_widths(graph)
        generic_scope = graph.graph["source_numeric_scope"]
        actual = book.post(SOURCE_VALUE_CLASS, (("prior_callsite", "read"), 1),
                           ("builtins.bool", 1, "callsite iterable element class"),
                           stage=INGESTION, provenance=Derived((target,)), mode=Mode.CONCORD)
        graph.nodes[1]["attributes"]["callsite_iterable_element_class_cell"] = actual
        specialize_python_precision_widths(graph)
        current_scope = graph.graph["source_numeric_scope"]
        assert current_scope != generic_scope
        assert book.page(SOURCE_VALUE_CLASS).latest((current_scope, 1))[0] == "builtins.bool"
        assert book.page(SOURCE_VALUE_CLASS).latest((generic_scope, 1))[0] == "builtins.int"
    finally:
        end_identity_book(token)


def test_equal_iterated_domains_reuse_scope_across_read_copies():
    book, token = begin_identity_book()
    try:
        graph, generic, source = _input(book, "builtins.list", "builtins.list")
        target = _loop_target(book, graph)
        scopes = []
        for index, domain in enumerate(("builtins.bool", "builtins.int", "", "builtins.bool")):
            current_target = target if index < 3 else _loop_target(book, graph, ("target_copy", 1), parent=target)
            actual = book.post(SOURCE_VALUE_CLASS, (("callsite", index), 1),
                               (domain, 1, "callsite iterable element class"), stage=INGESTION,
                               provenance=Derived((current_target,)), mode=Mode.CONCORD)
            graph.nodes[1]["attributes"]["callsite_iterable_element_class_cell"] = actual
            specialize_python_precision_widths(graph)
            scopes.append(graph.graph["source_numeric_scope"])
        assert len(set(scopes[:3])) == 3 and scopes[0] == scopes[3]
    finally:
        end_identity_book(token)


def test_same_local_id_in_foreign_canonical_target_is_not_authority():
    book, token = begin_identity_book()
    try:
        graph, generic, source = _input(book, "builtins.list", "builtins.list")
        target = _loop_target(book, graph)
        foreign = book.post(CANONICAL_VALUE, (("foreign_target", 0), 1), ("other",),
                            stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
        actual = book.post(SOURCE_VALUE_CLASS, (("foreign_scope", "read"), 1), ("builtins.bool", 1, "callsite iterable element class"),
                           stage=INGESTION, provenance=Derived((foreign,)), mode=Mode.CONCORD)
        graph.nodes[1]["attributes"]["callsite_iterable_element_class_cell"] = actual
        with pytest.raises(ValueError, match="does not derive from its canonical target"):
            specialize_python_precision_widths(graph)
    finally:
        end_identity_book(token)


def test_actual_parameter_class_keeps_its_own_subject_identity():
    from src.compiler.identity_concordance import _source_class_subject_bases
    book, token = begin_identity_book()
    try:
        graph, generic, caller = _input(book, "builtins.int", "builtins.int")
        formal = book.latest_ref(CANONICAL_VALUE, (graph.graph["lexical_read_scope"], 0))
        specialize_python_precision_widths(graph)
        ref = book.latest_ref(SOURCE_VALUE_CLASS, (graph.graph["source_numeric_scope"], 0))
        assert formal in _source_class_subject_bases(book, ref, 0)
        assert formal in {cell for cell, _ in book.edges_into(ref)}
        assert caller in {cell for cell, _ in book.edges_into(ref)}
    finally:
        end_identity_book(token)


@pytest.mark.parametrize("backend", ["c", "llvm", "fortran"])
def test_same_class_native_parameter_keeps_value_abi_subject(backend, tmp_path):
    from pathlib import Path
    import numpy as np
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import concordance_report
    source = """def consume(value: int):
    return value + 1
def root(value: int):
    return consume(value)
"""
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="same_class_subject",
        extraction_contract=Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml",
        progress=lambda _: None, progress_bars=False,
    )
    assert "0 finding(s)" in concordance_report(module)
    root = exports[0]
    argument = module.functions[root].args[0]
    feeds = {argument.id: np.array([0], dtype="int64")}
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c
        artifact = emit_ssa_to_c(module, root)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path)
        execution = artifact.prepare_execution(feeds)
    elif backend == "llvm":
        from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
        artifact = emit_ssa_function_to_llvm(module, root)
        assert artifact.complete, artifact.shortfalls
        execution = prepare_artifact_execution(compile_artifact(artifact, directory=tmp_path), feeds)
    else:
        from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative, fortran_compiler
        from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
        if fortran_compiler() is None:
            pytest.skip("Fortran compiler unavailable")
        artifact = emit_module(module, name="same_class_subject", outputs={root: outputs[root]}, extra_roots=(root,), progress=lambda _: None)
        assert artifact.complete, artifact.shortfalls
        native = compile_fortran_module_c_shell(artifact, {}, tmp_path, entrypoint=root, library=True)
        execution = FortranCoreNative(native.executable_path, artifact.api.entry_point(root), artifact.api).prepare_execution(feeds)
    for value in (-3, 0, 17):
        execution.buffers[argument.id][0] = value
        execution.run()
        assert execution.buffers[outputs[root][0].id].item() == value + 1
