"""A call-only loop consumes its exact declared source-control edge."""
import ast
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from src.compiler.control_source import (
    AuthoredControlSite, ConditionalBlock, ControlProgram, LoopBlock,
    SequenceBlock, StatementBlock, overlay_scheduled_control,
)


@pytest.mark.parametrize('arm', ['body', 'orelse'])
@pytest.mark.parametrize('has_region', [False, True])
def test_nested_regionless_loop_consumes_authored_arm(arm, has_region):
    source = ast.parse('if flag:\n    before=1\n    result=tuple(x for x in value)\n    after=2\nelse:\n    pass\n').body[0]
    if arm == 'orelse':
        source.body, source.orelse = source.orelse, source.body
    statements = getattr(source, arm)
    loop_expression = statements[1].value.args[0].generators[0]
    graph = nx.DiGraph()
    graph.add_node(11, expr_obj=source)
    graph.add_node(13, expr_obj=loop_expression)
    graph.add_node(20, expr_obj=statements[0].value)
    graph.add_node(21, expr_obj=statements[2].value)
    loop = LoopBlock('iteration_13', '0', 'n', '1', SequenceBlock(()), source_loop_node_id=13)
    markers = SequenceBlock((StatementBlock(('__scheduled_region_0__',)),
                             StatementBlock(('__scheduled_region_1__',)))) if has_region else SequenceBlock(())
    parent = ConditionalBlock(10, markers if arm == 'body' else SequenceBlock(()),
                              markers if arm == 'orelse' else None, source_node_id=11)
    order = (0, 1) if has_region else ()
    program = overlay_scheduled_control(order,
        (ControlProgram(loop), ControlProgram(SequenceBlock((parent,)), order)),
        known_nesting={1: (0,)}, source_graph=SimpleNamespace(G=graph),
        region_source_nodes={0: (20,), 1: (21,)})
    assert len(program.root.blocks) == 1
    projected = program.root.blocks[0]
    assert isinstance(projected, ConditionalBlock)
    members = getattr(projected, arm).blocks
    expected = (markers.blocks[0], loop, markers.blocks[1]) if has_region else (loop,)
    assert members == expected


def test_nested_regionless_loop_refuses_missing_source_instead_of_hoisting():
    loop = LoopBlock('iteration', '0', 'n', '1', SequenceBlock(()), source_loop_node_id=13)
    parent = ConditionalBlock(10, SequenceBlock(()), source_node_id=11)
    with pytest.raises(ValueError, match='lacks its source graph'):
        overlay_scheduled_control((), (ControlProgram(loop), ControlProgram(parent)), known_nesting={1: (0,)})


@pytest.mark.parametrize('fault', ['cloned_identity', 'cross_scope_region'])
def test_nested_regionless_loop_refuses_unproven_placement(fault):
    source = ast.parse('if flag:\n    before=1\n    result=tuple(x for x in value)\n    after=2\n').body[0]
    loop_expression = source.body[1].value.args[0].generators[0]
    if fault == 'cloned_identity':
        import copy
        loop_expression = copy.deepcopy(loop_expression)
    graph = nx.DiGraph()
    graph.add_node(11, expr_obj=source)
    graph.add_node(13, expr_obj=loop_expression)
    graph.add_node(20, expr_obj=source.body[0].value)
    graph.add_node(21, expr_obj=source.body[2].value)
    loop = LoopBlock('iteration', '0', 'n', '1', SequenceBlock(()), source_loop_node_id=13)
    parent = ConditionalBlock(10, StatementBlock(('__scheduled_region_0__',)), source_node_id=11)
    message = 'no exact authored arm' if fault == 'cloned_identity' else 'crosses a nested loop slot'
    with pytest.raises(ValueError, match=message):
        overlay_scheduled_control((0,), (ControlProgram(loop), ControlProgram(parent, (0,))),
            known_nesting={1: (0,)}, source_graph=SimpleNamespace(G=graph),
            region_source_nodes={0: (20, 21)})


@pytest.mark.parametrize("slot", ["iter", "ifs", "elt"])
def test_nested_regionless_loop_consumes_exact_comprehension_site(slot):
    parent = LoopBlock(
        "parent",
        "0",
        "n",
        "1",
        SequenceBlock((StatementBlock(("body",)),)),
        source_loop_node_id=11,
    )
    child = LoopBlock(
        "child", "0", "m", "1", SequenceBlock(()), source_loop_node_id=13
    )
    graph = nx.DiGraph()
    graph.add_nodes_from(((11, {"expr_obj": ast.comprehension()}), (13, {})))

    program = overlay_scheduled_control(
        (),
        (ControlProgram(parent), ControlProgram(child)),
        known_nesting={0: (1,)},
        authored_control_sites={
            (0, 1): AuthoredControlSite(11, 13, slot, path=(slot,))
        },
        source_graph=SimpleNamespace(G=graph),
    )

    if slot == "iter":
        assert program.root == SequenceBlock((child, parent))
    else:
        projected = program.root.blocks[0]
        assert isinstance(projected, LoopBlock)
        assert projected.body.blocks == (child, StatementBlock(("body",)))


@pytest.mark.parametrize('arm', ['body', 'orelse'])
def test_conditional_generator_runs_native_loop_only_in_selected_arm(tmp_path, arm):
    from tests.test_source_object_context import context
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import concordance_report
    from src.compiler.shell_external_references import ExternalReferenceValueCodec
    from src.compiler.ssa_object_context import lower_object_context
    from src.compiler.ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution

    repository = Path(__file__).resolve().parents[1]
    policy = ExtractionContract(repository / 'extraction_contracts/program_extraction.yaml').with_program_abi({
        'records': {}, 'bindings': [], 'values': [
            {'function': 'root', 'parameter': 'value', 'storage': 'reference', 'python_type': 'builtins.object'},
            {'function': 'root', 'parameter': 'flag', 'storage': 'scalar', 'dtype': 'float64', 'python_type': 'builtins.float'},
        ],
    }).with_execution_file(repository / 'extraction_contracts/vehicle_full_native_execution.yaml')
    expression = 'tuple(identity(item) for item in value)'
    true_value, false_value = (expression, 'value') if arm == 'body' else ('value', expression)
    source = ('def identity(item):\n    return item\n'
              'def root(value,flag):\n    if flag>0.0:\n'
              f'        result={true_value}\n    else:\n        result={false_value}\n    return result\n')
    scope = context()
    module, _, exports = lower_ast_source_to_ssa(source, 'root', extraction_contract=policy,
        native_object_context=scope, progress=lambda _: None, progress_bars=False)
    report = concordance_report(module)
    assert report.splitlines()[0].endswith('0 finding(s)'), report
    function = module.functions[exports[0]]
    native = compile_artifact(emit_ssa_function_to_llvm(module, function.name), directory=tmp_path / 'source')
    release_module = lower_object_context(scope, prefix='release_owned_loop')
    release = compile_artifact(emit_ssa_function_to_llvm(release_module, 'release_owned_loop_release'), directory=tmp_path / 'release')
    parameters = dict(function.metadata['parameter_names'])
    output = next(i.args[0] for b in function.blocks.values() for i in b.instrs if i.op == 'Ret' and i.args)
    leaf = object()
    for size in (0, 1, 17, 257):
        for flag in (0, 1):
            original = (leaf,) * size
            boundary = ExternalReferenceValueCodec().bind_native_objects(scope, (original,))
            count = int(boundary.buffers[scope.objects.length_address_id][0])
            try:
                invocation = boundary.bind_execution(native, function, {
                    parameters['value']: np.array([boundary.roots[0]], dtype=np.int64),
                    parameters['flag']: np.array([flag], dtype=np.float64),
                })
                invocation.run()
                assert invocation.native_invocations == 1
                result, = boundary.export((int(invocation.buffers[output.id].item()),))
                assert type(result) is tuple and len(result) == size
                assert all(item is leaf for item in result)
                selected = bool(flag) == (arm == 'body')
                assert int(boundary.buffers[scope.objects.length_address_id][0]) == count + int(selected)
                if not selected:
                    assert result is original
            finally:
                prepare_artifact_execution(release, boundary.buffers).run()
                boundary.close()
