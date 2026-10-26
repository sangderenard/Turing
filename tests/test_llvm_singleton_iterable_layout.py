"""An aggregate carrier retains its element layout even at length one."""
from pathlib import Path
import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock, IRModule
import numpy as np


@pytest.mark.parametrize('dtype,payload', [
    ('float64', (1.5,)), ('float64', (1.5, 2.25)),
    ('int64', (9007199254740993,)), ('int64', (9007199254740993, 9007199254740995)),
    ('bool', (True,)), ('bool', (False,)), ('bool', (False, True)),
])
def test_aggregate_carrier_native_element_layout(dtype, payload, tmp_path):
    index = SSAValue(0, 'int64')
    aggregate, address, value = SSAValue(1, 'ssa.aggregate'), SSAValue(2, 'ptr'), SSAValue(3, dtype)
    module = IRModule({'root': Function('root', [index], {
        'entry': BasicBlock('entry', [
            Instr('Const', [], aggregate, attributes={'value': payload, 'binding': 'static_iterable'}),
            Instr('Br', [], None, attributes={'target': 'read'}),
        ], successors=['read']),
        'read': BasicBlock('read', [
        Instr('GetElementPtr', [aggregate, index], address),
        Instr('Load', [address], value),
        Instr('Ret', [value], None),
    ])})})
    artifact = emit_ssa_function_to_llvm(module, 'root')
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / 'native')
    execution = prepare_artifact_execution(native, {0: np.zeros(1, dtype='int64')})
    for offset, expected in enumerate(payload):
        execution.buffers[0][0] = offset
        execution.run()
        assert execution.buffers[value.id].item() == expected


@pytest.mark.parametrize('kind,initial,payload,expected', [
    ('float', '0.0', '[1.5]', 1.5),
    ('float', '0.0', '[1.5, 2.25]', 2.25),
    ('int', '0', '[9007199254740993]', 9007199254740993),
    ('int', '0', '[9007199254740993, 9007199254740995]', 9007199254740995),
    ('bool', 'False', '[True]', True),
    ('bool', 'False', '[False]', False),
    ('bool', 'False', '[False, True]', True),
])
@pytest.mark.parametrize('backend', ['llvm', 'c', 'fortran'])
def test_static_iterable_native_element_layout(kind, initial, payload, expected, backend, tmp_path):
    rejected = {'float': '-1.0', 'int': '-1', 'bool': 'True'}[kind]
    source = f'''def last(values: list[{kind}]) -> {kind}:
    previous = {initial}
    for value in values:
        if not isinstance(value, {kind}):
            return {rejected}
        previous = value
    return previous
def root() -> {kind}:
    return last({payload})
'''
    module, outputs, exports = lower_ast_source_to_ssa(source, 'root', name='typed_static_iterable',
        extraction_contract=Path(__file__).resolve().parents[1] / 'extraction_contracts/program_extraction.yaml',
        progress=lambda _: None, progress_bars=False)
    report = concordance_report(module)
    assert '0 finding(s)' in report, report
    target = next(function for name, function in module.functions.items()
                  if '__last__specialized_' in name and '__planned_region_' not in name)
    iteration_value = next(instruction.res for block in target.blocks.values()
                          for instruction in block.instrs
                          if instruction.op == 'Load' and instruction.attributes.get('binding') == 'static_iterable')
    # ``previous`` and ``value`` share the body's producer, but the returned
    # continuation is its control-owned exit port, not the body Input itself.
    assert target.metadata['source_output_value_ids'] != (iteration_value.id,)
    assert all(instruction.args for block in target.blocks.values()
               for instruction in block.instrs if instruction.op == 'Ret')
    root = exports[0]
    if backend == 'llvm':
        artifact = emit_ssa_function_to_llvm(module, root)
        assert artifact.complete, artifact.shortfalls
        native = compile_artifact(artifact, directory=tmp_path / 'native')
        execution = prepare_artifact_execution(native, {})
    elif backend == 'c':
        from src.compiler.ssa_c_backend import emit_ssa_to_c
        artifact = emit_ssa_to_c(module, root)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / 'native')
        execution = artifact.prepare_execution({})
    else:
        from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative, fortran_compiler
        from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
        if fortran_compiler() is None:
            pytest.skip('Fortran compiler unavailable')
        artifact = emit_module(module, name='typed_static_iterable', outputs={root: outputs[root]}, extra_roots=(root,), progress=lambda _: None)
        assert artifact.complete, artifact.shortfalls
        native = compile_fortran_module_c_shell(artifact, {}, tmp_path / 'native', entrypoint=root, library=True)
        execution = FortranCoreNative(native.executable_path, artifact.api.entry_point(root), artifact.api).prepare_execution({})
    for _ in range(2):
        execution.run()
        assert execution.buffers[outputs[root][0].id].item() == expected


def test_typed_producer_resolves_only_undefined_unknown_loop_reservation():
    import ast
    from src.transmogrifier.graph.graph_express2 import ProcessGraph
    from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
    from src.compiler.control_source import ControlProgram, SequenceBlock
    from src.compiler.precompile_to_ssa import _ControlSSABuilder
    from src.compiler.concordance_declarations import ADDRESS, BindingKind, SSA_VALUE
    from src.compiler.identity_concordance import begin_identity_book, end_identity_book
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(ast.parse('def source(value: int):\n return value\n'))
        reduce_abstract_tensor_topology(graph)
        source = graph.function_table.entry('source').graph
        source_id = next(n for n, data in source.G.nodes(data=True) if data.get('type') == 'Input')
        builder = _ControlSSABuilder(ControlProgram(SequenceBlock(())),
            function_name='reserved_loop_update', first_value_id=100,
            region_callees={}, region_signatures={},
            lexical_read_scope=source.G.graph['lexical_read_scope'])
        reserved = SSAValue(source_id, None)
        builder._bind(source_id, reserved, BindingKind.LOOP_SEED, builder._function_root())
        address = builder.fresh_value(dtype='ptr', transform=ADDRESS,
            operands=(builder._function_root(),))
        produced = builder.produced_value(source_id, dtype='int64', cause=(address,))
        assert produced is reserved and produced.dtype == 'int64'
        cell = book.latest_ref(SSA_VALUE, (builder._scope(), source_id))
        assert book.page(SSA_VALUE).cell(cell.row, cell.column).dtype == 'int64'
        assert builder._value_cell(address) in {source for source, _ in book.edges_into(cell)}
        # A second producer's differing spelling cannot retype a concrete
        # reservation; that disagreement needs its actual conversion record.
        assert builder.produced_value(source_id, dtype='float64', cause=(address,)) is reserved
        assert reserved.dtype == 'int64'
    finally:
        end_identity_book(token)
