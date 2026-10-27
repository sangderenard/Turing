"""Call arguments read their record before that call can mutate it."""
from pathlib import Path
import pytest
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

CONTRACT = Path(__file__).resolve().parents[1] / 'extraction_contracts/program_extraction.yaml'
PREFIX = '''class Clock:
    def __init__(self, value: int):
        self.clock = value
def consume(clock: int, book: Clock) -> int:
    book.clock = clock + 1
    return clock
'''


def lower(source):
    return lower_ast_source_to_ssa(
        source, 'root', name='record_call_order', extraction_contract=CONTRACT,
        stop_after_compilation_unit_plan=True, progress=lambda _: None,
        progress_bars=False,
    )


def test_record_argument_read_before_same_call_store_is_accepted():
    lower(PREFIX + '''def root(value: int) -> int:
    projected = Clock(value)
    return consume(clock=projected.clock, book=projected)
''')


@pytest.mark.parametrize('body', [
    'consume(projected.clock, projected)\n    return projected.clock',
    'return consume(projected.clock, projected) + projected.clock',
])
def test_actual_prior_call_keeps_exact_slot_read(body):
    lower(PREFIX + f'''def root(value: int) -> int:
    projected = Clock(value)
    {body}
''')


def test_record_read_in_later_argument_follows_nested_store():
    lower(PREFIX + '''def root(value: int) -> int:
    projected = Clock(value)
    return consume(consume(projected.clock, projected), projected.clock)
''')


def test_clock_list_reference_is_read_before_constructor_escape_in_loop():
    lower('''class Clock:
    def __init__(self):
        self.clock = [0]
class Page:
    def __init__(self, clock: list[int], book: Clock):
        self.clock = clock
        self.book = book
def root(n: int):
    projected = Clock()
    for index in range(n):
        page = Page(clock=projected.clock, book=projected)
    return projected.clock[0]
''')


def test_structural_range_stop_retains_parameter_binding():
    from src.compiler.identity_concordance import CorrelationTable
    module, _, exports = lower_ast_source_to_ssa(
        'def root(count: int):\n    total = 0\n    for i in range(count):\n        total += i\n    return total\n',
        'root', name='structural_range_argument', extraction_contract=CONTRACT,
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    function = module.functions[exports[0]]
    parameters = dict(function.metadata['parameter_names'])
    assert set(parameters) == {'count'}
    formal = next(value for value in function.args if value.id == parameters['count'])
    assert formal.dtype == 'int64'
    assert formal.accounting['native_exact_type'] == 'builtins.int'


@pytest.mark.parametrize('lane', ['C', 'LLVM', 'Fortran'])
@pytest.mark.parametrize('surface', ['method', 'direct', 'loop', 'loop_then_call', 'isolation', 'temporal'])
@pytest.mark.parametrize('imported', [False, True])
def test_authored_add_method_is_not_dead_tensor_metadata(lane, surface, imported, tmp_path, monkeypatch):
    import numpy as np
    from src.compiler.identity_concordance import CorrelationTable
    from src.compiler.ssa_c_backend import emit_ssa_to_c
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
    from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative, fortran_compiler
    from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
    source = '''class Clock:
    def __init__(self, value: int):
        self.clock = value
    def add(self, step: int):
        self.clock = self.clock + step
    def read(self) -> int:
        return self.clock
def root(value: int) -> int:
    projected = Clock(value)
    projected.add(7)
    return projected.read()
'''
    if surface == 'direct':
        source = source.replace('return projected.read()', 'return projected.clock')
    elif surface in {'loop', 'loop_then_call'}:
        source = source.replace('    projected.add(7)\n    return projected.read()',
                                '    for index in range(3):\n        projected.add(7)\n' +
                                ('    projected.add(7)\n' if surface == 'loop_then_call' else '') +
                                '    return projected.clock')
    elif surface == 'isolation':
        source = source.replace('    projected.add(7)\n    return projected.read()',
                                '    other = Clock(value)\n    projected.add(7)\n    other.add(9)\n    return projected.clock * 100 + other.clock')
    elif surface == 'temporal':
        source = source.replace('    projected.add(7)\n    return projected.read()',
                                '    before = projected.clock\n    projected.add(7)\n    return before * 100 + projected.clock')
    contract = CONTRACT
    if imported:
        import hashlib
        from src.compiler.extraction_contract import ExtractionContract
        module_name = 'clock_' + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:8]
        definitions, root_source = source.split('def root(', 1)
        path = tmp_path / (module_name + '.py')
        path.write_text(definitions)
        monkeypatch.syspath_prepend(str(tmp_path))
        source = f'from {module_name} import Clock\n' + 'def root(' + root_source
        contract = ExtractionContract(CONTRACT).with_sources([(module_name, path)])
    field_cells = []
    def inspect(graph):
        for entry in graph.function_table:
            for _, data in entry.graph.G.nodes(data=True):
                attrs = data.get('attributes') or {}
                if attrs.get('attribute') == 'clock' and attrs.get('field_declaration_cell') is not None:
                    field_cells.append(attrs['field_declaration_cell'])
    module, outputs, exports = lower_ast_source_to_ssa(
        source, 'root', name='retained_source_add', extraction_contract=contract,
        resolved_process_graph_sink=inspect, progress=lambda _: None, progress_bars=False,
    )
    assert field_cells and len(set(field_cells)) == 1
    assert CorrelationTable.build(module).findings(module) == []
    assert any(record.callee_name == 'add' for record in module.call_table[exports[0]])
    root = module.functions[exports[0]]
    parameter = dict(root.metadata['parameter_names'])['value']
    result = next(i.args[0].id for block in root.blocks.values() for i in block.instrs if i.op == 'Ret')
    if lane == 'C':
        artifact = emit_ssa_to_c(module, exports[0])
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path)
        prepare = artifact.prepare_execution
    elif lane == 'LLVM':
        artifact = emit_ssa_function_to_llvm(module, exports[0])
        assert artifact.complete, artifact.shortfalls
        built = compile_artifact(artifact, directory=tmp_path)
        prepare = lambda feeds: prepare_artifact_execution(built, feeds)
    else:
        if fortran_compiler() is None:
            pytest.skip('native Fortran compiler unavailable')
        artifact = emit_module(module, outputs={name: outputs[name] for name in exports}, extra_roots=exports)
        assert artifact.complete, artifact.shortfalls
        built = compile_fortran_module_c_shell(artifact, {}, tmp_path, entrypoint=exports[0], library=True)
        prepare = FortranCoreNative(built.executable_path, artifact.api.entry_point(exports[0]), artifact.api).prepare_execution
    for value in ((0, 31, 4000) if surface in {'isolation', 'temporal'} else (0, 31, 2**53 + 1)):
        actual = prepare({parameter: np.array([value], dtype=np.int64)}).run().buffers[result]
        expected = (101 * value + 709 if surface == 'isolation' else
                    101 * value + 7 if surface == 'temporal' else
                    value + (28 if surface == 'loop_then_call' else 21 if surface == 'loop' else 7))
        assert int(np.asarray(actual).reshape(-1)[0]) == expected
