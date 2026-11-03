"""Authored exception/control combinations compiled to standalone C shells."""
import ast
import subprocess

import numpy as np
import pytest

from tests.test_source_ssa_synchronization import compile_source
from src.compiler.identity_concordance import concordance_report
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.transmogrifier.dtype_layout import c_lane_numpy_dtype


BASIC = '''def root(n: int) -> int:
    result = 5
    try:
        result = 12 // n
    except ZeroDivisionError:
        result = 77
    finally:
        result = result + 1
    return result
'''
LOOP = '''def root(n: int) -> int:
    total = 0
    for i in range(4):
        value = 0
        try:
            value = 12 // (i - n)
        except ZeroDivisionError:
            value = 30
        else:
            value = value + 2
        finally:
            value = value + 1
        total = total + value
    return total
'''
WORKER = '''import threading
def root(n: int) -> int:
    shared = [0]
    def worker():
        value = 5
        try:
            value = 12 // n
        except ZeroDivisionError:
            value = 77
        finally:
            value = value + 1
        shared[0] = value
    task = threading.Thread(target=worker)
    task.start()
    task.join()
    return shared[0]
'''


def test_protected_bare_return_materializes_the_literal_none():
    from src.compiler.source_exception_ingestion import Context, FailureArms

    lowering = FailureArms(set())
    lowering.return_flag = '__returned'
    lowering.return_value = '__return_value'
    statements = lowering.statement(
        ast.Return(value=None), Context('__error', protected=True),
    )
    assignment = statements[0]
    assert isinstance(assignment, ast.Assign)
    assert isinstance(assignment.value, ast.Constant)
    assert assignment.value.value is None


def run_source(tmp_path, source, n):
    namespace = {}
    exec(source, namespace)
    expected = namespace['root'](n)
    module = compile_source(source)
    assert '0 finding(s)' in concordance_report(module)
    root = module.functions['root__root']
    parameter = dict(root.metadata['parameter_names']).get('n')
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    executable = artifact.compile_standalone(tmp_path / 'program', ({parameter: np.array([n], dtype=np.int64)} if parameter is not None else {}))
    process = subprocess.run([str(executable.executable_path), '1', '0'], cwd=executable.directory,
                             capture_output=True, text=True, timeout=10)
    assert process.returncode == 0, process.stderr
    raw = executable.final_outputs_path.read_bytes()
    offset, outputs = 0, []
    for dtype, shape, role in zip(artifact.buffer_dtypes, artifact.buffer_shapes, artifact.buffer_roles, strict=True):
        count = int(np.prod(shape)) if shape else 1
        values = np.frombuffer(raw, dtype=c_lane_numpy_dtype(dtype), count=count, offset=offset)
        offset += values.nbytes
        if role == 'output':
            outputs.extend(values.tolist())
    assert outputs == [expected]
    assert '0 finding(s)' in concordance_report(module)


@pytest.mark.parametrize('n', [0, 3, -3])
def test_try_except_finally_standalone(tmp_path, n):
    run_source(tmp_path, BASIC, n)


@pytest.mark.parametrize('n', [0, 2, 5])
def test_try_except_else_finally_in_loop_standalone(tmp_path, n):
    run_source(tmp_path, LOOP, n)


@pytest.mark.parametrize('n', [0, 3])
def test_worker_catches_its_own_error_standalone(tmp_path, n):
    run_source(tmp_path, WORKER, n)


def test_loop_and_try_in_an_authored_worker(tmp_path):
    body = LOOP.split('    total = 0\n', 1)[1].rsplit('    return total\n', 1)[0]
    source = ('import threading\ndef root(n: int) -> int:\n    shared = [0]\n'
              '    def worker():\n        total = 0\n'
              + ''.join('    ' + line + '\n' for line in body.splitlines())
              + '        shared[0] = total\n    task = threading.Thread(target=worker)\n'
                '    task.start()\n    task.join()\n    return shared[0]\n')
    run_source(tmp_path, source, 2)


def test_nested_raise_rethrow_and_finally(tmp_path):
    source = '''def root(n: int) -> int:
    value = 0
    try:
        try:
            raise ValueError("authored failure")
        except ValueError:
            value = 3
            raise
        finally:
            value = value + 2
    except ValueError:
        value = value * 10
    return value
'''
    run_source(tmp_path, source, 0)


def test_raise_from_none_is_concorded_and_native(tmp_path):
    source = '''def root(n: int) -> int:
    try:
        if n < 0:
            raise TypeError("negative")
    except TypeError:
        raise KeyError("hidden context") from None
    return n + 1
'''
    module = compile_source(source)
    from src.compiler.concordance_declarations import SOURCE_EXCEPTION_OPERATION
    book = module.metadata['identity_book']
    facts = [book.page(SOURCE_EXCEPTION_OPERATION).latest(row)
             for row in book.page(SOURCE_EXCEPTION_OPERATION).rows()]
    key_error = [fact for fact in facts
                 if fact[0] == 'exception_make' and fact[1] == 'builtins.KeyError']
    assert len(key_error) == 1
    assert key_error[0][4:7] == (None, None, True)
    run_source(tmp_path, source, 1)


def test_explicit_exception_cause_is_an_exact_graph_operand():
    source = '''def root(n: int) -> int:
    try:
        raise ValueError("first")
    except ValueError as error:
        raise RuntimeError("second") from error
'''
    module = compile_source(source)
    from src.compiler.concordance_declarations import SOURCE_EXCEPTION_OPERATION
    from src.compiler.identity_concordance import Ref
    book = module.metadata['identity_book']
    facts = [(row, book.page(SOURCE_EXCEPTION_OPERATION).latest(row))
             for row in book.page(SOURCE_EXCEPTION_OPERATION).rows()]
    runtime_error = [(row, fact) for row, fact in facts
                     if fact[0] == 'exception_make' and fact[1] == 'builtins.RuntimeError']
    assert len(runtime_error) == 1
    row, fact = runtime_error[0]
    cause, cause_argument, suppress_context = fact[4:7]
    assert isinstance(cause, Ref)
    assert cause_argument == 0
    assert suppress_context is True
    assert cause in {source for source, _stage in book.edges_into(
        book.latest_ref(SOURCE_EXCEPTION_OPERATION, row)
    )}
    assert '0 finding(s)' in concordance_report(module)


@pytest.mark.parametrize('enabled', [True, False])
def test_division_safety_option_is_an_ssa_operator_contract(enabled):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.transmogrifier.ssa import operator_definitions
    source = 'def root(n: float) -> float:\n    return 12.0 / n\n'
    module, _, _ = lower_ast_source_to_ssa(source, 'root',
        extraction_contract='extraction_contracts/program_extraction.yaml',
        operator_safety_arms=enabled, progress=lambda _: None, progress_bars=False)
    divisions = [i for fn in module.functions.values() for b in fn.blocks.values()
                 for i in b.instrs if i.op == 'Div']
    assert divisions
    assert operator_definitions['Div'].failure_arms
    for instruction in divisions:
        assert bool(instruction.failure_arms_for(module.metadata["identity_book"])) is enabled
        assert instruction.attributes['operator_safety_arms'] is enabled
    artifact = emit_ssa_module_to_c(module, 'root__root')
    assert artifact.complete, artifact.shortfalls
    assert '0 finding(s)' in concordance_report(module)


@pytest.mark.parametrize('operator', ['/', '//', '%'])
@pytest.mark.parametrize('n', [0, -5])
def test_floating_division_family_owns_its_failure_arm(tmp_path, operator, n):
    source = f'''def root(n: int) -> float:
    value = 0.0
    try:
        value = 12.0 {operator} n
    except ZeroDivisionError:
        value = 23.0
    return value
'''
    run_source(tmp_path, source, n)


@pytest.mark.parametrize('n', [0, 3])
def test_return_in_finally_suppresses_pending_error(tmp_path, n):
    source = '''def root(n: int) -> int:
    try:
        value = 12 // n
    finally:
        return 9
'''
    run_source(tmp_path, source, n)


@pytest.mark.parametrize('n', [0, 3])
def test_returns_from_try_and_handler_run_finally(tmp_path, n):
    source = '''def root(n: int) -> int:
    value = [0]
    try:
        return 12 // n
    except ZeroDivisionError:
        return 77
    finally:
        value[0] = 5
'''
    run_source(tmp_path, source, n)


def test_unhandled_raise_uses_ssa_fault_exit_and_shell_channel(tmp_path):
    source = '''def root() -> int:
    try:
        raise ValueError("authored failure")
    finally:
        value = 7
    return 0
'''
    namespace = {}
    exec(source, namespace)
    with pytest.raises(ValueError, match='authored failure'):
        namespace['root']()
    module = compile_source(source)
    function = module.functions['root__root']
    raises = [(block, index) for block in function.blocks.values()
              for index, instruction in enumerate(block.instrs) if instruction.op == 'ExceptionRaise']
    assert raises
    for block, index in raises:
        branch = block.instrs[index + 1]
        assert branch.op == 'Br'
        fault_exit = function.blocks[branch.attributes['target']]
        assert fault_exit.instrs[-1].op == 'Trap'
        assert fault_exit.instrs[-1].attributes['fault_exit']
    artifact = emit_ssa_module_to_c(module, function.name)
    assert artifact.complete, artifact.shortfalls
    executable = artifact.compile_standalone(tmp_path / 'program', {})
    process = subprocess.run([str(executable.executable_path), '1', '0'], cwd=executable.directory,
                             capture_output=True, text=True, timeout=10)
    assert process.returncode == 7
    assert 'builtins.ValueError' in process.stderr
    assert 'authored failure' in process.stderr
    assert not executable.final_outputs_path.exists()
    assert '0 finding(s)' in concordance_report(module)


def test_protected_ordinary_call_requires_a_failure_return_contract():
    source = '''def helper(n: int) -> int:
    return 12 // n
def root(n: int) -> int:
    value = 0
    try:
        value = helper(n)
    except ZeroDivisionError:
        value = 77
    return value
'''
    with pytest.raises(ValueError, match='ordinary call requires an explicit failure-return contract'):
        compile_source(source)


@pytest.mark.parametrize('n', [-2, 0, 4])
def test_protected_loop_has_its_cleanup_continuation(tmp_path, n):
    source = '''def root(n: int) -> int:
    total = 0
    try:
        for i in range(n):
            total = total + i
    except ValueError:
        total = 77
    return total
'''
    run_source(tmp_path, source, n)


def test_exception_handler_does_not_catch_base_exception_itself(tmp_path):
    source = '''def root(n: int) -> int:
    value = 0
    try:
        raise BaseException("base")
    except Exception:
        value = 7
    except BaseException:
        value = 23
    return value
'''
    run_source(tmp_path, source, 0)


@pytest.mark.parametrize('n', [0, 3])
def test_worker_return_runs_observable_finally_store(tmp_path, n):
    source = '''import threading
def root(n: int) -> int:
    shared = [0, 0]
    def worker() -> int:
        try:
            shared[0] = 12 // n
            return 11
        except ZeroDivisionError:
            shared[0] = 77
            return 22
        finally:
            shared[1] = 5
    task = threading.Thread(target=worker)
    task.start()
    task.join()
    return shared[0] * 10 + shared[1]
'''
    run_source(tmp_path, source, n)


@pytest.mark.parametrize('n', [0, 2])
@pytest.mark.parametrize('worker', [False, True])
def test_while_loop_handlers_and_worker_contexts(tmp_path, n, worker):
    source = LOOP.replace('    for i in range(4):', '    i = 0\n    while i < 4:')
    source = source.replace('    return total\n', '        i = i + 1\n    return total\n')
    if worker:
        body = source.split('def root(n: int) -> int:\n', 1)[1].rsplit('    return total\n', 1)[0]
        source = ('import threading\ndef root(n: int) -> int:\n    shared = [0]\n    def worker():\n'
                  + ''.join('    ' + line + '\n' for line in body.splitlines())
                  + '        shared[0] = total\n    task = threading.Thread(target=worker)\n'
                    '    task.start()\n    task.join()\n    return shared[0]\n')
    run_source(tmp_path, source, n)


PROTECTED_LOOP = '''def root(n: int) -> int:
    total = 0
    try:
        for i in range(4):
            total = total + 12 // (i - n)
        else:
            total = total + 7
    except ZeroDivisionError:
        total = total + 50
    finally:
        total = total + 100
    return total
'''


@pytest.mark.parametrize('n', [0, 2, 5])
def test_loop_inside_try_exits_to_handler_and_finally(tmp_path, n):
    run_source(tmp_path, PROTECTED_LOOP, n)


def test_worker_has_loop_inside_try(tmp_path):
    body = PROTECTED_LOOP.split('def root(n: int) -> int:\n', 1)[1].rsplit('    return total\n', 1)[0]
    source = ('import threading\ndef root(n: int) -> int:\n    shared = [0]\n    def worker():\n'
              + ''.join('    ' + line + '\n' for line in body.splitlines())
              + '        shared[0] = total\n    task = threading.Thread(target=worker)\n'
                '    task.start()\n    task.join()\n    return shared[0]\n')
    run_source(tmp_path, source, 2)


@pytest.mark.parametrize('n', [-2, 0])
def test_zero_iteration_loop_inside_try_runs_else_and_finally(tmp_path, n):
    run_source(tmp_path, PROTECTED_LOOP.replace('range(4)', 'range(n)'), n)


@pytest.mark.parametrize('n', [-1, 2])
def test_standalone_raise_without_try_has_native_fault_edges(tmp_path, n):
    source = '''def root(n: int) -> int:
    if n < 0:
        raise ValueError("nonnegative input required")
    return n + 3
'''
    module = compile_source(source)
    function = module.functions['root__root']
    raises = [
        (block, index)
        for block in function.blocks.values()
        for index, instruction in enumerate(block.instrs)
        if instruction.op == 'ExceptionRaise'
    ]
    assert raises
    for block, index in raises:
        branch = block.instrs[index + 1]
        assert branch.op == 'Br'
        assert any(
            instruction.op == 'Trap' and instruction.attributes.get('fault_exit')
            for instruction in function.blocks[branch.attributes['target']].instrs
        )
    artifact = emit_ssa_module_to_c(module, function.name)
    assert artifact.complete, artifact.shortfalls
    parameter = dict(function.metadata['parameter_names'])['n']
    executable = artifact.compile_standalone(
        tmp_path / 'standalone_raise',
        {parameter: np.array([n], dtype=np.int64)},
    )
    completed = subprocess.run(
        [str(executable.executable_path), '1', '0'], cwd=executable.directory,
        capture_output=True, text=True, timeout=10,
    )
    if n < 0:
        assert completed.returncode == 7, completed.stderr
        assert 'builtins.ValueError' in completed.stderr
        assert 'nonnegative input required' in completed.stderr
        assert not executable.final_outputs_path.exists()
    else:
        assert completed.returncode == 0, completed.stderr
        raw = executable.final_outputs_path.read_bytes()
        offset, outputs = 0, []
        for dtype, shape, role in zip(
            artifact.buffer_dtypes, artifact.buffer_shapes,
            artifact.buffer_roles, strict=True,
        ):
            count = int(np.prod(shape)) if shape else 1
            values = np.frombuffer(
                raw, dtype=c_lane_numpy_dtype(dtype), count=count, offset=offset,
            )
            offset += values.nbytes
            if role == 'output':
                outputs.extend(values.tolist())
        assert outputs == [n + 3]
    assert '0 finding(s)' in concordance_report(module)


def test_standalone_raise_retains_dynamic_message_as_graph_operand():
    source = '''def root(n: int):
    raise ValueError(f"bad input {n}")
'''
    module = compile_source(source)
    instructions = [
        instruction
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == 'ExceptionMake'
    ]
    assert len(instructions) == 1 and len(instructions[0].args) == 1
    assert instructions[0].attributes.get('exception_operation_cell')
    artifact = emit_ssa_module_to_c(module, 'root__root')
    assert not artifact.complete
    assert any('owned native message ABI' in item.reason for item in artifact.shortfalls)
    assert '0 finding(s)' in concordance_report(module)
