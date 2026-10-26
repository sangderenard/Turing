"""Golden cases and timing use the existing native project-product contract."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
from src.compiler.project_compilation_product import (
    open_project_compilation_product, verify_project_product_cases,
)
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/compiler/compiler_benchmark_workload.py'
CASES = ROOT / 'tools/compiler_probes/native_library_cases.json'


def matrix_golden(seed, rounds):
    modulus = 65521
    transform = ((25173, 0, 13849), (25173, 1, 13849), (0, 0, 1))
    power = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    def multiply(a, b):
        return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) % modulus
                           for j in range(3)) for i in range(3))
    while rounds:
        if rounds & 1:
            power = multiply(power, transform)
        transform = multiply(transform, transform)
        rounds >>= 1
    vector = (seed % modulus, 0, 1)
    state, checksum, _ = (sum(row[j] * vector[j] for j in range(3)) % modulus
                          for row in power)
    return checksum * modulus + state


def test_matrix_golden_is_independent_and_matches_small_work():
    from src.compiler.compiler_benchmark_workload import recurrence
    for rounds in (0, 1, 7, 31):
        assert matrix_golden(12345, rounds) == recurrence(12345, rounds)
    case = json.loads(CASES.read_text())['recurrence']
    assert case['expected_probe_results'] == [matrix_golden(*probe) for probe in case['probes']]
    assert case['expected_probe_results'][-1] == 2135221737


@pytest.fixture(scope='module')
def native_product(tmp_path_factory):
    from src.compiler.ssa_fortran_backend import fortran_compiler
    if fortran_compiler() is None:
        pytest.skip('native Fortran compiler unavailable')
    directory = tmp_path_factory.mktemp('native_case_product')
    command = [sys.executable, '-u', '-m', 'tools.compile_project_catalogue',
               '--source', str(SOURCE), '--entry', 'recurrence', '--output', str(directory),
               '--jobs', '1', '--emit-native', '--verification-cases', str(CASES)]
    import os
    environment = os.environ.copy()
    environment['TURING_COMPILER_BOOTSTRAP_REGISTRY'] = str(directory / 'isolated-registry.json')
    environment.pop('TURING_COMPILER_BOOTSTRAP_PRODUCTS', None)
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    print(completed.stdout.splitlines()[-1])
    return open_project_compilation_product(directory)


def test_real_cli_native_library_has_golden_timing_and_artifacts(native_product):
    report = json.loads((native_product.root / 'native-case-verification.json').read_text())['units'][0]
    assert report['lowered'] and report['native_built'] and report['verified']
    assert report['native_probe_count'] == 3 and report['fallback_probe_count'] == 0
    timing = report['timing']
    assert timing['scope'] == 'direct-native-library'
    assert timing['warmups'] == 1 and timing['repetitions'] == 5
    assert timing['native_call_count'] == 18 and timing['fallback_call_count'] == 0
    assert timing['native_warm_median_seconds'] > 0
    assert timing['authored_warm_median_seconds'] > 0
    assert report['product_build_seconds'] >= report['native_build_seconds'] > 0
    assert report['ssa_compile_seconds'] > 0
    assert Path(report['library']).is_file() and Path(report['api']).is_file()
    for field in ('source_sha256', 'api_sha256', 'library_sha256'):
        assert len(report[field]) == 64


def test_supplied_golden_does_not_execute_authored_callable(native_product, monkeypatch):
    import src.compiler.compiler_benchmark_workload as workload
    def forbidden(seed, rounds):
        raise AssertionError('authored callable cannot be the golden oracle')
    monkeypatch.setattr(workload, 'recurrence', forbidden)
    case = {'recurrence': {'probes': [[12345, 7]], 'expected_probe_results': [matrix_golden(12345, 7)]}}
    assert verify_project_product_cases(native_product, case)[0]['verified']


@pytest.mark.parametrize('cases', [{}, [], None])
def test_empty_or_nonmapping_cases_refuse(native_product, cases):
    with pytest.raises(ValueError, match='nonempty mapping'):
        verify_project_product_cases(native_product, cases)


def test_bad_golden_and_invalid_timing_never_leave_verified_receipt(native_product):
    for specification in (
        {'probes': [[12345, 7]], 'expected_probe_results': [-1]},
        {'probes': [[12345, 7]], 'expected_probe_results': [matrix_golden(12345, 7)], 'repetitions': -1},
    ):
        result = verify_project_product_cases(native_product, {'recurrence': specification})[0]
        assert not result['verified']
        receipt = Path(result['library']).parent / 'native-verification.json'
        assert json.loads(receipt.read_text())['status'] == 'failed'
    good = {'recurrence': {'probes': [[12345, 7]], 'expected_probe_results': [matrix_golden(12345, 7)]}}
    assert verify_project_product_cases(native_product, good)[0]['verified']


def test_golden_adapter_rejects_missing_and_stale_replay(native_product):
    from src.compiler.compiler_bootstrap_runtime import _activate_descriptor_call
    link = native_product.links['recurrence']
    path = (native_product.root / link['native_library']).parent / 'native-cases.json'
    original = path.read_bytes()
    try:
        path.unlink()
        with pytest.raises(ValueError, match='missing or changed'):
            _activate_descriptor_call(native_product, 'recurrence')
        path.write_bytes(original + b' ')
        with pytest.raises(ValueError, match='missing or changed'):
            _activate_descriptor_call(native_product, 'recurrence')
    finally:
        path.write_bytes(original)
    _, deployed = _activate_descriptor_call(native_product, 'recurrence')
    assert deployed(12345, 7) == matrix_golden(12345, 7)
    assert deployed.__turing_native_route_counts__['fallback'] == 0


def test_native_case_ctype_catalogue_covers_all_exact_scalar_layouts():
    import ctypes
    from src.compiler.project_compilation_product import _native_scalar_ctypes, _native_scalar_probe_domains
    from src.transmogrifier.dtype_layout import layout_for
    factories, domains = _native_scalar_ctypes(), _native_scalar_probe_domains()
    names = ('bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float32', 'float64')
    for name in names:
        layout = layout_for(name)
        assert factories[layout.ctypes_name] is getattr(ctypes, layout.ctypes_name)
        assert ctypes.sizeof(factories[layout.ctypes_name]) == layout.byte_size
        assert all(factories[layout.ctypes_name](value).value == value for value in domains[layout.ctypes_name])
    assert 'c_not_a_declared_scalar' not in factories


def test_unresolved_unsigned_annotation_cannot_verify_lossy_native_boundary(tmp_path, monkeypatch):
    import os
    from src.compiler.ssa_fortran_backend import fortran_compiler
    from src.compiler.project_compilation_product import compile_project_product
    if fortran_compiler() is None:
        pytest.skip('native Fortran compiler unavailable')
    monkeypatch.setenv('TURING_COMPILER_BOOTSTRAP_REGISTRY', str(tmp_path / 'registry.json'))
    monkeypatch.delenv('TURING_COMPILER_BOOTSTRAP_PRODUCTS', raising=False)
    cases = {
        'uint32_equal': {'probes': [[2**32 - 1, 2**32 - 1], [2**32 - 1, 2**32 - 2]], 'expected_probe_results': [True, False]},
        'uint64_equal': {'probes': [[2**53 + 1, 2**53 + 1], [2**64 - 1, 2**64 - 2]], 'expected_probe_results': [True, False]},
    }
    manifest = compile_project_product(
        ROOT / 'src/compiler/compiler_integer_benchmark_workload.py', tmp_path / 'product',
        entries=tuple(cases), expand_entry_dependencies=False, jobs=1, emit_native=True,
        verification_cases=cases,
    )
    # The current source annotation contract does not declare uint32/uint64.
    # Refuse the lossy float64 boundary rather than verify authored fallback.
    # This test makes no claim of an unsigned native source ABI.
    reports = {item['qualified_name']: item for item in manifest['native_case_verification']}
    assert not reports['uint64_equal']['verified']
    assert 'forbids authored fallback' in reports['uint64_equal']['reason']
    receipt = Path(reports['uint64_equal']['library']).parent / 'native-verification.json'
    assert json.loads(receipt.read_text())['status'] == 'failed'


def test_static_ssa_library_reuse_in_fresh_native_consumer_worker(tmp_path, monkeypatch):
    import hashlib
    from src.compiler.project_compilation_product import compile_project_product
    from src.compiler.ssa_fortran_backend import fortran_compiler
    if fortran_compiler() is None:
        pytest.skip('native Fortran compiler unavailable')
    monkeypatch.setenv('TURING_COMPILER_BOOTSTRAP_REGISTRY', str(tmp_path / 'registry.json'))
    monkeypatch.delenv('TURING_COMPILER_BOOTSTRAP_PRODUCTS', raising=False)
    probes = [[12345, n] for n in (0, 7, 1000000)]
    cases = {
        'leaf': {'probes': probes, 'expected_probe_results': [matrix_golden(*probe) for probe in probes]},
        'linked_checksum': {'probes': probes, 'expected_probe_results': [matrix_golden(*probe) + 1 for probe in probes]},
    }
    manifest = compile_project_product(
        ROOT / 'src/compiler/compiler_benchmark_consumer.py', tmp_path / 'consumer',
        entries=tuple(cases), jobs=1, emit_native=True, verification_cases=cases,
    )
    assert all(record['verified'] for record in manifest['native_case_verification']), manifest['native_case_verification']
    root = tmp_path / 'consumer'
    records = {record['qualified_name']: record for record in manifest['units']}
    receipt = json.loads((root / records['linked_checksum']['path'] / 'unit.json').read_text())
    linked = [item for item in receipt['linked_repository_dependencies'] if item['qualified_name'] == 'leaf']
    assert len(linked) == 1
    accepted = Path(linked[0]['artifact'])
    leaf_receipt = json.loads((root / records['leaf']['path'] / 'unit.json').read_text())
    leaf_artifact = root / records['leaf']['path'] / leaf_receipt['artifact']
    assert accepted.resolve() == leaf_artifact.resolve()
    digest = hashlib.sha256(accepted.read_bytes()).hexdigest()
    assert digest == hashlib.sha256(leaf_artifact.read_bytes()).hexdigest()
    print('STATIC_SSA_LIBRARY_REUSE', accepted, digest)
    assert len(digest) == 64
    for record in manifest['native_case_verification']:
        assert record['native_probe_count'] == 3 and record['fallback_probe_count'] == 0


def test_unsupported_span_abi_refuses_before_native_invocation(native_product, monkeypatch):
    import copy
    from src.compiler.compiled_program_api import load_api
    link = native_product.links['recurrence']
    descriptor = copy.deepcopy(load_api(native_product.root / link['native_api']))
    entry = next(item for item in descriptor['entry_points']
                 if item['name'] == link['native_entrypoint'])
    entry['parameters'][0]['shape'] = [3]
    specification = {'probes': [[12345, 7]], 'expected_probe_results': [matrix_golden(12345, 7)]}
    with monkeypatch.context() as changed:
        changed.setattr('src.compiler.compiled_program_api.load_api', lambda _: descriptor)
        report = verify_project_product_cases(native_product, {'recurrence': specification})[0]
        assert not report['verified']
        assert 'unsupported native case ABI' in report['reason']
    assert verify_project_product_cases(native_product, {'recurrence': specification})[0]['verified']


def test_scalar_cli_with_private_record_workspace_uses_published_boundary(tmp_path, monkeypatch):
    import os
    from src.compiler.compiled_program_api import load_api
    from src.compiler.compiler_bootstrap_runtime import _activate_descriptor_call
    from src.compiler.ssa_fortran_backend import fortran_compiler
    if fortran_compiler() is None:
        pytest.skip('native Fortran compiler unavailable')
    monkeypatch.setenv('TURING_COMPILER_BOOTSTRAP_REGISTRY', str(tmp_path / 'registry.json'))
    monkeypatch.delenv('TURING_COMPILER_BOOTSTRAP_PRODUCTS', raising=False)
    source = ROOT / 'src/compiler/compiler_counter_benchmark_consumer.py'
    cases = ROOT / 'tools/compiler_probes/native_counter_cases.json'
    directory = tmp_path / 'counter-product'
    completed = subprocess.run([
        sys.executable, '-u', '-m', 'tools.compile_project_catalogue',
        '--source', str(source), '--entry', 'counter_checksum', '--output', str(directory),
        '--jobs', '1', '--emit-native', '--verification-cases', str(cases),
    ], cwd=ROOT, env=os.environ.copy(), text=True, capture_output=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    print(completed.stdout.splitlines()[-1])
    product = open_project_compilation_product(directory)
    link = product.links['counter_checksum']
    api = load_api(directory / link['native_api'])
    entry = next(item for item in api['entry_points'] if item['name'] == link['native_entrypoint'])
    parameters = entry['parameters']
    assert [item['source_name'] for item in parameters if item['role'] == 'input'] == ['rounds']
    workspaces = {item['name'] for item in parameters if item['role'] == 'workspace'}
    records = api['metadata']['record_tables'][entry['name']]
    assert records and all(f't{value}' in workspaces for record in records
                           for field in record['fields'] for value in field['value_ids'])
    report = json.loads((directory / 'native-case-verification.json').read_text())['units'][0]
    assert report['verified'] and report['abi_kind'] == 'scalar'
    assert report['native_probe_count'] == 3 and report['fallback_probe_count'] == 0
    _, deployed = _activate_descriptor_call(product, 'counter_checksum')
    for rounds in (7, 0, 7, 17):
        assert deployed(rounds) == rounds * (rounds - 1) // 2
    assert deployed.__turing_native_route_counts__ == {'native': 7, 'fallback': 0}
    import copy
    unbound = copy.deepcopy(api)
    unbound_entry = next(item for item in unbound['entry_points'] if item['name'] == entry['name'])
    old_extent = next(item['name'] for item in unbound_entry['parameters'] if item['role'] == 'extent')
    for parameter in unbound_entry['parameters']:
        if parameter['name'] == old_extent:
            parameter['name'] = 'extent_private_unbound'
        if parameter.get('extent') == old_extent:
            parameter['extent'] = 'extent_private_unbound'
        if parameter.get('extents'):
            parameter['extents'] = ['extent_private_unbound' if name == old_extent else name
                                    for name in parameter['extents']]
    specification = json.loads(cases.read_text())
    with monkeypatch.context() as changed:
        changed.setattr('src.compiler.compiled_program_api.load_api', lambda _: unbound)
        rejected = verify_project_product_cases(product, specification)[0]
        assert not rejected['verified']
        assert 'extent_overrides' in rejected['reason']
        receipt = Path(rejected['library']).parent / 'native-verification.json'
        assert json.loads(receipt.read_text())['status'] == 'failed'
    assert verify_project_product_cases(product, specification)[0]['verified']
