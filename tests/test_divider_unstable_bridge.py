"""Permission and source-only receipt proofs. No compiler is invoked."""
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'examples')]
from meowdoku import authz, compile_artifacts as ca, divider_nodus_bridge as br
from meowdoku import source_divider as sd, workspace_accounting as wa
from meowdoku.state_paths import StatePaths
from meowdoku.site_compiler import SiteCompiler

SOURCE = 'def program(*shader_pairs: tuple[str, int]) -> int:\n    return 0\n'


@pytest.fixture
def setup(tmp_path):
    state = StatePaths(tmp_path / 'state')
    db = sd.connect(state.accounts_db)
    db.execute('CREATE TABLE users(id INTEGER PRIMARY KEY, username TEXT)')
    db.execute("INSERT INTO users VALUES(1, 'owner')")
    db.execute('INSERT INTO divider_jobs VALUES(?,?,?,?,?,?,?)',
               ('job', 'account:1', '', '', time.time(), 'running', 'c'))
    db.execute("INSERT INTO divider_attempts(job_id,piece_id,backend,state,lease_owner,lease_generation,lease_deadline) "
               "VALUES('job',0,'c','leased','worker',1,?)", (time.time() + 60,))
    claim = dict(job_id='job', piece_id=0, backend='c', worker_id='worker', generation=1,
                 names=['program'], file='original.py')
    source = state.jobs_dir / 'job/source/original.py'
    source.parent.mkdir(parents=True)
    source.write_text(SOURCE, encoding='utf-8')
    receipt = state.jobs_dir / 'divider/job/upload.json'
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({'unstable_source_only': True}))
    contract = tmp_path / 'contract.yaml'
    contract.write_text('program_abi:\n  values: []\n# preserve caller contract exactly\n')
    root = tmp_path / 'account-workspaces'
    root.mkdir()
    store = wa.Store(state.accounts_db, root, capacity_bytes=1_000_000,
                     user_quota_bytes=500_000, free_floor_bytes=0)
    artifacts = ca.Artifacts(state, store, workspace_root=root, reservation_bytes=10_000, review_after_s=60)
    policy = copy.deepcopy(authz.DEFAULT_POLICY)
    policy['role_caps']['member'].append('compile_unstable')
    x = SimpleNamespace(state=state, db=db, claim=claim, source=source, receipt=receipt,
                        contract=contract, root=root, store=store, artifacts=artifacts, policy=policy)
    try:
        yield x
    finally:
        db.close()


def request(x, **kwargs):
    opts = dict(jobs_dir=x.root, source_root=x.state.jobs_dir, extraction_contract=x.contract,
                artifacts=x.artifacts, allow_unstable_source=True,
                authz_policy=SimpleNamespace(get=lambda: x.policy))
    opts.update(kwargs)
    return br.request_for_piece(x.db, x.claim, **opts)


def test_non_tensor_source_preserves_source_and_contract(setup):
    x = setup
    identifier, path = request(x)
    req = json.loads(path.read_text())
    assert req['unstable_source_only'] is True
    assert req['source_names'] == ['program']
    assert req['source_text'] == x.source.read_bytes().decode('utf-8')
    assert x.source.read_text() == SOURCE
    assert Path(req['extraction_contract']).read_bytes() == x.contract.read_bytes()
    assert not {'row_entry', 'columns', 'product_name', 'allow_uncompiled', 'root_calls'} & req.keys()
    assert req['site_artifact_context']['owner_id'] == 1
    assert len(x.store.list_owned(1)) == 1


@pytest.mark.parametrize('change', ['operator', 'grant', 'submit', 'owner', 'deleted', 'role'])
def test_current_permissions_rechecked_before_artifact_admission(setup, change):
    x = setup
    opts = {}
    if change == 'operator':
        opts['allow_unstable_source'] = False
    elif change == 'grant':
        x.policy['role_caps']['member'].remove('compile_unstable')
    elif change == 'submit':
        x.policy['role_caps']['member'].remove('compile_submit')
    elif change == 'owner':
        x.db.execute("UPDATE divider_jobs SET owner_key='browser:fake'")
    elif change == 'deleted':
        x.db.execute('DELETE FROM users')
    else:
        authz.role_of(x.state.accounts_db, 1)  # initialize incumbent role table
        x.db.execute("INSERT INTO account_roles VALUES(1, 'operator', 'test', ?)", (time.time(),))
    assert request(x, **opts) is None
    row = br._attempt_row(x.db, x.claim)
    assert row['state'] == 'refused' and row['error_type'] == 'UnstablePermissionDenied'
    assert not list(x.root.iterdir())
    assert x.db.execute('SELECT COUNT(*) FROM compile_artifact_runs').fetchone()[0] == 0


def test_no_opt_in_retains_existing_tensor_restriction(setup):
    x = setup
    x.receipt.write_text('{}')
    assert request(x) is None
    assert br._attempt_row(x.db, x.claim)['error_type'] == 'RequestNotExpressible'
    assert not list(x.root.iterdir())


@pytest.mark.parametrize('value', ['true', 1, None])
def test_malformed_receipt_cannot_enable_unstable(setup, value):
    x = setup
    x.receipt.write_text(json.dumps({'unstable_source_only': value}))
    assert request(x) is None
    assert not list(x.root.iterdir())


@pytest.mark.parametrize('metadata', [[], {'unstable_source_only': True, 'unstable_allow_inference': 'true'},
                                    {'unstable_source_only': False, 'unstable_allow_inference': True}])
def test_inference_receipt_requires_source_mode_and_boolean(setup, metadata):
    x = setup
    x.receipt.write_text(json.dumps(metadata))
    assert request(x) is None
    assert br._attempt_row(x.db, x.claim)['error_type'] == 'RequestNotExpressible'
    assert not list(x.root.iterdir())


def outcome(x, status=4):
    expected = dict(backend='c', source_names=['program'], site_artifact_context={'owner_id': 1},
                    authored_sha256=hashlib.sha256(SOURCE.encode()).hexdigest())
    product = dict(schema='nodus-turing-source-only-product-v1', unstable_source_only=True,
                   state='emitted-not-built', loadable=False, binary=None, adoption_receipt=None,
                   backend='c', source_names=['program'], site_artifact_context={'owner_id': 1},
                   source={'authored_sha256': expected['authored_sha256']},
                   attempt={'backend': 'c', 'entries': [dict(name='program', state='complete', note='emitted-not-built')]})
    return br.CompileOutcome(status, br.STATUS_NAMES[status], True, None,
                             dict(unstable_source_only=True, source_expectation=expected, source_product=product))


def test_emitted_not_built_record_is_explicit(setup):
    x = setup
    assert br._record(x.db, x.claim, outcome(x), .1) == 'complete'
    assert 'emitted-not-built (unstable source-only)' in br._attempt_row(x.db, x.claim)['stage']


@pytest.mark.parametrize('change', ['loadable', 'binary', 'buffer_abi', 'schema', 'names', 'hash', 'owner', 'attempt', 'entry', 'status'])
def test_mismatched_or_malformed_receipt_never_completes(setup, change):
    x = setup
    result = outcome(x)
    product = result.detail['source_product']
    if change == 'loadable': product['loadable'] = True
    elif change == 'binary': product['binary'] = 'fake.dll'
    elif change == 'buffer_abi': product['buffer_abi'] = {'fake': True}
    elif change == 'schema': product['schema'] = 'wrong'
    elif change == 'names': product['source_names'] = ['other']
    elif change == 'hash': product['source']['authored_sha256'] = '0' * 64
    elif change == 'owner': product['site_artifact_context']['owner_id'] = 2
    elif change == 'attempt': product['attempt'] = ['not-a-record']
    elif change == 'entry': product['attempt']['entries'] = ['not-a-record']
    else: result.status_code = 2
    assert br._record(x.db, x.claim, result, .1) == 'error'
    assert br._attempt_row(x.db, x.claim)['error_type'] == 'SourceOnlyReceiptMismatch'


def test_actual_compiler_diagnostic_retained(setup):
    x = setup
    result = outcome(x, 17)
    result.detail['source_product']['state'] = 'refused'
    result.detail['source_product']['attempt']['entries'] = [dict(name='program', state='error',
        stage='ssa-lowering', error_type='NotImplementedError', error='unsupported object call')]
    assert br._record(x.db, x.claim, result, .1) == 'error'
    row = br._attempt_row(x.db, x.claim)
    assert row['error_type'] == 'NotImplementedError' and row['error'] == 'unsupported object call'
    assert row['stage'].endswith('ssa-lowering')


def test_attempt_forwards_selected_contract_without_compiler(tmp_path, monkeypatch):
    source = tmp_path / 'source.py'
    source.write_text(SOURCE)
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps(dict(source_file=str(source), names=['program'], backend='c',
                                  out_dir=str(tmp_path), extraction_contract='selected-contract.yaml')))
    seen = []
    def fake(text, name, backend, out, mark, **options):
        seen.append((text, options))
        return {'name': name, 'state': 'complete'}
    monkeypatch.setattr(sd, '_attempt_entry', fake)
    assert sd._attempt_main(str(spec)) == 0
    assert seen == [(SOURCE, {'extraction_contract': 'selected-contract.yaml'})]


def test_operator_config_requires_real_boolean_before_files(setup):
    x = setup
    target = x.root / 'not-created'
    config = dict(jobs_dir=str(target), service_id=7, port=1234, backend='c', capacity_bytes=1000,
                  user_quota_bytes=1000, free_floor_bytes=0, reservation_bytes=100, review_after_s=60,
                  allow_unstable_source='true')
    with pytest.raises(ValueError, match='must be a boolean'):
        SiteCompiler(x.state, SimpleNamespace(_conn=x.db), config)
    assert not target.exists()


def test_revocation_after_connect_prevents_submit(setup, monkeypatch):
    x = setup
    closed = []
    def connect(route, producer):
        x.policy['role_caps']['member'].remove('compile_unstable')
        return SimpleNamespace(submit=lambda _: pytest.fail('revoked contract was submitted'),
                               close=lambda: closed.append(True))
    monkeypatch.setattr(br, 'load_peer_module', lambda _: object())
    route = br.NodusCompileRoute('127.0.0.1', 1, 7, x.root)
    with pytest.raises(br.UnstablePermissionDenied):
        br.submit_and_wait(route, 42, timeout_s=1, peer_factory=connect,
            authorize=lambda: br._authorize_unstable(x.db, 'job', True, SimpleNamespace(get=lambda: x.policy)))
    assert closed == [True]
