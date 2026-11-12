"""CPU-only policy, fake HTTP and metadata proofs; no compiler or worker runs."""
import copy
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'examples')]
from meowdoku import authz, divider_intake as di, server


def policy(grant=False, tier=False):
    result = copy.deepcopy(authz.DEFAULT_POLICY)
    if grant:
        result['role_caps']['member'].append('compile_unstable')
    if tier:
        for caps in result['tier_caps'].values():
            caps.append('compile_unstable')
    return result


@pytest.mark.parametrize('role', authz.ROLES)
@pytest.mark.parametrize('tier', authz.TIERS)
def test_defaults_and_tiers_never_grant_unstable(role, tier):
    p = authz.Principal(role, None if role == 'guest' else 1, 'actor', 'lan', tier)
    for pol in (policy(), policy(tier=True)):
        assert 'compile_unstable' not in authz.capabilities(p, pol)


def test_explicit_role_grant_requires_account(tmp_path):
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps(policy(grant=True, tier=True)))
    pol = authz.load_policy(path)
    for role, uid, allowed in [('member', 1, True), ('admin', 2, False), ('member', None, False), ('guest', None, False)]:
        p = authz.Principal(role, uid, 'actor', 'lan', 'edit')
        assert ('compile_unstable' in authz.capabilities(p, pol)) is allowed
    pol['role_caps']['guest'].append('compile_unstable')
    assert 'compile_unstable' not in authz.capabilities(authz.Principal('guest', None, None, 'lan', 'edit'), pol)


class Hook:
    backends = ('wasm',)
    def __init__(self, enabled):
        self.allow_unstable_source = enabled
        self.jobs = []
    def __call__(self, job_id):
        self.jobs.append(job_id)


def request(tmp_path, *, grant=False, tier=False, uid=1, enabled=False, opt='source-only', inference=None, ctype='application/octet-stream', submit=True):
    sessions = SimpleNamespace(web_dir=tmp_path, state=tmp_path, divider_hook=Hook(enabled))
    handler_type = server.make_handler(sessions)
    handler = handler_type.__new__(handler_type)
    principal = authz.Principal('member' if uid is not None else 'guest', uid, 'actor', 'lan', 'edit')
    pol = policy(grant, tier)
    if not submit:
        pol['role_caps']['member'].remove('compile_submit')
    handler._principal = lambda: principal
    handler._caps = lambda: authz.capabilities(principal, pol)
    handler._classify = lambda: None
    sessions.authz_policy = SimpleNamespace(get=lambda: pol)
    handler._divider_who = lambda: ('account:1', False)
    handler._json = lambda body, status=200: (int(status), body)
    handler.headers = {'Content-Type': ctype, 'Content-Length': '22', 'X-Filename': 'source.py'}
    if opt is not None:
        handler.headers['X-Compile-Unstable'] = opt
    if inference is not None:
        handler.headers['X-Compile-Inference'] = inference
    handler.rfile = io.BytesIO(b'def f(x):\n    return x\n')
    handler.headers['Content-Length'] = str(len(handler.rfile.getvalue()))
    handler._body_json = lambda limit: pytest.fail('must reject before reading JSON')
    return handler, sessions


@pytest.mark.parametrize('kwargs,status', [
    ({'inference': 'allow'}, 403),
    ({'inference': 'allow', 'enabled': True}, 403),
    ({'inference': 'allow', 'grant': True}, 403),
    ({'inference': 'allow', 'opt': None, 'grant': True, 'enabled': True}, 400),
    ({'inference': '', 'grant': True, 'enabled': True}, 400),
    ({'inference': 'yes', 'grant': True, 'enabled': True}, 400),
    ({'inference': 'allow', 'grant': True, 'enabled': True, 'ctype': 'application/json'}, 400),
    ({}, 403), ({'grant': True}, 403), ({'grant': True, 'enabled': 'true'}, 403),
    ({'enabled': True}, 403), ({'tier': True, 'enabled': True}, 403),
    ({'grant': True, 'enabled': True, 'uid': None}, 403),
    ({'grant': True, 'enabled': True, 'opt': ''}, 400),
    ({'grant': True, 'enabled': True, 'opt': 'yes'}, 400),
    ({'grant': True, 'enabled': True, 'ctype': 'application/json'}, 400),
])
def test_refused_before_body_or_intake(tmp_path, monkeypatch, kwargs, status):
    h, sessions = request(tmp_path, **kwargs)
    monkeypatch.setattr(di, 'accept_upload', lambda *a, **kw: pytest.fail('intake called'))
    monkeypatch.setattr(di, 'accept_repo_link', lambda *a, **kw: pytest.fail('repo intake called'))
    responses = []
    h._json = lambda body, status=200: responses.append((int(status), body))
    h._divider_post('/api/divider/jobs')
    assert responses[0][0] == status
    assert h.rfile.tell() == 0
    assert not sessions.divider_hook.jobs
    assert not list(tmp_path.iterdir())


def test_compile_submit_still_required(tmp_path, monkeypatch):
    h, _ = request(tmp_path, grant=True, enabled=True, submit=False)
    responses = []
    h._json = lambda body, status=200: responses.append((int(status), body))
    monkeypatch.setattr(di, 'accept_upload', lambda *a, **kw: pytest.fail('intake called'))
    h._divider_post('/api/divider/jobs')
    assert responses == [(403, {'error': 'this needs compile_submit'})]
    assert h.rfile.tell() == 0


@pytest.mark.parametrize('opt,grant,enabled,expected', [(None, False, False, False), ('source-only', True, True, True)])
def test_accepted_metadata_and_owner_isolation(tmp_path, opt, grant, enabled, expected):
    h, sessions = request(tmp_path, opt=opt, grant=grant, enabled=enabled)
    status, job = h._divider_post('/api/divider/jobs')
    assert status == 201
    assert job['unstable_source_only'] is expected
    metadata = json.loads((tmp_path / 'jobs' / 'divider' / job['job_id'] / 'upload.json').read_text())
    assert metadata['unstable_source_only'] is expected
    assert di.get_job(tmp_path, job['job_id'], 'account:other') is None
    assert di.get_job(tmp_path, job['job_id'], 'account:1')['unstable_source_only'] is expected
    assert all(p['state'] == 'needs-annotation' for p in job['pieces_detail'])
    assert sessions.divider_hook.jobs == [job['job_id']]


def test_default_internal_upload_is_stable_and_legacy_detail_is_stable(tmp_path):
    job_id = di.accept_upload(tmp_path, 'account:1', 'source.py', b'def f(x):\n    return x\n')
    assert di.get_job(tmp_path, job_id, 'account:1')['unstable_source_only'] is False
    path = tmp_path / 'jobs' / 'divider' / job_id / 'upload.json'
    metadata = json.loads(path.read_text())
    del metadata['unstable_source_only']
    path.write_text(json.dumps(metadata))
    assert di.get_job(tmp_path, job_id, 'account:1')['unstable_source_only'] is False


def test_ui_opt_in_default_and_headers():
    html = (ROOT / 'examples/meowdoku/web/divider.html').read_text()
    checkbox = re.search(r'<input[^>]*id="unstable"[^>]*>', html).group()
    assert 'type="checkbox"' in checkbox and 'checked' not in checkbox
    inference = re.search(r'<input[^>]*id="inference"[^>]*>', html).group()
    assert 'type="checkbox"' in inference and 'checked' not in inference
    # Exercise the actual form listener with a fake DOM and fetch, not a browser or renderer.
    script = r'''
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const calls = [], nodes = {};
for (const id of ['fileform','repoform','file','unstable','inference','cancel','status','gate']) nodes[id] = {addEventListener: (name, fn) => nodes[id][name] = fn};
nodes.file.files = [{name: 'test.py'}];
const context = {document: {getElementById: id => nodes[id], addEventListener() {}}, fetch: async (path, options) => {calls.push(options); return {ok:false,status:403,json:async()=>({error:'test'})};}, console};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8').replace('export function', 'function'), context);
vm.runInContext('init()', context);
(async () => {
  for (const [checked, infer] of [[false,false], [true,false], [true,true], [true,false], [false,false]]) {
    nodes.unstable.checked = checked;
    nodes.inference.checked = infer;
    await nodes.fileform.submit({preventDefault(){}});
    const headers = calls[calls.length-1].headers;
    assert.strictEqual(headers['X-Compile-Unstable'], checked ? 'source-only' : undefined);
    assert.strictEqual(headers['X-Compile-Inference'], infer ? 'allow' : undefined);
  }
  const before = calls.length;
  nodes.inference.checked = true;
  nodes.unstable.checked = false;
  await nodes.fileform.submit({preventDefault(){}});
  assert.strictEqual(calls.length, before);
})().catch(e => { console.error(e); process.exitCode=1; });
'''
    subprocess.run(['node', '-e', script, str(ROOT / 'examples/meowdoku/web/divider.js')], check=True, timeout=10)


def test_privileged_inference_upload_admitted_unchanged(tmp_path):
    h, sessions = request(tmp_path, grant=True, enabled=True, inference='allow')
    source = h.rfile.getvalue()
    status, job = h._divider_post('/api/divider/jobs')
    assert status == 201
    assert job['unstable_source_only'] is True and job['unstable_allow_inference'] is True
    assert job['pieces_detail']
    assert all(not p['annotated'] and p['state'] == 'queued' for p in job['pieces_detail'])
    assert (tmp_path / 'jobs' / job['job_id'] / 'source' / 'source.py').read_bytes() == source
    metadata = json.loads((tmp_path / 'jobs' / 'divider' / job['job_id'] / 'upload.json').read_text())
    assert metadata['unstable_allow_inference'] is True
    assert di.get_job(tmp_path, job['job_id'], 'account:other') is None


@pytest.mark.parametrize('source,inference', [(False, True), ('true', True), (True, 'true'), (True, 1), (1, False), (True, None)])
def test_invalid_intake_options_refuse_before_write(tmp_path, source, inference):
    with pytest.raises(di.DividerError):
        di.accept_upload(tmp_path, 'account:1', 'm.py', b'def f(x): return x',
                         unstable_source_only=source, unstable_allow_inference=inference)
    assert not list(tmp_path.iterdir())
