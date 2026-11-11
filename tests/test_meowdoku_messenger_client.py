"""Offline client regression checks: no production accounts or transport involved."""
import copy
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples/meowdoku/deploy'))
import agent_messenger as client


def args():
    return client.build_parser().parse_args(['--base', 'https://example.test', '--user', 'ann', 'send', 'BEN', 'hello'])


@pytest.mark.parametrize('base', ['http://example.test', 'https://user:pass@example.test', 'https://example.test?secret=x'])
def test_https_validated_before_login(base, monkeypatch):
    monkeypatch.setattr(client.Client, '_login', lambda *a: pytest.fail('login attempted'))
    with pytest.raises(client.ClientError):
        client.Client(base, 'ann', 'not-a-real-secret')


def test_redirect_refused():
    with pytest.raises(client.ClientError, match='redirect refused'):
        client._NoRedirect().redirect_request(None, None, 307, '', {}, 'http://elsewhere.test')


def test_malformed_success_json_is_uncertain():
    c = client.Client.__new__(client.Client)
    c.base = 'https://example.test'
    c.opener = SimpleNamespace(open=lambda *a, **k: io.BytesIO(b'not json'))
    with pytest.raises(client.ClientError) as e:
        c.call('/api/messages/send', {'body': 'hello'})
    assert e.value.uncertain


class Fake:
    def __init__(self, receipt=None, fail_refresh=False):
        self.calls = []
        self.issued = 0
        self.receipt = receipt or {'ok': True, 'id': 12, 'created': 1.0}
        self.fail_refresh = fail_refresh

    def call(self, path, body=None):
        self.calls.append((path, copy.deepcopy(body)))
        if path.endswith('/conversations'):
            self.issued += 1
            if self.fail_refresh and self.issued > 1:
                raise client.ClientError('session expired', status=403)
            return {'contacts': [{'username': 'ben', 'conversation_handle': 'opaque-' + str(self.issued)}]}
        if len([p for p, b in self.calls if p.endswith('/send')]) == 1:
            raise client.ClientError('lost response', uncertain=True)
        return self.receipt


def test_retry_refreshes_handle_keeps_operation_and_omits_name(capsys):
    c = Fake()
    assert client.run(c, args()) == 0
    sent = [b for p, b in c.calls if p.endswith('/send')]
    assert sent[0]['idempotency_key'] == sent[1]['idempotency_key']
    assert sent[0]['conversation_handle'] != sent[1]['conversation_handle']
    assert all(set(b) == {'body', 'idempotency_key', 'conversation_handle'} for b in sent)
    assert json.loads(capsys.readouterr().out)['id'] == 12


def test_prior_uncertain_send_survives_definitive_refresh_failure():
    c = Fake(fail_refresh=True)
    with pytest.raises(client.ClientError) as e:
        client.run(c, args())
    key = next(b['idempotency_key'] for p, b in c.calls if p.endswith('/send'))
    assert e.value.uncertain and key in str(e.value)


@pytest.mark.parametrize('receipt', [{'ok': True}, {'ok': True, 'id': 0, 'created': 1}, {'ok': True, 'id': 2, 'created': float('nan')}])
def test_invalid_send_receipt_keeps_retry_key(receipt):
    c = Fake(receipt=receipt)
    with pytest.raises(client.ClientError, match='idempotency-key'):
        client.run(c, args())


def test_widget_offline_regressions():
    result = subprocess.run(['node', str(ROOT / 'tests/meowdoku_messenger_client.mjs')], cwd=ROOT,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


def test_truncated_response_retries_with_same_operation_key(capsys):
    class Truncated(io.BytesIO):
        def read(self, *a, **k):
            raise client.http.client.IncompleteRead(b'{"ok":', 40)

    c = client.Client.__new__(client.Client)
    c.base = 'https://example.test'
    calls = []
    def open_response(request, **kwargs):
        body = json.loads(request.data) if request.data else None
        calls.append((request.full_url, body))
        if request.full_url.endswith('/conversations'):
            return io.BytesIO(json.dumps({'contacts': [{'username': 'ben', 'conversation_handle': 'opaque'}]}).encode())
        sends = [b for p, b in calls if p.endswith('/send')]
        if len(sends) == 1:
            return Truncated()
        return io.BytesIO(b'{"ok":true,"id":13,"created":1.0}')
    c.opener = SimpleNamespace(open=open_response)
    assert client.run(c, args()) == 0
    sends = [b for p, b in calls if p.endswith('/send')]
    assert len(sends) == 2
    assert sends[0]['idempotency_key'] == sends[1]['idempotency_key']
    assert json.loads(capsys.readouterr().out)['id'] == 13


def test_truncated_response_final_failure_preserves_key():
    class Truncated(io.BytesIO):
        def read(self, *a, **k):
            raise client.http.client.IncompleteRead(b'{', 10)
    c = client.Client.__new__(client.Client)
    c.base = 'https://example.test'
    sent = []
    def open_response(request, **kwargs):
        if request.full_url.endswith('/conversations'):
            return io.BytesIO(b'{"contacts":[{"username":"ben","conversation_handle":"opaque"}]}')
        sent.append(json.loads(request.data))
        return Truncated()
    c.opener = SimpleNamespace(open=open_response)
    with pytest.raises(client.ClientError) as e:
        client.run(c, args())
    assert e.value.uncertain
    assert len(sent) == 2 and sent[0]['idempotency_key'] == sent[1]['idempotency_key']
    assert sent[0]['idempotency_key'] in str(e.value)


@pytest.mark.parametrize('bad', [None, {}, {'contacts': None}, {'contacts': [None]}, {'contacts': [{'username': 9}]}, {'contacts': [{'username': 'ben'}]}])
def test_malformed_refresh_after_uncertain_send_preserves_key(bad):
    class Malformed(Fake):
        def call(self, path, body=None):
            if path.endswith('/conversations') and self.issued:
                return bad
            return super().call(path, body)
    c = Malformed()
    with pytest.raises(client.ClientError) as e:
        client.run(c, args())
    key = next(b['idempotency_key'] for p, b in c.calls if p.endswith('/send'))
    assert e.value.uncertain and key in str(e.value)


def test_truncated_error_body_keeps_http_status_and_uncertainty():
    class Truncated(io.BytesIO):
        def read(self, *a, **k):
            raise client.http.client.IncompleteRead(b'{', 10)
    def fail(*a, **k):
        raise client.urllib.error.HTTPError('https://example.test/api/messages/send', 500, 'failed', {}, Truncated())
    c = client.Client.__new__(client.Client)
    c.base = 'https://example.test'
    c.opener = SimpleNamespace(open=fail)
    with pytest.raises(client.ClientError) as e:
        c.call('/api/messages/send', {'body': 'hello'})
    assert e.value.status == 500 and e.value.uncertain
